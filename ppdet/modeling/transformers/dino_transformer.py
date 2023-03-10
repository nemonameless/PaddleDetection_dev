# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Modified from detrex (https://github.com/IDEA-Research/detrex)
# Copyright 2022 The IDEA Authors. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register
from ..layers import MultiHeadAttention
from .position_encoding import PositionEmbedding
from ..heads.detr_head import MLP
from .deformable_transformer import (MSDeformableAttention,
                                     DeformableTransformerEncoderLayer,
                                     DeformableTransformerEncoder)
from ..initializer import (linear_init_, constant_, xavier_uniform_, normal_,
                           bias_init_with_prob)
from .utils import (_get_clones, get_valid_ratio,
                    get_contrastive_denoising_training_group,
                    get_sine_pos_embed, inverse_sigmoid)

__all__ = ['DINOTransformer']


class DINOTransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,
                 lr_mult=1.0,
                 weight_attr=None,
                 bias_attr=None):
        super(DINOTransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(
            d_model, weight_attr=weight_attr, bias_attr=bias_attr)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels,
                                                n_points, lr_mult)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(
            d_model, weight_attr=weight_attr, bias_attr=bias_attr)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(
            d_model, weight_attr=weight_attr, bias_attr=bias_attr)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = paddle.where(
                attn_mask.astype('bool'),
                paddle.zeros(attn_mask.shape, tgt.dtype),
                paddle.full(attn_mask.shape, float("-inf"), tgt.dtype))
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class DINOTransformerDecoder(nn.Layer):
    def __init__(self,
                 hidden_dim,
                 decoder_layer,
                 num_layers,
                 weight_attr=None,
                 bias_attr=None):
        super(DINOTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(
            hidden_dim, weight_attr=weight_attr, bias_attr=bias_attr)

    def forward(self,
                tgt,
                ref_points_unact, # refpoints_unsigmoid [nq, bs, 2/4]
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                query_pos_head,
                valid_ratios=None,
                attn_mask=None,
                memory_mask=None):
        if valid_ratios is None:
            valid_ratios = paddle.ones(
                [memory.shape[0], memory_spatial_shapes.shape[0], 2])

        output = tgt
        intermediate = []
        inter_bboxes = []
        ref_points = F.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.layers):
            reference_points_input = ref_points.detach().unsqueeze(
                2) * valid_ratios.tile([1, 1, 2]).unsqueeze(1)
            query_pos_embed = get_sine_pos_embed(
                reference_points_input[..., 0, :], self.hidden_dim // 2)
            query_pos_embed = query_pos_head(query_pos_embed)

            output = layer(output, reference_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            ref_points = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(
                ref_points.detach()))

            intermediate.append(self.norm(output))
            inter_bboxes.append(ref_points)

        return paddle.stack(intermediate), paddle.stack(inter_bboxes)


@register
class DINOTransformer(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'for_distill']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=900,
                 position_embed_type='sine',
                 in_feats_channel=[512, 1024, 2048],
                 num_levels=4,
                 num_encoder_points=4,
                 num_decoder_points=4,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 lr_mult=1.0,
                 pe_temperature=10000,
                 pe_offset=-0.5,
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=True,
                 for_distill=False,
                 eps=1e-2):
        super(DINOTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(in_feats_channel) <= num_levels

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers

        weight_attr = ParamAttr(regularizer=L2Decay(0.0))
        bias_attr = ParamAttr(regularizer=L2Decay(0.0))
        # backbone feature projection
        self._build_input_proj_layer(in_feats_channel, weight_attr, bias_attr)

        # Transformer module
        encoder_layer = DeformableTransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_encoder_points, lr_mult, weight_attr, bias_attr)
        self.encoder = DeformableTransformerEncoder(encoder_layer,
                                                    num_encoder_layers)
        decoder_layer = DINOTransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_decoder_points, lr_mult, weight_attr, bias_attr)
        self.decoder = DINOTransformerDecoder(hidden_dim, decoder_layer,
                                              num_decoder_layers, weight_attr,
                                              bias_attr)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # position embedding
        self.position_embedding = PositionEmbedding(
            hidden_dim // 2,
            temperature=pe_temperature,
            normalize=True if position_embed_type == 'sine' else False,
            embed_type=position_embed_type,
            offset=pe_offset)
        self.level_embed = nn.Embedding(num_levels, hidden_dim)
        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(2 * hidden_dim,
                                  hidden_dim,
                                  hidden_dim,
                                  num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim, weight_attr=weight_attr, bias_attr=bias_attr))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        # decoder head
        self.dec_score_head = nn.LayerList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.LayerList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # detr distill
        self.for_distill = for_distill
        if for_distill:
            self.distill_pairs = dict()

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight)
        constant_(self.enc_bbox_head.layers[-1].bias)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight)
            constant_(reg_.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        normal_(self.level_embed.weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)
            constant_(l[0].bias)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_feats_channel': [i.channels for i in input_shape], }

    def _build_input_proj_layer(self,
                                in_feats_channel,
                                weight_attr=None,
                                bias_attr=None):
        self.input_proj = nn.LayerList()
        for in_channels in in_feats_channel:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels, self.hidden_dim, kernel_size=1)), (
                            'norm', nn.GroupNorm(
                                32,
                                self.hidden_dim,
                                weight_attr=weight_attr,
                                bias_attr=bias_attr))))
        in_channels = in_feats_channel[-1]
        for _ in range(self.num_levels - len(in_feats_channel)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1)), ('norm', nn.GroupNorm(
                            32,
                            self.hidden_dim,
                            weight_attr=weight_attr,
                            bias_attr=bias_attr))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats, pad_mask=None):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios = []
        for i, feat in enumerate(proj_feats):
            bs, _, h, w = paddle.shape(feat)
            spatial_shapes.append(paddle.concat([h, w]))
            # [b,c,h,w] -> [b,h*w,c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            if pad_mask is not None:
                mask = F.interpolate(pad_mask.unsqueeze(0), size=(h, w))[0]
            else:
                mask = paddle.ones([bs, h, w])
            valid_ratios.append(get_valid_ratio(mask))
            # [b, h*w, c]
            pos_embed = self.position_embedding(mask).flatten(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed.weight[i]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            if pad_mask is not None:
                # [b, h*w]
                mask_flatten.append(mask.flatten(1))

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        # [b, l]
        mask_flatten = None if pad_mask is None else paddle.concat(mask_flatten,
                                                                   1)
        # [b, l, c]
        lvl_pos_embed_flatten = paddle.concat(lvl_pos_embed_flatten, 1)
        # [num_levels, 2]
        spatial_shapes = paddle.to_tensor(
            paddle.stack(spatial_shapes).astype('int64'))
        # [l] start index of each level
        level_start_index = paddle.concat([
            paddle.zeros(
                [1], dtype='int64'), spatial_shapes.prod(1).cumsum(0)[:-1]
        ])
        # [b, num_levels, 2]
        valid_ratios = paddle.stack(valid_ratios, 1)
        return (feat_flatten, spatial_shapes, level_start_index, mask_flatten,
                lvl_pos_embed_flatten, valid_ratios)

    def forward(self, feats, pad_mask=None, gt_meta=None):
        if self.for_distill:
            aux_refpoints = gt_meta.get('aux_refpoints', None)

        # input projection and embedding
        (feat_flatten, spatial_shapes, level_start_index, mask_flatten,
         lvl_pos_embed_flatten,
         valid_ratios) = self._get_encoder_input(feats, pad_mask)

        # encoder
        memory = self.encoder(feat_flatten, spatial_shapes, level_start_index,
                              mask_flatten, lvl_pos_embed_flatten, valid_ratios)

        # prepare denoising training
        is_teacher = gt_meta.get('is_teacher', False)
        if self.training and not is_teacher:
            # prepare_for_cdn
            # input_query_label, input_query_bbox, attn_mask, dn_meta
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(gt_meta,
                                            self.num_classes,
                                            self.num_queries,
                                            self.denoising_class_embed.weight,
                                            self.num_denoising,
                                            self.label_noise_ratio,
                                            self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
        # tgt_embed is target,   refpoint_embed is init_ref_points_unact
        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits, \
        enc_outputs_class, enc_outputs_coord_unact, enc_output_memory = \
            self._get_decoder_input(
            memory, spatial_shapes, mask_flatten, denoising_class, ### denoising_class is input_query_label
            denoising_bbox_unact) ### denoising_bbox_unact is input_query_bbox
        # [2, 1100, 256] [2, 1100, 4] [2, 900, 4] [2, 900, 80] train
        # [2, 900, 256] [2, 900, 4] [2, 900, 4] [2, 900, 80] val

        # decoder
        ### hs, reference, ref_undetach
        # inter_feats, inter_ref_bboxes_unact, inter_ref_bboxes_unact_undetach = self.decoder(
        # inter_feats, inter_ref_bboxes_unact+sigmoid
        inter_feats, inter_bboxes = self.decoder(
            target, init_ref_points_unact, memory, spatial_shapes,
            level_start_index, self.dec_bbox_head, self.query_pos_head,
            valid_ratios, attn_mask, mask_flatten)

        if self.for_distill and aux_refpoints is not None:
            # tgt_embed, refpoint_embed, attn_mask = aux_refpoints from teacher
            # target, init_ref_points_unact, attn_mask = aux_refpoints from teacher
            #hs_aux, ref_aux, ref_undetach_aux = self.decoder(
            #inter_feats_aux, inter_ref_bboxes_unact_aux, inter_ref_bboxes_unact_aux_undetach = self.decoder(
            inter_feats_aux, inter_bboxes_aux = self.decoder(
                aux_refpoints[0], aux_refpoints[1], memory, spatial_shapes,
                level_start_index, self.dec_bbox_head, self.query_pos_head,
                valid_ratios, aux_refpoints[2], mask_flatten)
        else:
            inter_feats_aux = None
            inter_bboxes_aux = None

        out_bboxes = []
        out_logits = []
        for i in range(self.num_decoder_layers):
            out_logits.append(self.dec_score_head[i](inter_feats[i]))
            if i == 0:
                out_bboxes.append(
                    F.sigmoid(self.dec_bbox_head[i](inter_feats[i]) +
                              init_ref_points_unact))
            else:
                out_bboxes.append(
                    F.sigmoid(self.dec_bbox_head[i](inter_feats[i]) +
                              inverse_sigmoid(inter_bboxes[i - 1])))
        out_bboxes = paddle.stack(out_bboxes)
        out_logits = paddle.stack(out_logits)

        if self.for_distill and inter_feats_aux is not None:
            aux_kd_out_bboxes = []
            aux_kd_out_logits = []
            for i in range(self.num_decoder_layers):
                # [6, 1, 900, 256] -> [6, 1, 900, 4/80]
                aux_kd_out_logits.append(self.dec_score_head[i](inter_feats_aux[i]))
                if i == 0:
                    aux_kd_out_bboxes.append(
                        F.sigmoid(self.dec_bbox_head[i](inter_feats_aux[i]) +
                                aux_refpoints[1]))
                else:
                    aux_kd_out_bboxes.append(
                        F.sigmoid(self.dec_bbox_head[i](inter_feats_aux[i]) +
                                inverse_sigmoid(inter_bboxes_aux[i - 1])))

        if self.for_distill:
            self.distill_pairs['hs'] = inter_feats
            self.distill_pairs['reference'] = inter_bboxes
            self.distill_pairs['enc_class'] = enc_outputs_class
            self.distill_pairs['enc_coord'] = enc_outputs_coord_unact
            self.distill_pairs['enc_memory'] = enc_output_memory

            self.distill_pairs['pred_logits'] = out_logits[-1]
            self.distill_pairs['pred_boxes'] = out_bboxes[-1]

            if inter_feats_aux is not None:
                self.distill_pairs['aux_hs'] = inter_feats_aux
                self.distill_pairs['aux_reference'] = inter_bboxes_aux
                self.distill_pairs['aux_pred_logits'] = aux_kd_out_logits[-1]
                self.distill_pairs['aux_pred_boxes'] = aux_kd_out_bboxes[-1]
                self.distill_pairs['auxrf_aux_outputs'] = {}
                self.distill_pairs['auxrf_aux_outputs']['pred_logits'] = aux_kd_out_logits[:-1]
                self.distill_pairs['auxrf_aux_outputs']['pred_boxes'] = aux_kd_out_bboxes[:-1]

            self.distill_pairs['dn_meta'] = dn_meta
            self.distill_pairs['refpoints'] = (target.detach(), init_ref_points_unact.detach(), attn_mask)

        return (out_bboxes, out_logits, enc_topk_bboxes, enc_topk_logits,
                dn_meta)

    def _get_encoder_output_anchors(self,
                                    memory,
                                    spatial_shapes,
                                    memory_mask=None,
                                    grid_size=0.05):
        output_anchors = []
        idx = 0
        for lvl, (h, w) in enumerate(spatial_shapes):
            if memory_mask is not None:
                mask_ = memory_mask[:, idx:idx + h * w].reshape([-1, h, w])
                valid_H = paddle.sum(mask_[:, :, 0], 1)
                valid_W = paddle.sum(mask_[:, 0, :], 1)
            else:
                valid_H, valid_W = h, w

            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(end=h), paddle.arange(end=w))
            grid_xy = paddle.stack([grid_x, grid_y], -1).astype(memory.dtype)

            valid_WH = paddle.stack([valid_W, valid_H], -1).reshape(
                [-1, 1, 1, 2]).astype(grid_xy.dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0**lvl)
            output_anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))
            idx += h * w

        output_anchors = paddle.concat(output_anchors, 1)
        valid_mask = ((output_anchors > self.eps) *
                      (output_anchors < 1 - self.eps)).all(-1, keepdim=True)
        output_anchors = paddle.log(output_anchors / (1 - output_anchors))
        if memory_mask is not None:
            valid_mask = (valid_mask * (memory_mask.unsqueeze(-1) > 0)) > 0
        output_anchors = paddle.where(valid_mask, output_anchors,
                                      paddle.to_tensor(float("inf")))

        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        enc_output_memory = self.enc_output[0](memory) # before bn
        output_memory = self.enc_output[1](enc_output_memory)
        return enc_output_memory, output_memory, output_anchors

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           memory_mask=None,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        ### output_memory, output_proposals = gen_encoder_output_proposals(
        enc_output_memory, output_memory, output_anchors = self._get_encoder_output_anchors(
            memory, spatial_shapes, memory_mask)
        enc_outputs_class = self.enc_score_head(output_memory)
        # enc_outputs_class_unselected
        enc_outputs_coord_unact = self.enc_bbox_head(
            output_memory) + output_anchors
        # enc_outputs_coord_unselected

        _, topk_ind = paddle.topk(
            enc_outputs_class.max(-1), self.num_queries, axis=1)
        # extract region proposal boxes
        batch_ind = paddle.arange(end=bs).astype(topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)
        reference_points_unact = paddle.gather_nd(enc_outputs_coord_unact,
                                                  topk_ind)  # unsigmoided.
        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
        enc_topk_logits = paddle.gather_nd(enc_outputs_class, topk_ind)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = paddle.gather_nd(output_memory, topk_ind).detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)

        return target, reference_points_unact.detach(
        ), enc_topk_bboxes, enc_topk_logits, enc_outputs_class, enc_outputs_coord_unact, enc_output_memory
