# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ..post_process import JDEBBoxPostProcess
import paddle
import paddle.nn.functional as F
from ..ssod_utils import QFLv2, giou_loss
from IPython import embed

__all__ = ['YOLOv3']


@register
class YOLOv3(BaseArch):
    __category__ = 'architecture'
    __shared__ = ['data_format']
    __inject__ = ['post_process']

    def __init__(self,
                 backbone='DarkNet',
                 neck='YOLOv3FPN',
                 yolo_head='YOLOv3Head',
                 post_process='BBoxPostProcess',
                 data_format='NCHW',
                 for_mot=False):
        """
        YOLOv3 network, see https://arxiv.org/abs/1804.02767

        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck instance
            yolo_head (nn.Layer): anchor_head instance
            bbox_post_process (object): `BBoxPostProcess` instance
            data_format (str): data format, NCHW or NHWC
            for_mot (bool): whether return other features for multi-object tracking
                models, default False in pure object detection models.
        """
        super(YOLOv3, self).__init__(data_format=data_format)
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head
        self.post_process = post_process
        self.for_mot = for_mot
        self.return_idx = isinstance(post_process, JDEBBoxPostProcess)

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # fpn
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        # head
        kwargs = {'input_shape': neck.out_shape}
        yolo_head = create(cfg['yolo_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "yolo_head": yolo_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        if self.for_mot:
            neck_feats = self.neck(body_feats, self.for_mot)
        else:
            neck_feats = self.neck(body_feats)

        if isinstance(neck_feats, dict):
            assert self.for_mot == True
            emb_feats = neck_feats['emb_feats']
            neck_feats = neck_feats['yolo_feats']

        is_teacher = self.inputs.get('is_teacher', False)
        if self.training or is_teacher:
            yolo_losses = self.yolo_head(neck_feats, self.inputs)

            if self.for_mot:
                return {'det_losses': yolo_losses, 'emb_feats': emb_feats}
            else:
                return yolo_losses

        else:
            yolo_head_outs = self.yolo_head(neck_feats)

            if self.for_mot:
                boxes_idx, bbox, bbox_num, nms_keep_idx = self.post_process(
                    yolo_head_outs, self.yolo_head.mask_anchors)
                output = {
                    'bbox': bbox,
                    'bbox_num': bbox_num,
                    'boxes_idx': boxes_idx,
                    'nms_keep_idx': nms_keep_idx,
                    'emb_feats': emb_feats,
                }
            else:
                if self.return_idx:
                    _, bbox, bbox_num, _ = self.post_process(
                        yolo_head_outs, self.yolo_head.mask_anchors)
                elif self.post_process is not None:
                    bbox, bbox_num = self.post_process(
                        yolo_head_outs, self.yolo_head.mask_anchors,
                        self.inputs['im_shape'], self.inputs['scale_factor'])
                else:
                    bbox, bbox_num = self.yolo_head.post_process(
                        yolo_head_outs, self.inputs['scale_factor'])
                output = {'bbox': bbox, 'bbox_num': bbox_num}

            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

    def get_distill_loss(self, head_outs, teacher_head_outs, ratio=0.1):
        # student_probs: already sigmoid
        student_probs, student_deltas ,student_dfl= head_outs
        teacher_probs, teacher_deltas , teacher_dfl= teacher_head_outs
        nc = student_probs.shape[-1]
        student_probs = student_probs.reshape([-1, nc])
        student_deltas = student_deltas.reshape([-1, 4])
        teacher_probs = teacher_probs.transpose([0, 2, 1]).reshape([-1, nc])
        teacher_deltas = teacher_deltas.reshape([-1, 4])
        student_dfl= student_dfl.reshape([-1, 4, 17])
        teacher_dfl = teacher_dfl.reshape([-1, 4])
        with paddle.no_grad():
            # Region Selection
            count_num = int(teacher_probs.shape[0] * ratio) # top 84
            # teacher_probs = F.sigmoid(teacher_logits) # [8400, 80] already sigmoid
            max_vals = paddle.max(teacher_probs, 1)
            sorted_vals, sorted_inds = paddle.topk(max_vals, teacher_probs.shape[0])
            mask = paddle.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            b_mask = mask > 0.

        loss_logits = QFLv2(
            student_probs,
            teacher_probs,
            weight=mask,
            reduction="sum") / fg_num

        inputs = paddle.concat(
            (-student_deltas[b_mask][..., :2], student_deltas[b_mask][..., 2:]),
            axis=-1)
        targets = paddle.concat(
            (-teacher_deltas[b_mask][..., :2], teacher_deltas[b_mask][..., 2:]),
            axis=-1)
        ious =iou(inputs ,targets)
        loss_deltas = giou_loss(inputs, targets).mean()
        bbox_weight = teacher_probs[b_mask] / teacher_probs[b_mask].max(axis=-1).unsqueeze(axis=-1)
        # max_metrics_per_instance =bbox_weight.max(axis=-1, keepdim=True)
        # max_ious = ious.max(axis=-1,  keepdim=True)
        # bbox_weight = bbox_weight /(max_metrics_per_instance + 1e-7) *max_ious 
        loss_dfl = _df_loss(student_dfl[b_mask],
                                     teacher_dfl[b_mask]).sum()/bbox_weight.sum()
        return {
            "distill_loss_cls": loss_logits,
            "distill_loss_box": loss_deltas,
            "distill_loss_dfl": loss_dfl,
            "fg_sum": fg_num,
        }

def iou(inputs, targets, eps=1e-7):
    inputs_area = (inputs[..., 2] - inputs[..., 0]).clip_(min=0) \
        * (inputs[..., 3] - inputs[..., 1]).clip_(min=0)
    targets_area = (targets[..., 2] - targets[..., 0]).clip_(min=0) \
        * (targets[..., 3] - targets[..., 1]).clip_(min=0)

    w_intersect = (paddle.minimum(inputs[..., 2], targets[..., 2]) -
                   paddle.maximum(inputs[..., 0], targets[..., 0])).clip_(min=0)
    h_intersect = (paddle.minimum(inputs[..., 3], targets[..., 3]) -
                   paddle.maximum(inputs[..., 1], targets[..., 1])).clip_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = targets_area + inputs_area - area_intersect

    ious = area_intersect / area_union.clip(min=eps)
    return ious


def _df_loss(pred_dist, target):
    target_left = paddle.cast(target, 'int64')
    target_right = target_left + 1
    weight_left = target_right.astype('float32') - target
    weight_right = 1 - weight_left
    loss_left = F.cross_entropy(
        pred_dist, target_left, reduction='none') * weight_left
    loss_right = F.cross_entropy(
        pred_dist, target_right, reduction='none') * weight_right
    return (loss_left + loss_right).mean(-1, keepdim=True)