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
from ..bbox_utils import batch_distance2bbox
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

    def decode_head_outs(self, head_outs):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs

        if pred_dist.shape[-1] > 4: # self.proj_conv 68->4
            anchor_points_s = anchor_points / stride_tensor
            pred_bboxes = self.yolo_head._bbox_decode(anchor_points_s, pred_dist)
        else:
            pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
            pred_bboxes *= stride_tensor

        # scale bbox to origin
        scale_y, scale_x = paddle.split(self.inputs['scale_factor'], 2, axis=-1)
        scale_factor = paddle.concat(
            [scale_x, scale_y, scale_x, scale_y],
            axis=-1).reshape([-1, 1, 4])
        pred_bboxes /= scale_factor

        return pred_scores, pred_bboxes

    def get_distill_loss(self, head_outs, teacher_head_outs, ratio=0.1):
        # student_probs: already sigmoid
        student_probs, student_deltas = self.decode_head_outs(head_outs)
        teacher_probs, teacher_deltas = self.decode_head_outs(teacher_head_outs)

        nc = student_probs.shape[-1]
        student_probs = student_probs.reshape([-1, nc])
        student_deltas = student_deltas.reshape([-1, 4])
        teacher_probs = teacher_probs.transpose([0, 2, 1]).reshape([-1, nc])
        teacher_deltas = teacher_deltas.reshape([-1, 4])

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
        loss_deltas = giou_loss(inputs, targets).mean()

        return {
            "distill_loss_cls": loss_logits,
            "distill_loss_box": loss_deltas,
            "fg_sum": fg_num,
        }
