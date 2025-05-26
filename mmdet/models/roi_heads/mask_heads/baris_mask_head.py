# Copyright (c) OpenMMLab. All rights reserved.
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops import SimpleRoIAlign
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from mmcv.cnn.bricks.transformer import FFN
from torch.nn.modules.utils import _pair

from mmdet.core import mask_target
from mmdet.models.builder import HEADS, build_loss
from .fcn_mask_head import BYTES_PER_FLOAT, GPU_MEM_LIMIT, _do_paste_mask


class DepthwiseSeparableUpsample(BaseModule):
    def __init__(self, in_features, scale_factor=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=(3, 3), padding=(1, 1), groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=(5, 5), padding=(2, 2), groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=(7, 7), padding=(3, 3), groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features * scale_factor * scale_factor, kernel_size=(1, 1))
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0

        x = self.projector(x)
        return self.pixel_shuffle(x)


class RefineBlock(BaseModule):
    def __init__(self, in_channels, out_channels, stages_channels, mask_out_size, out_stride,
                 act_cfg=dict(type='GELU'), upsample_flag=False, with_MSGRN=True, with_DSU=True):
        super().__init__()
        self.upsample_flag = upsample_flag
        self.with_MSGRN = with_MSGRN
        self.with_DSU = with_DSU

        if self.with_MSGRN:
            # for extracting stage 1 ~ 3 feats
            self.transform_in1 = nn.Sequential(
                nn.Conv2d(stages_channels, stages_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                          groups=stages_channels),
                nn.Conv2d(stages_channels, out_channels, kernel_size=(1, 1))
            )
            self.transform_in2 = nn.Sequential(
                nn.Conv2d(stages_channels, stages_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                          groups=stages_channels),
                nn.Conv2d(stages_channels, out_channels, kernel_size=(1, 1))
            )
            self.transform_in3 = nn.Sequential(
                nn.Conv2d(stages_channels, stages_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                          groups=stages_channels),
                nn.Conv2d(stages_channels, out_channels, kernel_size=(1, 1))
            )

            self.roi_extractor = SimpleRoIAlign(output_size=mask_out_size, spatial_scale=1.0 / out_stride)

            # for extracting stage 4 feats
            self.transform_in4 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                          groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
            )

            # for project down
            self.project_down = nn.Conv2d(out_channels * 4, out_channels, kernel_size=(1, 1))

            # for gated weight
            self.gated_weight = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1)),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1), groups=out_channels)
            )

            # for mask attn
            self.spatial_attn = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                                           padding=(1, 1), groups=out_channels)
            self.norm1 = nn.LayerNorm([out_channels, mask_out_size, mask_out_size])
            self.channel_attn = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))

            # for FFN
            self.norm2 = nn.LayerNorm(out_channels)
            self.ffn = FFN(
                embed_dims=out_channels,
                feedforward_channels=out_channels * 4,
                num_fcs=2,
                ffn_drop=0.,
                dropout_layer=dict(type='DropPath', drop_prob=0.3),
                act_cfg=act_cfg,
                add_identity=True,
                init_cfg=None)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # for upsample
        if upsample_flag:
            if with_DSU:
                self.upsample = DepthwiseSeparableUpsample(in_features=out_channels, scale_factor=2)
            else:
                upsample_cfg = dict(type='bilinear', scale_factor=2)
                self.upsample = build_upsample_layer(upsample_cfg.copy())
        else:
            self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, stages_feat, rois):
        if self.with_MSGRN:
            # for stage 1 ~ 3 feats
            ins_stage_1_feat = self.roi_extractor(self.act(self.transform_in1(stages_feat[0])), rois)
            ins_stage_2_feat = self.roi_extractor(self.act(self.transform_in2(stages_feat[1])), rois)
            ins_stage_3_feat = self.roi_extractor(self.act(self.transform_in3(stages_feat[2])), rois)

            # for stage 4 feats
            stage_4_feat = self.act(self.transform_in4(x))

            # for gated weight
            gated_weight = self.sigmoid(self.gated_weight(ins_stage_1_feat))

            # for project down
            x = torch.cat([stage_4_feat, ins_stage_3_feat, ins_stage_2_feat, ins_stage_1_feat], dim=1)
            x = self.project_down(x)

            # for mask attn
            x = self.norm1(self.spatial_attn(x)) * self.channel_attn(x)
            x = x * gated_weight

            # FFN
            b, c, h, w = x.shape
            n = h * w
            x = x.permute(0, 2, 3, 1).reshape(b, n, c)
            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

            x = x + stage_4_feat
        else:
            x = self.conv(x)

        return self.upsample(x)


@HEADS.register_module()
class BARISMaskHead(BaseModule):
    def __init__(self,
                 num_convs=2,
                 in_channels=256,
                 conv_out_channels=256,
                 out_stride=4,
                 num_classes=7,
                 conv_cfg=None,
                 norm_cfg=None,
                 stage_output_mask_size=[14, 28, 56],
                 predictor_cfg=dict(type='Conv'),
                 pre_upsample_last_stage=False,
                 loss_mask=None,
                 loss_ce=None,
                 init_cfg=None,
                 # ablation
                 with_MSGRN=True,
                 with_DSU=True):
        super(BARISMaskHead, self).__init__(init_cfg)
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.stage_output_mask_size = stage_output_mask_size
        self.pre_upsample_last_stage = pre_upsample_last_stage
        self.loss_ce = build_loss(loss_ce)

        self.stages = nn.ModuleList()
        stages_channels = in_channels
        out_channel = in_channels
        for idx, out_size in enumerate(stage_output_mask_size):
            in_channel = out_channel
            out_channel = in_channel // 2

            new_stage = RefineBlock(in_channels=in_channel,
                                    out_channels=out_channel,
                                    stages_channels=stages_channels,
                                    mask_out_size=out_size,
                                    out_stride=out_stride,
                                    upsample_flag=self.pre_upsample_last_stage or idx < len(self.stage_output_mask_size) - 1,
                                    with_MSGRN=with_MSGRN,
                                    with_DSU=with_DSU,
                                    )

            self.stages.append(new_stage)

        # for mask preds
        self.conv_logits = build_conv_layer(predictor_cfg, out_channel, num_classes, 1)

    def forward(self, x, stages_feats, rois):
        for idx, stage in enumerate(self.stages):
            x = stage(x, stages_feats, rois)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def loss(self, mask_pred, mask_targets, pos_labels):
        # cross-entropy loss
        loss_mask = self.loss_ce(mask_pred, mask_targets, pos_labels)
        return dict(loss_mask=loss_mask)

    '''
        This function is come from 
        mmdet.models.roi_heads.mask_heads.fcn_mask_head.py
        and has some change
        '''

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale):
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_pred = det_bboxes.new_tensor(mask_pred)
        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        # In most cases, scale_factor should have been
        # converted to Tensor when rescale the bbox
        if not isinstance(scale_factor, torch.Tensor):
            if isinstance(scale_factor, float):
                scale_factor = np.array([scale_factor] * 4)
                warn('Scale_factor should be a Tensor or ndarray '
                     'with shape (4,), float would be deprecated. ')
            assert isinstance(scale_factor, np.ndarray)
            scale_factor = torch.Tensor(scale_factor)

        if rescale:
            img_h, img_w = ori_shape[:2]
            bboxes = bboxes / scale_factor.to(bboxes)
        else:
            w_scale, h_scale = scale_factor[0], scale_factor[1]
            img_h = np.round(ori_shape[0] * h_scale.item()).astype(np.int32)
            img_w = np.round(ori_shape[1] * w_scale.item()).astype(np.int32)

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT)
            )
            assert (num_chunks <= N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8
        )

        if mask_pred.shape[1] > 1:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu'
            )

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds,) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())
        return cls_segms

