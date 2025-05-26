import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

import numpy as np
from mmcv.cnn import ConvModule, build_upsample_layer
from mmcv.ops.roi_align import roi_align

from mmdet.core.mask.structures import polygon_to_bitmap, BitmapMasks
from mmdet.models.builder import HEADS, build_loss, build_roi_extractor
from .fcn_mask_head import _do_paste_mask
from .fcn_mask_head import BYTES_PER_FLOAT, GPU_MEM_LIMIT
from mmcv.ops import SimpleRoIAlign

from mmdet.core import mask_target


class MultiBranchFusion(nn.Module):

    def __init__(self, feat_dim, dilations=[1, 3, 5]):
        super(MultiBranchFusion, self).__init__()

        for idx, dilation in enumerate(dilations):
            self.add_module(f'dilation_conv_{idx + 1}', ConvModule(
                feat_dim, feat_dim, kernel_size=3, padding=dilation, dilation=dilation))

        self.merge_conv = ConvModule(feat_dim, feat_dim, kernel_size=1, act_cfg=None)

    def forward(self, x):
        feat_1 = self.dilation_conv_1(x)
        feat_2 = self.dilation_conv_2(x)
        feat_3 = self.dilation_conv_3(x)
        out_feat = self.merge_conv(feat_1 + feat_2 + feat_3)
        return out_feat


class MultiBranchFusionAvg(MultiBranchFusion):

    def forward(self, x):
        feat_1 = self.dilation_conv_1(x)
        feat_2 = self.dilation_conv_2(x)
        feat_3 = self.dilation_conv_3(x)
        feat_4 = F.avg_pool2d(x, x.shape[-1])
        out_feat = self.merge_conv(feat_1 + feat_2 + feat_3 + feat_4)
        return out_feat


class SFMStage(nn.Module):

    def __init__(self,
                 semantic_in_channel=256,
                 semantic_out_channel=256,
                 instance_in_channel=256,
                 instance_out_channel=256,
                 fusion_type='MultiBranchFusion',
                 dilations=[1, 3, 5],
                 out_size=14,
                 num_classes=80,
                 semantic_out_stride=4,
                 mask_use_sigmoid=False,
                 upsample_cfg=dict(type='bilinear', scale_factor=2)):
        super(SFMStage, self).__init__()

        self.semantic_out_stride = semantic_out_stride
        self.mask_use_sigmoid = mask_use_sigmoid
        self.num_classes = num_classes

        # for extracting instance-wise semantic feats
        self.semantic_transform_in = nn.Conv2d(semantic_in_channel, semantic_out_channel, 1)
        self.semantic_roi_extractor = build_roi_extractor(dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=out_size, sampling_ratio=0),
                out_channels=semantic_out_channel,
                featmap_strides=[semantic_out_stride, ]))
        self.semantic_transform_out = nn.Conv2d(semantic_out_channel, semantic_out_channel, 1)

        self.instance_logits = nn.Conv2d(instance_in_channel, num_classes, 1)

        fuse_in_channel = instance_in_channel + semantic_out_channel + 2
        self.fuse_conv = nn.ModuleList([
            nn.Conv2d(fuse_in_channel, instance_in_channel, 1),
            globals()[fusion_type](instance_in_channel, dilations=dilations)])

        self.fuse_transform_out = nn.Conv2d(instance_in_channel, instance_out_channel - 2, 1)
        self.upsample = build_upsample_layer(upsample_cfg.copy())
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in [self.semantic_transform_in, self.semantic_transform_out, self.instance_logits, self.fuse_transform_out]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in self.fuse_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, semantic_feat, semantic_pred, rois, roi_labels):
        concat_tensors = [instance_feats]

        # instance-wise semantic feats
        semantic_feat = self.relu(self.semantic_transform_in(semantic_feat))
        ins_semantic_feats = self.semantic_roi_extractor([semantic_feat,], rois)
        ins_semantic_feats = self.relu(self.semantic_transform_out(ins_semantic_feats))
        concat_tensors.append(ins_semantic_feats)

        # instance masks
        instance_preds = self.instance_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        _instance_preds = instance_preds.sigmoid() if self.mask_use_sigmoid else instance_preds
        instance_masks = F.interpolate(_instance_preds, instance_feats.shape[-2], mode='bilinear', align_corners=True)
        concat_tensors.append(instance_masks)

        # instance-wise semantic masks
        _semantic_pred = semantic_pred.sigmoid() if self.mask_use_sigmoid else semantic_pred
        ins_semantic_masks = roi_align(
            _semantic_pred.float(), rois, instance_feats.shape[-2:], 1.0 / self.semantic_out_stride, 0, 'avg', True)
        ins_semantic_masks = F.interpolate(
            ins_semantic_masks, instance_feats.shape[-2:], mode='bilinear', align_corners=True)
        concat_tensors.append(ins_semantic_masks)

        # fuse instance feats & instance masks & semantic feats & semantic masks
        fused_feats = torch.cat(concat_tensors, dim=1)
        for conv in self.fuse_conv:
            fused_feats = self.relu(conv(fused_feats))

        fused_feats = self.relu(self.fuse_transform_out(fused_feats))
        fused_feats = self.relu(self.upsample(fused_feats))

        # concat instance and semantic masks with fused feats again
        instance_masks = F.interpolate(_instance_preds, fused_feats.shape[-2], mode='bilinear', align_corners=True)
        ins_semantic_masks = F.interpolate(ins_semantic_masks, fused_feats.shape[-2], mode='bilinear', align_corners=True)
        fused_feats = torch.cat([fused_feats, instance_masks, ins_semantic_masks], dim=1)

        return instance_preds, fused_feats


@HEADS.register_module()
class RefineMaskHead(nn.Module):

    def __init__(self,
                 num_convs=2,
                 in_channels=256,
                 conv_out_channels=256,
                 out_stride=4,
                 num_classes=7,
                 stage_output_mask_size=[14, 28, 56],
                 predictor_cfg=dict(type='Conv'),
                 pre_upsample_last_stage=False,
                 loss_mask=None,
                 num_convs_instance=2,
                 num_convs_semantic=4,
                 conv_in_channels_instance=256,
                 conv_in_channels_semantic=256,
                 conv_kernel_size_instance=3,
                 conv_kernel_size_semantic=3,
                 conv_out_channels_instance=256,
                 conv_out_channels_semantic=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 fusion_type='MultiBranchFusion',
                 dilations=[1, 3, 5],
                 semantic_out_stride=4,
                 mask_use_sigmoid=False,
                 stage_num_classes=[80, 80, 80, 80],
                 stage_sup_size=[14, 28, 56, 112],
                 upsample_cfg=dict(type='bilinear', scale_factor=2),
                 loss_cfg=None,
                 loss_ce=None):
        super(RefineMaskHead, self).__init__()

        self.num_convs_instance = num_convs_instance
        self.conv_kernel_size_instance = conv_kernel_size_instance
        self.conv_in_channels_instance = conv_in_channels_instance
        self.conv_out_channels_instance = conv_out_channels_instance

        self.num_convs_semantic = num_convs_semantic
        self.conv_kernel_size_semantic = conv_kernel_size_semantic
        self.conv_in_channels_semantic = conv_in_channels_semantic
        self.conv_out_channels_semantic = conv_out_channels_semantic

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.semantic_out_stride = semantic_out_stride
        self.stage_sup_size = stage_sup_size
        self.stage_num_classes = stage_num_classes

        self._build_conv_layer('instance')
        self._build_conv_layer('semantic')
        self.loss_ce = build_loss(loss_ce) if loss_ce is not None else None
        self.loss_func = build_loss(loss_cfg) if loss_cfg is not None else None

        assert len(self.stage_sup_size) > 1
        self.stages = nn.ModuleList()
        out_channel = conv_out_channels_instance
        for idx, out_size in enumerate(self.stage_sup_size[:-1]):
            in_channel = out_channel
            out_channel = in_channel // 2

            new_stage = SFMStage(
                semantic_in_channel=conv_out_channels_semantic,
                semantic_out_channel=in_channel,
                instance_in_channel=in_channel,
                instance_out_channel=out_channel,
                fusion_type=fusion_type,
                dilations=dilations,
                out_size=out_size,
                num_classes=self.stage_num_classes[idx],
                semantic_out_stride=semantic_out_stride,
                mask_use_sigmoid=mask_use_sigmoid,
                upsample_cfg=upsample_cfg)

            self.stages.append(new_stage)

        self.final_instance_logits = nn.Conv2d(out_channel, self.stage_num_classes[-1], 1)
        self.semantic_logits = nn.Conv2d(conv_out_channels_semantic, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def _build_conv_layer(self, name):
        out_channels = getattr(self, f'conv_out_channels_{name}')
        conv_kernel_size = getattr(self, f'conv_kernel_size_{name}')

        convs = []
        for i in range(getattr(self, f'num_convs_{name}')):
            in_channels = getattr(self, f'conv_in_channels_{name}') if i == 0 else out_channels
            conv = ConvModule(in_channels, out_channels, conv_kernel_size, dilation=1, padding=1)
            convs.append(conv)

        self.add_module(f'{name}_convs', nn.ModuleList(convs))

    def init_weights(self):
        for m in [self.final_instance_logits, self.semantic_logits]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, semantic_feat, rois, roi_labels):
        for conv in self.instance_convs:
            instance_feats = conv(instance_feats)

        for conv in self.semantic_convs:
            semantic_feat = conv(semantic_feat)

        semantic_pred = self.semantic_logits(semantic_feat)

        stage_instance_preds = []
        for stage in self.stages:
            instance_preds, instance_feats = stage(instance_feats, semantic_feat, semantic_pred, rois, roi_labels)
            stage_instance_preds.append(instance_preds)

        # for LVIS, use class-agnostic classifier for the last stage
        if self.stage_num_classes[-1] == 1:
            roi_labels = roi_labels.clamp(max=0)

        # instance_preds = self.final_instance_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        instance_preds = self.final_instance_logits(instance_feats)
        stage_instance_preds.append(instance_preds)

        return stage_instance_preds, semantic_pred

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    # def loss(self, stage_instance_preds, semantic_pred, stage_instance_targets, semantic_target):
    #
    #     loss_instance, loss_semantic = self.loss_func(
    #         stage_instance_preds, semantic_pred, stage_instance_targets, semantic_target)
    #
    #     return dict(loss_instance=loss_instance), dict(loss_semantic=loss_semantic)
    def loss(self, stage_instance_preds, mask_targets, pos_labels):
        if self.loss_ce is not None:
            loss_mask = self.loss_ce(stage_instance_preds[-1], mask_targets, pos_labels)
        else:
            loss_mask = self.loss_func(stage_instance_preds, mask_targets)
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
        cls_segms = [[] for _ in range(self.stage_num_classes[-1])]  # BG is not included in num_classes
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


class SimpleSFMStage(nn.Module):

    def __init__(self,
                 semantic_in_channel=256,
                 semantic_out_channel=256,
                 instance_in_channel=256,
                 instance_out_channel=256,
                 fusion_type='MultiBranchFusion',
                 dilations=[1, 3, 5],
                 out_size=14,
                 num_classes=80,
                 semantic_out_stride=4,
                 upsample_cfg=dict(type='bilinear', scale_factor=2)):
        super(SimpleSFMStage, self).__init__()

        self.semantic_out_stride = semantic_out_stride
        self.num_classes = num_classes

        # for extracting instance-wise semantic feats
        self.semantic_transform_in = nn.Conv2d(semantic_in_channel, semantic_out_channel, 1)
        self.semantic_roi_extractor = SimpleRoIAlign(output_size=out_size, spatial_scale=1.0/semantic_out_stride)

        fuse_in_channel = instance_in_channel + semantic_out_channel + 1
        self.fuse_conv = nn.ModuleList([
            nn.Conv2d(fuse_in_channel, instance_in_channel, 1),
            globals()[fusion_type](instance_in_channel, dilations=dilations)])

        self.fuse_transform_out = nn.Conv2d(instance_in_channel, instance_out_channel - 1, 1)
        self.upsample = build_upsample_layer(upsample_cfg.copy())
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in [self.semantic_transform_in, self.fuse_transform_out]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in self.fuse_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, instance_logits, semantic_feat, rois, upsample=True):
        # instance-wise semantic feats
        semantic_feat = self.relu(self.semantic_transform_in(semantic_feat))
        ins_semantic_feats = self.semantic_roi_extractor(semantic_feat, rois)

        # fuse instance feats & instance masks & semantic feats
        concat_tensors = [instance_feats, ins_semantic_feats, instance_logits.sigmoid()]
        fused_feats = torch.cat(concat_tensors, dim=1)
        for conv in self.fuse_conv:
            fused_feats = self.relu(conv(fused_feats))

        # concat instance masks with fused feats again
        fused_feats = self.relu(self.fuse_transform_out(fused_feats))
        fused_feats = torch.cat([fused_feats, instance_logits.sigmoid()], dim=1)   # concat before upsampling, same as paper
        fused_feats = self.upsample(fused_feats) if upsample else fused_feats
        return fused_feats


@HEADS.register_module()
class SimpleRefineMaskHead(nn.Module):

    def __init__(self,
                 num_convs_instance=2,
                 num_convs_semantic=4,
                 conv_in_channels_instance=256,
                 conv_in_channels_semantic=256,
                 conv_kernel_size_instance=3,
                 conv_kernel_size_semantic=3,
                 conv_out_channels_instance=256,
                 conv_out_channels_semantic=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 fusion_type='MultiBranchFusionAvg',
                 dilations=[1, 3, 5],
                 semantic_out_stride=4,
                 stage_num_classes=[80, 80, 80, 80],
                 stage_sup_size=[14, 28, 56, 112],
                 pre_upsample_last_stage=False,  # if True, upsample features and then compute logits
                 upsample_cfg=dict(type='bilinear', scale_factor=2),
                 loss_cfg=dict(
                    type='BARCrossEntropyLoss',
                    stage_instance_loss_weight=[0.25, 0.5, 0.75, 1.0],
                    boundary_width=2,
                    start_stage=1)):
        super(SimpleRefineMaskHead, self).__init__()

        self.num_convs_instance = num_convs_instance
        self.conv_kernel_size_instance = conv_kernel_size_instance
        self.conv_in_channels_instance = conv_in_channels_instance
        self.conv_out_channels_instance = conv_out_channels_instance

        self.num_convs_semantic = num_convs_semantic
        self.conv_kernel_size_semantic = conv_kernel_size_semantic
        self.conv_in_channels_semantic = conv_in_channels_semantic
        self.conv_out_channels_semantic = conv_out_channels_semantic

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.semantic_out_stride = semantic_out_stride
        self.stage_sup_size = stage_sup_size
        self.stage_num_classes = stage_num_classes
        self.pre_upsample_last_stage = pre_upsample_last_stage

        self._build_conv_layer('instance')
        self._build_conv_layer('semantic')
        self.loss_func = build_loss(loss_cfg)

        assert len(self.stage_sup_size) > 1
        self.stages = nn.ModuleList()
        out_channel = conv_out_channels_instance
        stage_out_channels = [conv_out_channels_instance]
        for idx, out_size in enumerate(self.stage_sup_size[:-1]):
            in_channel = out_channel
            out_channel = in_channel // 2

            new_stage = SimpleSFMStage(
                semantic_in_channel=conv_out_channels_semantic,
                semantic_out_channel=in_channel,
                instance_in_channel=in_channel,
                instance_out_channel=out_channel,
                fusion_type=fusion_type,
                dilations=dilations,
                out_size=out_size,
                num_classes=self.stage_num_classes[idx],
                semantic_out_stride=semantic_out_stride,
                upsample_cfg=upsample_cfg)

            self.stages.append(new_stage)
            stage_out_channels.append(out_channel)

        self.stage_instance_logits = nn.ModuleList([
            nn.Conv2d(stage_out_channels[idx], num_classes, 1) for idx, num_classes in enumerate(self.stage_num_classes)])
        self.relu = nn.ReLU(inplace=True)

    def _build_conv_layer(self, name):
        out_channels = getattr(self, f'conv_out_channels_{name}')
        conv_kernel_size = getattr(self, f'conv_kernel_size_{name}')

        convs = []
        for i in range(getattr(self, f'num_convs_{name}')):
            in_channels = getattr(self, f'conv_in_channels_{name}') if i == 0 else out_channels
            conv = ConvModule(in_channels, out_channels, conv_kernel_size, dilation=1, padding=1)
            convs.append(conv)

        self.add_module(f'{name}_convs', nn.ModuleList(convs))

    def init_weights(self):
        for m in self.stage_instance_logits:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, semantic_feat, rois, roi_labels):
        for conv in self.instance_convs:
            instance_feats = conv(instance_feats)

        for conv in self.semantic_convs:
            semantic_feat = conv(semantic_feat)

        stage_instance_preds = []
        for idx, stage in enumerate(self.stages):
            instance_logits = self.stage_instance_logits[idx](instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
            upsample_flag = self.pre_upsample_last_stage or idx < len(self.stages) - 1
            instance_feats = stage(instance_feats, instance_logits, semantic_feat, rois, upsample_flag)
            stage_instance_preds.append(instance_logits)

        # if use class-agnostic classifier for the last stage
        if self.stage_num_classes[-1] == 1:
            roi_labels = roi_labels.clamp(max=0)

        instance_preds = self.stage_instance_logits[-1](instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        if not self.pre_upsample_last_stage:
            instance_preds = F.interpolate(instance_preds, scale_factor=2, mode='bilinear', align_corners=True)
        stage_instance_preds.append(instance_preds)

        return stage_instance_preds

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def loss(self, stage_instance_preds, stage_instance_targets):
        loss_instance = self.loss_func(stage_instance_preds, stage_instance_targets)
        return dict(loss_instance=loss_instance)

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale):
        mask_pred = mask_pred.sigmoid()
        device = mask_pred[0].device
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor

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
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <= N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        if mask_pred.shape[1] > 1:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        im_segms = [im_mask[i].cpu().numpy() for i in range(N)]
        return im_segms