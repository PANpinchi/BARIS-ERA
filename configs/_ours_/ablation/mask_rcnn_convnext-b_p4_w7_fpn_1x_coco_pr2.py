_base_ = [
    '../../_base_/models/mask_rcnn_r50_fpn.py',
    '../../_base_/datasets/coco_instance.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k_20230104-c48d16a5.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='ConvNeXtWithEMA',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.1,
        layer_scale_init_value=0.,
        use_grn=True,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.'),
        with_ema=True,
        projection_ratio=2,
        num_learnable_embedding=16),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    roi_head=dict(
        type='BARISRoIHead',
        mask_head=dict(
            type='BARISMaskHead',
            in_channels=256,
            num_classes=7,
            stage_output_mask_size=[14, 28, 56],
            loss_ce=dict(
                type='BoundaryAwareCrossEntropyLoss',
                scale=4,
                use_mask=True,
                loss_weight=1.0),
        )
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=56,
            pos_weight=-1,
            debug=False)),
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'layer_wise',
        'num_layers': 6
    })

lr_config = dict(warmup_iters=1000, step=[8, 11])
evaluation = dict(metric=['segm'], classwise=True, interval=3)
runner = dict(max_epochs=12)

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))
