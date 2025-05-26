_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k_20230104-c48d16a5.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.1,
        layer_scale_init_value=0.,
        use_grn=True,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    roi_head=dict(
        type='WaterRoIHead',
        mask_head=dict(
            _delete_=True,
            type='WaterMaskHead',
            num_convs_gff=2,
            num_convs_lcf=2,
            image_patch_token=3,
            graph_top_k=11,
            num_heads_in_gat=1,
            classes_num_in_stages=[7, 7, 1],
            stage_output_mask_size=[14, 28, 56],
            loss_cfg=dict(
                type='LaplacianCrossEntropyLoss',
                stage_lcf_loss_weight=[0.25, 0.65, 1],
                boundary_width=3,
                start_stage=2)
        )
    )
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
