_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth'  # noqa
model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
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
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[8, 11])
evaluation = dict(metric=['segm'], classwise=True, interval=3)
runner = dict(max_epochs=12)

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))

