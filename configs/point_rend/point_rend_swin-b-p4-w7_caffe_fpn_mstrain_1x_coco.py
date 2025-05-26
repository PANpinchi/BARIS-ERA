_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
# model settings
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth'  # noqa
model = dict(
    type='PointRend',
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
        type='PointRendRoIHead',
        mask_roi_extractor=dict(
            type='GenericRoIExtractor',
            aggregation='concat',
            roi_layer=dict(
                _delete_=True, type='SimpleRoIAlign', output_size=14),
            out_channels=256,
            featmap_strides=[4]),
        mask_head=dict(
            _delete_=True,
            type='CoarseMaskHead',
            num_fcs=2,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=7,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        point_head=dict(
            type='MaskPointHead',
            num_fcs=3,
            in_channels=256,
            fc_channels=256,
            num_classes=7,
            coarse_pred_each_layer=True,
            loss_point=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(
            mask_size=7,
            num_points=14 * 14,
            oversample_ratio=3,
            importance_sample_ratio=0.75)),
    test_cfg=dict(
        rcnn=dict(
            subdivision_steps=5,
            subdivision_num_points=28 * 28,
            scale_factor=2)))

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
