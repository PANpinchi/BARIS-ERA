# dataset settings
dataset_type = 'USODDataset'
data_root = 'data/USOD10k/'
img_norm_cfg = dict(
    mean=[81.236, 113.761, 117.095], std=[60.598, 58.471, 62.821], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_depth']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_prefix=data_root + 'TR/RGB',
        depth_prefix=data_root + 'TR/depth',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_prefix=data_root + 'Val/RGB',
        depth_prefix=data_root + 'Val/depth',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=data_root + 'TE/RGB',
        depth_prefix=data_root + 'TE/depth',
        pipeline=test_pipeline))
evaluation = dict(metric=['depth'])
