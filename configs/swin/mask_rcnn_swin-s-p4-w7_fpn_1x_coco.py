_base_ = './mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth'  # noqa
model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))
evaluation = dict(metric=['segm'], classwise=True, interval=3)