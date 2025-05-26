_base_ = './point_rend_swin-b-p4-w7_caffe_fpn_mstrain_1x_coco.py'
# learning policy
lr_config = dict(step=[28, 34])
runner = dict(type='EpochBasedRunner', max_epochs=36)