# dataset settings
dataset_type = 'CocoDataset'
#data_root = '/cpfs/user/wenzhuangwang/FICGen/datasets/voc/'
#data_root = '/cpfs/user/wenzhuangwang/FICGen/datasets/ruod/'
#data_root = '/cpfs/user/wenzhuangwang/FICGen/datasets/exdark/'
#data_root = '/cpfs/user/wenzhuangwang/FICGen/datasets/hrsc2016/'
#data_root = '/cpfs/user/wenzhuangwang/Ctrl-RS/datasets/hrrsd/'
#data_root = '/cpfs/user/wenzhuangwang/FICGen/datasets/lis/'
#data_root = '/cpfs/user/wenzhuangwang/FICGen/datasets/dota/'
data_root = '/cpfs/user/wenzhuangwang/FICGen/datasets/dior/'
#data_root = '/cpfs/user/wenzhuangwang/FICGen/datasets/acdc/'
#data_root = '/cpfs/user/wenzhuangwang/FICGen/datasets/dawn/'
#data_root = '/cpfs/user/wenzhuangwang/FICGen/datasets/camo/'
#data_root = "/cpfs/user/wenzhuangwang/geodiffusion/data/nuimages/"
#data_root = '/cpfs/user/wenzhuangwang/FICGen/datasets/coco2017/'
#data_root = '/cpfs/user/wenzhuangwang/FICGen/datasets/voc/'
#data_root = "/cpfs/user/wenzhuangwang/FICGen/datasets/deepspace/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
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
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + "train/",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test.json',
        img_prefix='/cpfs/user/wenzhuangwang/FICGen/generated_images/dior/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test.json',
        img_prefix='/cpfs/user/wenzhuangwang/FICGen/generated_images/dior/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox', classwise=True)
