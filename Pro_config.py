_base_ = "F:\\桌面\\大三下\\神经网络\\mmclassification\\configs\\resnet\\resnet50_8xb32_in1k.py"
#python train.py pro_config.py (tools)--work-dir work_dirs/flower_classifier --gpus 1
model = dict(
    head=dict(num_classes=5)
)

data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
    num_classes=5
)

# 类别名称请替换为你实际数据集子目录名
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type='ImageNet',
        data_root="F:\\桌面\\大三下\\神经网络\\EX1\\flower_dataset",
        split="train",  # 指定子目录
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', scale=224),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs'),
        ]
    ),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type='ImageNet',
        data_root="F:\\桌面\\大三下\\神经网络\\EX1\\flower_dataset",
        split="val",  # 指定子目录
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs'),
        ]
    ),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
)

evaluation = dict(metric='accuracy')

lr_config = dict(
    policy='step',
    step=[10, 20],
)

load_from = "F:\\桌面\\大三下\\神经网络\\EX1\\resnet50_8xb32_in1k_20210831-ea4938fc.pth"

work_dir = "F:\\桌面\\大三下\\神经网络\\EX1\\work_dirs\\flower_classifier"
