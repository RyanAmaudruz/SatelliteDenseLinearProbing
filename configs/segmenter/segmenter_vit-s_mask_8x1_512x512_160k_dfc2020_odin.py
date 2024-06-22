_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/dfc2020.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# checkpoint = '/gpfs/work5/0/prjs0790/data/modified_checkpoints/ssl_s2c_new_transforms_0099_ckpt_MODIFIED.pth'  # noqa
# checkpoint = '/gpfs/work5/0/prjs0790/data/modified_checkpoints/ssl_leopart_mgpu_49_ckpt_MODIFIED.pth'
# checkpoint = '/gpfs/work5/0/prjs0790/data/modified_checkpoints/odin_checkpoint_1000_MODIFIED.pth'
# checkpoint = '/gpfs/home2/ramaudruz/detcon-pytorch/detcon/odin_checkpoint_1000.pth'
# checkpoint = '/gpfs/work5/0/prjs0790/data/modified_checkpoints/odin_checkpoint_e1_s7845_MODIFIED.pth'
# checkpoint = '/gpfs/work5/0/prjs0790/data/modified_checkpoints/odin_checkpoint_e4_s19614_MODIFIED.pth'
# checkpoint = '/gpfs/work5/0/prjs0790/data/modified_checkpoints/odin_checkpoint_flip_e0_s3922_MODIFIED.pth'
# checkpoint = '/gpfs/work5/0/prjs0790/data/modified_checkpoints/odin_checkpoint_cosine_e0_s3922_MODIFIED.pth'
# checkpoint = '/gpfs/work5/0/prjs0790/data/modified_checkpoints/odin_checkpoint_cosine_e1_s7845_MODIFIED.pth'
# checkpoint = '/gpfs/work5/0/prjs0790/data/modified_checkpoints/odin_checkpoint_cosine_e2_s11768_MODIFIED.pth'
# checkpoint = '/gpfs/work5/0/prjs0790/data/modified_checkpoints/odin_run_2024-03-21_18-59_ckp4_MODIFIED.pth'
# checkpoint = '/gpfs/work5/0/prjs0790/data/modified_checkpoints/odin_checkpoint_cosine_e1_s7845_MODIFIED.pth'
checkpoint = '/gpfs/work5/0/prjs0790/data/modified_checkpoints/odin_checkpoint_cosine_e0_s3922_MODIFIED.pth'




backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        in_channels=13,
        img_size=[224, 224],
        # img_size=(512, 512),
        embed_dims=384,
        num_heads=6,
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=384,
        channels=384,
        num_classes=8,
        num_layers=2,
        num_heads=6,
        embed_dims=384,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))

optimizer = dict(lr=0.001, weight_decay=0.0)

# img_norm_cfg = dict(
#     mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
# crop_size = (512, 512)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', reduce_zero_label=True),
#     dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(2048, 512),
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])
#         ])
# ]
# data = dict(
#     # num_gpus: 8 -> batch_size: 8
#     samples_per_gpu=1,
#     train=dict(pipeline=train_pipeline),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))
