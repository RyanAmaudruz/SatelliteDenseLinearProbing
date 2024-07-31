_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/mados.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# checkpoint = '/gpfs/work5/0/prjs0790/data/modified_checkpoints/ssl_s2c_new_transforms_0099_ckpt_MODIFIED.pth'  # noqa
# checkpoint = '/gpfs/work5/0/prjs0790/data/modified_checkpoints/ssl_leopart_mgpu_49_ckpt_MODIFIED.pth'
# checkpoint = '/var/node433/local/ryan_a/data/old_checkpoints/B13_vits16_dino_0099_ckpt.pth'
# checkpoint = '/var/node433/local/ryan_a/data/ssl4eo_ssl/ssl4eo_ssl/distillation_l2_normalised/checkpoint.pth'
checkpoint = '/var/node433/local/ryan_a/data/ssl4eo_ssl/ssl4eo_ssl/ssl_s2c_new_transforms/checkpoint.pth'
# checkpoint = '/var/node433/local/ryan_a/data/odin_missing_runs/trans_mixed_aug_wo_loc_neg-k=20_4/2024-04-11_16-55_ckp-epoch=04.ckpt'
# checkpoint =  '/var/node433/local/ryan_a/data/odin_missing_runs/transform_fixed-mixed_aug-w_local_negs/2024-04-06_12-31_ckp-epoch=04.ckpt'
# checkpoint = '/var/node433/local/ryan_a/data/odin_missing_runs/transform_fixed-mixed_aug/2024-04-06_08-14_ckp-epoch=04.ckpt'
# checkpoint = '/var/node433/local/ryan_a/data/leo_missing/single_queue-with_dino/20240421-164828_ckp-epoch=24_mod.ckpt'
# checkpoint = '/var/node433/local/ryan_a/data/leo_missing/leopart_new_transform_leopart-20240221-081849/ckp-epoch=14_mod.ckpt'
# checkpoint = '/var/node433/local/ryan_a/data/leo_missing/single_queue-with_dino/20240421-164828_ckp-epoch=19_mod.ckpt'
# checkpoint = '/var/node433/local/ryan_a/data/leo_missing/new_queue-with_dino_loss/20240419-002419_ckp-epoch=24_mod.ckpt'
# checkpoint = '/var/node433/local/ryan_a/data/leo_missing/leo_new_queue/ckp-epoch=19_mod.ckpt'
# checkpoint = '/var/node433/local/ryan_a/data/leo_missing/simple_leo/ckp-epoch=24_mod1.ckpt'

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
