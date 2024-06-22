_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/dfc2020.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(
        type='ResNet',
        in_channels=13,
        ),
    decode_head=dict(num_classes=8),
    auxiliary_head=dict(num_classes=8),
    pretrained='/gpfs/work5/0/prjs0790/data/old_checkpoints/B13_rn50_moco_0099_ckpt_MODIFIED.pth',)
    # pretrained='/gpfs/work5/0/prjs0790/data/run_outputs/checkpoints/leopart_ssl/leopart_run3/leopart-20240217-100845/ckp-epoch=39.ckpt',)
    # pretrained='/gpfs/home2/ramaudruz/SSL4EO-S12/old_checkpoints/B13_rn50_moco_0099_ckpt.pth',)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)



#evaluation = dict(interval=400, metric='mIoU', pre_eval=True)