import numpy as np

_base_ = './segmenter_vit-s_mask_8x1_512x512_160k_dfc2020.py'

# MADOS class distribution
# class_dist = [
#     0.00336, 0.00241, 0.00336, 0.00142, 0.00775, 0.18452, 0.34775, 0.20638, 0.00062, 0.1169, 0.09188, 0.01309, 0.00917,
#     0.00176, 0.00963
# ]
# Segmunich class distribution
# class_dist = [
#     0.1819087, 0.244692, 0.0154577, 0.1137857, 0.28226, 0.017560357, 0.02637844, 0.008453476, 0.009553523, 0.0053915823, 0.01950901, 0.062555, 0.0124880
# ]
# DFC2020 class distribution
class_dist = [
    0.209744, 0.0067099, 0.1465833, 0.016390, 0.2295087, 0.1316990, 0.00074441, 0.2586201
]

class_weight = [
    1/np.log(1.02 + d) for d in class_dist
]

model = dict(
    decode_head=dict(
        _delete_=True,
        type='FCNHead',
        in_channels=384,
        channels=384,
        num_convs=0,
        dropout_ratio=0.0,
        concat_input=False,
        num_classes=8,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            class_weight=class_weight
        )
    )
)
