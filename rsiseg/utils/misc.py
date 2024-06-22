# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
import warnings
import os


def prepare_dir(file_path):
    """
    This function is used to create the directories needed to output a path. If the directories already exist, the
    function continues.
    """
    # Remove the file name to only keep the directory path.
    dir_path = '/'.join(file_path.split('/')[:-1])
    # Try to create the directory. Will have no effect if the directory already exists.
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass

def find_latest_checkpoint(path, suffix='pth'):
    """This function is for finding the latest checkpoint.

    It will be used when automatically resume, modified from
    https://github.com/open-mmlab/mmdetection/blob/dev-v2.20.0/mmdet/utils/misc.py

    Args:
        path (str): The path to find checkpoints.
        suffix (str): File extension for the checkpoint. Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.
    """
    if not osp.exists(path):
        warnings.warn("The path of the checkpoints doesn't exist.")
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    checkpoints = glob.glob(osp.join(path, f'*.{suffix}'))
    if len(checkpoints) == 0:
        warnings.warn('The are no checkpoints in the path')
        return None
    latest = -1
    latest_path = ''
    for checkpoint in checkpoints:
        if len(checkpoint) < len(latest_path):
            continue
        # `count` is iteration number, as checkpoints are saved as
        # 'iter_xx.pth' or 'epoch_xx.pth' and xx is iteration number.
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path
