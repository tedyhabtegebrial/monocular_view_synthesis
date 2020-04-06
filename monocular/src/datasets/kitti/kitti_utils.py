import torch
import torch.nn.functional as F

def resize_segmentation_maps(seg_map, target_size):
    h, w = seg_map.shape
    seg_map = seg_map.view(1, 1, h, w)
    seg_map = F.interpolate(input=seg_map, size=(target_size), mode='nearest')
    return seg_map.view(target_size)

def label_to_one_hot(input_seg, num_classes):
    # num_classes = CARLA_NUM_CLASSES
    # assert input_seg.max() < num_classes, f'Num classes == {input_seg.max()} exceeds {num_classes}'
    b, _, h, w = input_seg.shape
    lables = torch.zeros(b, num_classes, h, w).float()
    labels[input_seg.lt(num_classes)] = lables.lt(num_classes).scatter_(dim=1, index=input_seg.long(), value=1.0)
    labels = labels.to(input_seg.device)
    return labels

# dirve, ['start_frame', 'end_frame']
sequence_to_raw = {
    '00': ['2011_10_03_drive_0027' ,'000000' ,'004540'],
    '01': ['2011_10_03_drive_0042' ,'000000' ,'001100'],
    '02': ['2011_10_03_drive_0034' ,'000000' ,'004660'],
    '03': ['2011_09_26_drive_0067' ,'000000' ,'000800'],
    '04': ['2011_09_30_drive_0016' ,'000000' ,'000270'],
    '05': ['2011_09_30_drive_0018' ,'000000' ,'002760'],
    '06': ['2011_09_30_drive_0020' ,'000000' ,'001100'],
    '07': ['2011_09_30_drive_0027' ,'000000' ,'001100'],
    '08': ['2011_09_30_drive_0028' ,'001100' ,'005170'],
    '09': ['2011_09_30_drive_0033' ,'000000' ,'001590'],
    '10': ['2011_09_30_drive_0034' ,'000000' ,'001200']}
# https://github.com/tedyhabtegebrial/inverse_warping/blob/clean/nvdataloader/nvdataloader/kitti/kitti_warping_utils.py
