import argparse
import os.path as osp
import shutil
from functools import partial

import mmcv
import numpy as np
from PIL import Image
from scipy.io import loadmat

COCO_LEN = 1024

clsID_to_trID = {
    0: 0,
    1: 1,
    2: 2
    
}


def convert_to_trainID(tuple_path, in_img_dir, in_ann_dir, out_img_dir,
                       out_mask_dir, is_train):
    imgpath, maskpath = tuple_path
    shutil.copyfile(
        osp.join(in_img_dir, imgpath),
        osp.join(out_img_dir, 'train2014', imgpath) if is_train else osp.join(
            out_img_dir, 'test2014', imgpath))
    annotate = loadmat(osp.join(in_ann_dir, maskpath))
    mask = annotate['S'].astype(np.uint8)
    mask_copy = mask.copy()
    for clsID, trID in clsID_to_trID.items():
        mask_copy[mask == clsID] = trID
    seg_filename = osp.join(out_mask_dir, 'train2014',
                            maskpath.split('.')[0] +
                            '_labelTrainIds.png') if is_train else osp.join(
                                out_mask_dir, 'test2014',
                                maskpath.split('.')[0] + '_labelTrainIds.png')
    Image.fromarray(mask_copy).save(seg_filename, 'PNG')


def generate_coco_list(folder):
    train_list = osp.join(folder, 'imageLists', 'train.txt')
    test_list = osp.join(folder, 'imageLists', 'test.txt')
    train_paths = []
    test_paths = []

    with open(train_list) as f:
        for filename in f:
            basename = filename.strip()
            imgpath = basename + '.jpg'
            maskpath = basename + '.png'
            train_paths.append((imgpath, maskpath))

    with open(test_list) as f:
        for filename in f:
            basename = filename.strip()
            imgpath = basename + '.jpg'
            maskpath = basename + '.png'
            test_paths.append((imgpath, maskpath))

    return train_paths, test_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description=\
        'Convert COCO Stuff 10k annotations to mmsegmentation format')  # noqa
    parser.add_argument('coco_path', help='coco stuff path')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--nproc', default=16, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    coco_path = args.coco_path
    nproc = args.nproc

    out_dir = args.out_dir or coco_path
    out_img_dir = osp.join(out_dir, 'images')
    out_mask_dir = osp.join(out_dir, 'annotations')

    mmcv.mkdir_or_exist(osp.join(out_img_dir, 'train2014'))
    mmcv.mkdir_or_exist(osp.join(out_img_dir, 'test2014'))
    mmcv.mkdir_or_exist(osp.join(out_mask_dir, 'train2014'))
    mmcv.mkdir_or_exist(osp.join(out_mask_dir, 'test2014'))

    train_list, test_list = generate_coco_list(coco_path)
    assert (len(train_list) +
            len(test_list)) == COCO_LEN, 'Wrong length of list {} & {}'.format(
                len(train_list), len(test_list))

    if args.nproc > 1:
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                in_img_dir=osp.join(coco_path, 'images'),
                in_ann_dir=osp.join(coco_path, 'annotations'),
                out_img_dir=out_img_dir,
                out_mask_dir=out_mask_dir,
                is_train=True),
            train_list,
            nproc=nproc)
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                in_img_dir=osp.join(coco_path, 'images'),
                in_ann_dir=osp.join(coco_path, 'annotations'),
                out_img_dir=out_img_dir,
                out_mask_dir=out_mask_dir,
                is_train=False),
            test_list,
            nproc=nproc)
    else:
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                in_img_dir=osp.join(coco_path, 'images'),
                in_ann_dir=osp.join(coco_path, 'annotations'),
                out_img_dir=out_img_dir,
                out_mask_dir=out_mask_dir,
                is_train=True), train_list)
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                in_img_dir=osp.join(coco_path, 'images'),
                in_ann_dir=osp.join(coco_path, 'annotations'),
                out_img_dir=out_img_dir,
                out_mask_dir=out_mask_dir,
                is_train=False), test_list)

    print('Done!')


if __name__ == '__main__':
    main()
