import argparse
from os import path as osp

from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs


def create_lmdb_for_div2k():
    """Create lmdb files for DIV2K dataset.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LR_bicubic/X2_sub
            DIV2K_train_LR_bicubic/X3_sub
            DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """
    # HR images
    folder_path = 'datasets/DIV2K/DIV2K_train_HR_sub'
    lmdb_path = 'datasets/DIV2K/DIV2K_train_HR_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True,n_thread=16)

    # LRx2 images
    # folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2_sub'
    # lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2_sub.lmdb'
    # img_path_list, keys = prepare_keys_div2k(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # LRx3 images
    # folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X3_sub'
    # lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X3_sub.lmdb'
    # img_path_list, keys = prepare_keys_div2k(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # LRx4 images
    # folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
    # lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub.lmdb'
    # img_path_list, keys = prepare_keys_div2k(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # LRx8 images
    # folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X8_sub'
    # lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X8_sub.lmdb'
    # img_path_list, keys = prepare_keys_div2k(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_div2k_flickr2k():
    """Create lmdb files for DIV2K dataset.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LR_bicubic/X2_sub
            DIV2K_train_LR_bicubic/X3_sub
            DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """
    # HR images
    folder_path = 'datasets/DIV2K_Flickr2K/HR_sub'
    lmdb_path = 'datasets/DIV2K_Flickr2K/HR_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # LRx2 images
    folder_path = 'datasets/DIV2K_Flickr2K/LR_bicubic/X2_sub'
    lmdb_path = 'datasets/DIV2K_Flickr2K/LR_bicubic/X2_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # LRx3 images
    folder_path = 'datasets/DIV2K_Flickr2K/LR_bicubic/X3_sub'
    lmdb_path = 'datasets/DIV2K_Flickr2K/LR_bicubic/X3_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # LRx4 images
    folder_path = 'datasets/DIV2K_Flickr2K/LR_bicubic/X4_sub'
    lmdb_path = 'datasets/DIV2K_Flickr2K/LR_bicubic/X4_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # LRx8 images
    # folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X8_sub'
    # lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X8_sub.lmdb'
    # img_path_list, keys = prepare_keys_div2k(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def prepare_keys_div2k(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys



def create_lmdb_for_realsr():
    # # HR images  change dataset path
    folder_path = 'datasets/RealSR(V3)/All/Train/3_HR_sub'
    lmdb_path = 'datasets/RealSR(V3)/All/Train/3_HR_sub.lmdb'
    img_path_list, keys = prepare_keys_realsr(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)
    #
    # # LRx2 images
    # folder_path = 'datasets/RealSR(V3)/All/Train/2_LR_sub'
    # lmdb_path = 'datasets/RealSR(V3)/All/Train/2_LR_sub.lmdb'
    # img_path_list, keys = prepare_keys_div2k(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)
    #
    # # LRx3 images
    folder_path = 'datasets/RealSR(V3)/All/Train/3_LR_sub'
    lmdb_path = 'datasets/RealSR(V3)/All/Train/3_LR_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)
    #
    # # LRx4 images
    # folder_path = 'datasets/RealSR(V3)/All/Train/4_LR_sub'
    # lmdb_path = 'datasets/RealSR(V3)/All/Train/4_LR_sub.lmdb'
    # img_path_list, keys = prepare_keys_realsr(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # LRx8 images
    # folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X8_sub'
    # lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X8_sub.lmdb'
    # img_path_list, keys = prepare_keys_div2k(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)


def prepare_keys_realsr(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys

def create_lmdb_for_drealsr():
    # LRx2 images
    # # HR x2 images  change dataset path
    folder_path = 'datasets/DRealSR/x2/Train_x2/train_HR'
    lmdb_path = 'datasets/DRealSR/x2/Train_x2/train_HR.lmdb'
    img_path_list, keys = prepare_keys_drealsr(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)
    #
    # # LRx2 images
    folder_path = 'datasets/DRealSR/x2/Train_x2/train_LR'
    lmdb_path = 'datasets/DRealSR/x2/Train_x2/train_LR.lmdb'
    img_path_list, keys = prepare_keys_drealsr(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # # HR x3 images  change dataset path
    folder_path = 'datasets/DRealSR/x3/Train_x3/train_HR'
    lmdb_path = 'datasets/DRealSR/x3/Train_x3/train_HR.lmdb'
    img_path_list, keys = prepare_keys_drealsr(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)
    #
    # # LRx3 images
    folder_path = 'datasets/DRealSR/x3/Train_x3/train_LR'
    lmdb_path = 'datasets/DRealSR/x3/Train_x3/train_LR.lmdb'
    img_path_list, keys = prepare_keys_drealsr(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # # HR x4 images  change dataset path
    folder_path = 'datasets/DRealSR/x4/Train_x4/train_HR'
    lmdb_path = 'datasets/DRealSR/x4/Train_x4/train_HR.lmdb'
    img_path_list, keys = prepare_keys_drealsr(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)
    #
    # # LRx4 images
    folder_path = 'datasets/DRealSR/x4/Train_x4/train_LR'
    lmdb_path = 'datasets/DRealSR/x4/Train_x4/train_LR.lmdb'
    img_path_list, keys = prepare_keys_drealsr(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)




def prepare_keys_drealsr(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys

def create_lmdb_for_reds():
    """Create lmdb files for REDS dataset.

    Usage:
        Before run this script, please run `merge_reds_train_val.py`.
        We take two folders for example:
            train_sharp
            train_sharp_bicubic
        Remember to modify opt configurations according to your settings.
    """
    # train_sharp
    folder_path = 'datasets/REDS/train_sharp'
    lmdb_path = 'datasets/REDS/train_sharp_with_val.lmdb'
    img_path_list, keys = prepare_keys_reds(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # train_sharp_bicubic
    folder_path = 'datasets/REDS/train_sharp_bicubic'
    lmdb_path = 'datasets/REDS/train_sharp_bicubic_with_val.lmdb'
    img_path_list, keys = prepare_keys_reds(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_reds(folder_path):
    """Prepare image path list and keys for REDS dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=True)))
    keys = [v.split('.png')[0] for v in img_path_list]  # example: 000/00000000

    return img_path_list, keys


def create_lmdb_for_vimeo90k():
    """Create lmdb files for Vimeo90K dataset.

    Usage:
        Remember to modify opt configurations according to your settings.
    """
    # GT
    folder_path = 'datasets/vimeo90k/vimeo_septuplet/sequences'
    lmdb_path = 'datasets/vimeo90k/vimeo90k_train_GT_only4th.lmdb'
    train_list_path = 'datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
    img_path_list, keys = prepare_keys_vimeo90k(folder_path, train_list_path, 'gt')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # LQ
    folder_path = 'datasets/vimeo90k/vimeo_septuplet_matlabLRx4/sequences'
    lmdb_path = 'datasets/vimeo90k/vimeo90k_train_LR7frames.lmdb'
    train_list_path = 'datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
    img_path_list, keys = prepare_keys_vimeo90k(folder_path, train_list_path, 'lq')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_vimeo90k(folder_path, train_list_path, mode):
    """Prepare image path list and keys for Vimeo90K dataset.

    Args:
        folder_path (str): Folder path.
        train_list_path (str): Path to the official train list.
        mode (str): One of 'gt' or 'lq'.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    with open(train_list_path, 'r') as fin:
        train_list = [line.strip() for line in fin]

    img_path_list = []
    keys = []
    for line in train_list:
        folder, sub_folder = line.split('/')
        img_path_list.extend([osp.join(folder, sub_folder, f'im{j + 1}.png') for j in range(7)])
        keys.extend([f'{folder}/{sub_folder}/im{j + 1}' for j in range(7)])

    if mode == 'gt':
        print('Only keep the 4th frame for the gt mode.')
        img_path_list = [v for v in img_path_list if v.endswith('im4.png')]
        keys = [v for v in keys if v.endswith('/im4')]

    return img_path_list, keys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='div2k',
        help=("Options: 'DIV2K', 'REDS', 'Vimeo90K' 'realsr', 'drealsr' "
              'You may need to modify the corresponding configurations in codes.'))
    args = parser.parse_args()
    dataset = args.dataset.lower()
    if dataset == 'div2k':
        create_lmdb_for_div2k()
    elif dataset == 'div2k_flickr2k':
        create_lmdb_for_div2k_flickr2k()
    elif dataset == 'realsr':
        create_lmdb_for_realsr()
    elif dataset == 'drealsr':
        create_lmdb_for_drealsr()
    elif dataset == 'reds':
        create_lmdb_for_reds()
    elif dataset == 'vimeo90k':
        create_lmdb_for_vimeo90k()
    else:
        raise ValueError('Wrong dataset.')
