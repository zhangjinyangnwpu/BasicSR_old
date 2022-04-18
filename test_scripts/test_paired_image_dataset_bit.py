import math
import os
import torchvision.utils
from PIL import Image
from basicsr.data import build_dataloader, build_dataset
import torchvision,torch

def main(mode='folder'):
    """Test paired image dataset.

    Args:
        mode: There are three modes: 'lmdb', 'folder', 'meta_info_file'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'train'

    opt['name'] = 'DIV2K'
    opt['type'] = 'PairedImageDatasetBit'
    if mode == 'folder':
        opt['dataroot_gt'] = 'datasets/DIV2K/DIV2K_train_HR_sub'
        opt['dataroot_lq'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
        opt['filename_tmpl'] = '{}'
        opt['io_backend'] = dict(type='disk')
    elif mode == 'meta_info_file':
        opt['dataroot_gt'] = 'datasets/DIV2K/DIV2K_train_HR_sub'
        opt['dataroot_lq'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
        opt['meta_info_file'] = 'basicsr/data/meta_info/meta_info_DIV2K800sub_GT.txt'  # noqa:E501
        opt['filename_tmpl'] = '{}'
        opt['io_backend'] = dict(type='disk')
    elif mode == 'lmdb':
        opt['dataroot_gt'] = 'datasets/DIV2K/DIV2K_train_HR_sub.lmdb'
        opt['dataroot_lq'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb'  # noqa:E501
        opt['io_backend'] = dict(type='lmdb')

    opt['gt_size'] = 128
    opt['use_hflip'] = True
    opt['use_rot'] = True

    opt['use_shuffle'] = True
    opt['num_worker_per_gpu'] = 1
    opt['batch_size_per_gpu'] = 16
    opt['scale'] = 4

    opt['dataset_enlarge_ratio'] = 1

    os.makedirs('tmp', exist_ok=True)

    dataset = build_dataset(opt)
    data_loader = build_dataloader(dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        print(i)
        if i > 5:
            break
        lq = data['lq']
        gt = data['gt']
        lq_path = data['lq_path']
        gt_path = data['gt_path']
        print(lq_path, gt_path)
        def gray_save(img,imgPath,Gray=True):
            grid = torchvision.utils.make_grid(img, nrow=nrow, padding=padding,normalize=False)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            # im.show()
            if Gray:
                im.convert('L').save(imgPath)  # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
            else:
                im.save(imgPath)
        gray_save(lq[:,:,4], f'tmp/g_lq_{i:03d}.png')
        gray_save(gt[:,:,4], f'tmp/g_gt_{i:03d}.png')
        torchvision.utils.save_image(lq[:,:,4], f'tmp/lq_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)
        torchvision.utils.save_image(gt[:,:,4], f'tmp/gt_{i:03d}.png', nrow=nrow, padding=padding, normalize=False)

        


if __name__ == '__main__':
    main()
