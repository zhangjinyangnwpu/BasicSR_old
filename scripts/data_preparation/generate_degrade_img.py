import os
import glob

import cv2
from PIL import Image
import torchvision
import scipy.io as sio
from matplotlib import pyplot as plt
import scipy.misc
import imageio

from basicsr.utils import SRMDPreprocessing

# 在代码修改开头的数据配置，然后代码根目录，执行 python scripts/data_preparation/generate_degrade_img.py

root_path = './datasets' # 修改这个目录，指定为数据的根目录，可以用 ln -s 指令和代码目录的datasets文件夹关联
dataset_names = ['set5'] # div2k set5 set14 B100 Urban100 可以处理的数据集
scales = [2] # 不同大小的scale
blur_types = ['iso_gaussian']  # iso_gaussian or aniso_gaussian，高斯核类型，aniso_gaussian还未处理

# for iso_gaussian
kernel_widths = [0,0.6,1.2,1.8] # 不同的同向高斯核宽度

# for aniso_gaussian
lambda1s = [0.5]
lambda2s = [3]
thetas = [60]
noises = [30]

for dataset_name in dataset_names:
    if dataset_name == 'div2k':
        path_root_div2k = os.path.join(root_path,'DIV2K')
        hr_names = glob.glob(os.path.join(path_root_div2k,'DIV2K_train_HR','0[89]*.png'))
        hr_names.sort()
        hr_names = hr_names[1:]
        save_path_root = os.path.join(path_root_div2k, 'DIV2K_train_HR_with_degrade')
    elif dataset_name == 'set5':
        path_root_set5 = os.path.join(root_path, 'Classical/Set5/')
        hr_names = glob.glob(os.path.join(path_root_set5, 'GTmod12', '*.png'))
        hr_names.sort()
        save_path_root = os.path.join(path_root_set5, 'GTmod12_with_degrade')
    elif dataset_name == 'set14':
        path_root_set14 = os.path.join(root_path, 'Classical/Set14/')
        hr_names = glob.glob(os.path.join(path_root_set14,'GTmod12','*.png'))
        hr_names.sort()
        save_path_root = os.path.join(path_root_set14, 'GTmod12_with_degrade')
    elif dataset_name == 'B100':
        path_root_b100 = os.path.join(root_path, 'Classical/BSDS100/')
        hr_names = glob.glob(os.path.join(path_root_b100, '*.png'))
        hr_names.sort()
        save_path_root = os.path.join(path_root_b100, 'BSDS100_with_degrade')
    elif dataset_name == 'Urban100':
        path_root_urban100 = os.path.join(root_path, 'Classical/urban100/')
        hr_names = glob.glob(os.path.join(path_root_urban100, '*.png'))
        hr_names.sort()
        save_path_root = os.path.join(path_root_urban100, 'urban100_with_degrade')
    else:
        raise TypeError

    os.makedirs(save_path_root, exist_ok=True)
    mode = 'bicubic'

    for scale in scales:
        for blur_type in blur_types:
            if blur_type == 'iso_gaussian':
                for sig in kernel_widths:
                    kernel_size = 21  # gaussian kernel size
                    degrade_type = '{}_sig_{}'.format(blur_type,sig)
                    save_path = os.path.join(save_path_root, 'X{}'.format(scale), degrade_type)
                    os.makedirs(save_path, exist_ok=True)

                    degrade = SRMDPreprocessing(
                        scale=scale,
                        mode=mode,
                        kernel_size=kernel_size,
                        blur_type=blur_type,
                        sig=sig,
                        sig_min=0.2,
                        sig_max=2.0,
                        lambda_1=0.2,
                        lambda_2=0.4,
                        theta=0,
                        lambda_min=0.2,
                        lambda_max=4.0,
                        noise=0
                    )

                    for name in hr_names:
                        hr = Image.open(name)
                        hr = torchvision.transforms.ToTensor()(hr)
                        hr = hr.unsqueeze(0).cuda()
                        print(hr.shape)
                        lr, kernel = degrade(hr, random=False)
                        if kernel is not None:
                            kernel = kernel.cpu()[0]
                        print(hr.shape,lr.shape)
                        lr = lr.squeeze(0).cpu()
                        lr_img = torchvision.transforms.ToPILImage()(lr)
                        lr_img.save(os.path.join(save_path, os.path.basename(name)))
                        if kernel is not None:
                            plt.clf()
                            plt.clf()
                            plt.axis('off')
                            plt.imshow(kernel, cmap='binary_r', interpolation='bicubic')  # binary_r binary
                            plt.savefig(os.path.join(save_path, 'kernel.jpg'), bbox_inches='tight', pad_inches=0)
                        # break
            elif blur_type == 'aniso_gaussian':
                kernel_size = 11  # gaussian kernel size
                for lambda1 in lambda1s:
                    for lambda2 in lambda2s:
                        for theta in thetas:
                            for noise in noises:
                                degrade_type = '{}_lambda1_{}_lambda2_{}_theta_{}_noise_{}'.format(blur_type, lambda1,lambda2,theta,noise)
                                save_path = os.path.join(save_path_root, 'X{}'.format(scale), degrade_type)
                                os.makedirs(save_path, exist_ok=True)
                                degrade = SRMDPreprocessing(
                                    scale=scale,
                                    mode=mode,
                                    kernel_size=kernel_size,
                                    blur_type=blur_type,
                                    sig=0,
                                    sig_min=0.2,
                                    sig_max=2.0,
                                    lambda_1=lambda1,
                                    lambda_2=lambda2,
                                    theta=theta,
                                    lambda_min=0.2,
                                    lambda_max=4.0,
                                    noise=noise
                                )
                                for name in hr_names:
                                    hr = Image.open(name)
                                    hr = torchvision.transforms.ToTensor()(hr)
                                    hr = hr.unsqueeze(0).cuda()
                                    lr, kernel = degrade(hr, random=False)
                                    kernel = kernel.cpu()[0]
                                    print(hr.shape, lr.shape,kernel.shape)
                                    lr = lr.squeeze(0).cpu()
                                    lr_img = torchvision.transforms.ToPILImage()(lr)
                                    lr_img.save(os.path.join(save_path, os.path.basename(name)))
                                    sio.savemat(os.path.join(save_path,'kernel.mat'),{'kernel':kernel})
                                    plt.clf()
                                    plt.axis('off')
                                    plt.imshow(kernel, cmap='binary_r', interpolation='bicubic')  # binary_r binary
                                    plt.savefig(os.path.join(save_path,'kernel.jpg'), bbox_inches='tight', pad_inches=0)
            else:
                raise TypeError




