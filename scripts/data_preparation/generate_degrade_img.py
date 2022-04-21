import os
import glob
from PIL import Image
import torchvision

from basicsr.utils import SRMDPreprocessing

# 在代码修改开头的数据配置，然后代码根目录，执行 python scripts/data_preparation/generate_degrade_img.py

root_path = './datasets/sr' # 修改这个目录，指定为数据的根目录，可以用 ln -s 指令和代码目录的datasets文件夹关联
dataset_names = ['Urban100'] # div2k set5 set14 B100 Urban100 可以处理的数据集
scales = [2,3,4] # 不同大小的scale
kernel_widths = [0,0.6,1.2,1.8] # 不同的同向高斯核宽度
blur_types = ['iso_gaussian']  # iso_gaussian or aniso_gaussian，高斯核类型，aniso_gaussian还未处理

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
    sig_min = 0.2  # training 0.2 for x2, 0.2 for x3, 0.2 for x4 for iso_gaussian
    sig_max = 2.0  # training 2.0 for x2, 3.0 for x2, 4.0 for x4 for iso_gaussian
    lambda_1 = 0.2  # test with a cetrain value for aniso_gaussian
    lambda_2 = 4.0  # test with a cetrain value for aniso_gaussian
    theta = 0  # angle for aniso_gaussian, set with angle when testing
    lambda_min = 0.2  # training 0.2 for x2,x3,x4 for aniso_gaussian
    lambda_max = 4.0  # training 4.0 for x2,x3,x4 for aniso_gaussian
    noise = 0  # random for training and testing for valiation

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
                        sig_min=sig_min,
                        sig_max=sig_max,
                        lambda_1=lambda_1,
                        lambda_2=lambda_2,
                        theta=theta,
                        lambda_min=lambda_min,
                        lambda_max=lambda_max,
                        noise=noise
                    )

                    for name in hr_names:
                        hr = Image.open(name)
                        hr = torchvision.transforms.ToTensor()(hr)
                        hr = hr.unsqueeze(0).cuda()
                        lr, _ = degrade(hr, random=False)
                        print(hr.shape,lr.shape)
                        lr = lr.squeeze(0).cpu()
                        lr_img = torchvision.transforms.ToPILImage()(lr)
                        lr_img.save(os.path.join(save_path, os.path.basename(name)))
                        # break
            elif blur_type == 'aniso_gaussian':
                kernel_size = 11  # gaussian kernel size
                degrade_type = '{}_lambda1_{}_lambda2_{}_theta_{}'.format(blur_type,lambda_1,lambda_2,theta)
            else:
                raise TypeError




