import os
import glob
from PIL import Image
import torchvision

from basicsr.utils import SRMDPreprocessing

path_root = os.path.join('./datasets/sr','DIV2K')
hr_names = glob.glob(os.path.join(path_root,'DIV2K_train_HR','0[89]*.png'))
hr_names.sort()
hr_names = hr_names[1:]

scale = 4
mode = 'bicubic'
kernel_size = 21  # gaussian kernel size
blur_type = 'iso_gaussian'  # iso_gaussian or aniso_gaussian
sig = 0.6  # test with a certain value for iso_gaussian
sig_min = 0.2  # training 0.2 for x2, 0.2 for x3, 0.2 for x4 for iso_gaussian
sig_max = 2.0  # training 2.0 for x2, 3.0 for x2, 4.0 for x4 for iso_gaussian
lambda_1 = 0.2  # test with a cetrain value for aniso_gaussian
lambda_2 = 4.0  # test with a cetrain value for aniso_gaussian
theta = 0  # angle for aniso_gaussian, set with angle when testing
lambda_min = 0.2  # training 0.2 for x2,x3,x4 for aniso_gaussian
lambda_max = 4.0  # training 4.0 for x2,x3,x4 for aniso_gaussian
noise = 0  # random for training and testing for valiation

save_path_root = os.path.join('./datasets/sr','DIV2K','DIV2K_train_HR_with_degrade')
os.makedirs(save_path_root,exist_ok=True)

if blur_type == 'iso_gaussian':
    degrade_type = '{}_sig_{}'.format(blur_type,sig)
elif blur_type == 'aniso_gaussian':
    degrade_type = '{}_lambda1_{}_lambda2_{}_theta_{}'.format(blur_type,lambda_1,lambda_2,theta)
else:
    raise TypeError

save_path = os.path.join(save_path_root,'X{}'.format(scale),degrade_type)
os.makedirs(save_path,exist_ok=True)

degrade = SRMDPreprocessing(
            scale=scale,
            mode = mode,
            kernel_size = kernel_size,
            blur_type=blur_type,
            sig = sig,
            sig_min=sig_min,
            sig_max=sig_max,
            lambda_1 = lambda_1,
            lambda_2 = lambda_2,
            theta = theta,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            noise=noise
        )

for name in hr_names:
    hr = Image.open(name)
    hr = torchvision.transforms.ToTensor()(hr)
    hr = hr.unsqueeze(0).cuda()
    print(hr.shape)
    lr,_ = degrade(hr,random=False)
    print(lr.shape)
    lr = lr.squeeze(0).cpu()
    lr_img = torchvision.transforms.ToPILImage()(lr)
    lr_img.save(os.path.join(save_path,os.path.basename(name)))
    # break


