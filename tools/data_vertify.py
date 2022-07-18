import cv2
import glob
import os

# data_path_hr = "../datasets/RealSR_V3/Test/2/HR"
# data_path_lr = "../datasets/RealSR_V3/Test/2/LR"


# hrs = glob.glob(os.path.join(data_path_hr,'*.png'))
# lrs = glob.glob(os.path.join(data_path_lr,'*.png'))

# hrs.sort()
# lrs.sort()

# for hr_name,lr_name in zip(hrs,lrs):
#     # print(os.path.basename(hr_name),os.path.basename(lr_name))
#     hr = cv2.imread(hr_name)
#     lr = cv2.imread(lr_name)
#     print(hr.shape,lr.shape)
#     assert hr.shape[0]==lr.shape[0]*2
#     assert hr.shape[1]==lr.shape[1]*2


pp = "/mnt/hdd/zhangkang/zjy/BasicSR/experiments/debug_PASR_RealSR_x2_50k_20220620/visualization/Nikon040/Nikon040_8.png"

img = cv2.imread(pp)
print(img.shape)


