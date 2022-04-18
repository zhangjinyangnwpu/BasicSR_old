import os
import cv2
path = "datasets/DIV2K/DIV2K_train_HR"
names = os.listdir(path)
dest_path = "datasets/DIV2K/DIV2K_train_LR_bicubic/X8"
os.makedirs(dest_path,exist_ok=True)

for name in names:
    name_v = os.path.join(path,name)
    img = cv2.imread(name_v)
    w,h,c = img.shape
    img_r = cv2.resize(img,(h//8,w//8))
    cv2.imwrite(os.path.join(dest_path,name[:4]+'x8.png'),img_r)
    # exit()