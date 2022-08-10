import os
import cv2
from glob import glob
from pprint import pprint
import numpy as np

current_dir=os.getcwd()
raw_images_dir=current_dir+"/spm/c_spm2/noshiro_lab/raw_images"
raw_images=sorted(glob(raw_images_dir+"/*"))
resized_images_dir=current_dir+"/spm/c_spm2/noshiro_lab/resized_images"
pprint(raw_images)

w,h=640,480

for i, raw_image in enumerate(raw_images):
    img=cv2.imread(raw_image)
    resized = cv2.resize(img, dsize=(640, 480))
    cv2.imwrite(resized_images_dir+f"/r_img{i}.jpeg",resized)

resized_images_dir=current_dir+"/spm/c_spm2/noshiro_lab/resized_images"
resized_images=sorted(glob(resized_images_dir+"/*"))


affined_images_dir=current_dir+"/spm/c_spm2/noshiro_lab/affined_images"

for i, resized_image in enumerate(resized_images):
    for j in [0,1,2]:
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), np.random.randint(0,90), 0.5)
        affine_img = cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_WRAP)
        cv2.imwrite(affine_img+f'/img{i}_ver{j}.jpeg', affine_img)
