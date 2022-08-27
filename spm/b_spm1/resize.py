import cv2
import numpy as np
from glob import glob
from time import time

paths = sorted(glob("test_resize/*.jpg"))

for k, importPath in enumerate(paths):
    img=cv2.imread(importPath, 1)
    img = img[int(0.025*img.shape[0]):]
    cv2.imwrite(f"test/{time():.3f}.jpg", img)