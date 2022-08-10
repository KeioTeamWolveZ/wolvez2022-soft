import numpy as np
import cv2
from cv2 import aruco
import sys
import time


def main(args):
    """
    引数0→写真
    引数1→動画
    """
    cap = cv2.VideoCapture(0)
    if args == "0":
        for i in range (1, 51):
            ret,img = cap.read()
            cv2.imshow("image", img)
            cv2.imwrite(f'stack_pic2/{time.time():.0f}.jpg', img)
            time.sleep(2)
            if i%5==0:
                time.sleep(8)
            cv2.destroyWindow("image")
if __name__ == '__main__':
    args = sys.argv
    main(args[1])