from pickletools import int4
from typing import List
import cv2
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
from time import time
from FEATURE import Feature_img

class ReadFeaturedImg():
    """画像読込関数
    
    Args:
        importPath (str): Original img path
        saveDir (str): Save directory path that allowed tmp
        Save(bool):Save or not, defalt:False
    """
    def __init__(self, importPath:str=None, saveDir:str=None, Save:any=False):
        self.imp_p = importPath
        self.sav_d = saveDir
        self.save = Save
    
    def feature_img(self, frame_num, feature_names=None):
        '''Change to treated img
        Args:
            frame_num(int):Frame number or time
            feature_name(str):
        Return:
            fmg_list(list):List of featured img paths
        '''
        self.treat = Feature_img(self.imp_p, frame_num, self.sav_d)
        if feature_names == None:
            self.treat.normalRGB()
            self.treat.vari()
            self.treat.rgbvi()
            self.treat.grvi()
            self.treat.ior()
            self.treat.enphasis()
            self.treat.edge()
            self.treat.hsv()
            self.treat.r()
            self.treat.b()
            self.treat.g()
            self.treat.rb()
            self.treat.gb()
            self.treat.rg()
            
        else:
            for feature_name in feature_names:
                if feature_name == "normalRGB":
                    self.treat.normalRGB()
                elif feature_name == "vari":
                    self.treat.vari()
                elif feature_name == "rgbvi":
                    self.treat.rgbvi()
                elif feature_name == "grvi":
                    self.treat.grvi()
                elif feature_name == "ior":
                    self.treat.ior()
                elif feature_name == "enphasis":
                    self.treat.enphasis()
                elif feature_name == "edge":
                    self.treat.edge()
                elif feature_name == "hsv":
                    self.treat.hsv()
                elif feature_name == "red":
                    self.treat.r()
                elif feature_name == "blue":
                    self.treat.b()
                elif feature_name == "green":
                    self.treat.g()
                elif feature_name == "purple":
                    self.treat.rb()
                elif feature_name == "emerald":
                    self.treat.gb()
                elif feature_name == "yellow":
                    self.treat.rg()
                else:
                    self.treat.other()
        fmg_list = self.treat.output()
        
        return fmg_list


    def read_img(self, path):
        #print("===== func read_img starts =====")
        self.img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        self.img = self.img[int(0.225*self.img.shape[0]):int(0.75*self.img.shape[0])]
        # 読み込めないエラーが生じた際のロバスト性も検討したい
        return self.img