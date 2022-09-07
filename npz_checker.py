import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import sys


# 読み込むフォルダを指定
FOLDER = 'pre_data_new_14'

class FetureValueHistory():
    global FOLDER

    def __init__(self, dir_path:str=None):
        self.file_path = sorted(glob.glob(dir_path+'/*'))
        self.keys:list = None
        self.values_dict:dict = None
    
    def set_params(self, name:str='normalRGB', win:int=5):
        self.feature_name = name
        self.win = win

    def set_box(self, data:dict):
        self.keys = [key for key in data.keys()]
        self.values_dict = {}
        for key in self.keys:
            self.values_dict[key] = []
        return self.values_dict
    
    def load_file(self, path):
        raw_data = np.load(path,allow_pickle=True)
        all_data = raw_data['array_1'][0][self.feature_name][f'win_{self.win}']
        return all_data
    
    def get_data(self, data:dict):
        for key in self.keys:
            self.values_dict[key].append(data[key])
        return self.values_dict

    
    def values_list(self):
        self.frame = [k for k in range(len(self.file_path))]
        for k, path in enumerate(self.file_path):
            all_data = self.load_file(path=path)
            if k == 0:
                self.set_box(data=all_data)
            self.get_data(data=all_data)
    
    def data_logger(self):
        self.values_list()
        plt.figure(figsize=(17,7))
        for i, key in enumerate(self.keys):
            # plt.subplot(int(f'23{i+1}'))
            plt.plot(self.frame, self.values_dict[key], label=key)
            # plt.title(key)
        plt.xlabel('frame')
        plt.ylabel('value')
        plt.legend()
        plt.grid(True)
        plt.title(f'{FOLDER} Feature Values History\n{self.feature_name} win{self.win}')
        # plt.show()
        plt.savefig(f'npz_checker/{FOLDER}/{self.feature_name}-win{self.win}.jpg')
        plt.close()

if len(sys.argv) >= 2:
    feature_name = sys.argv[1]
    FVH = FetureValueHistory(FOLDER)
    FVH.set_params(name=feature_name)
    FVH.data_logger()
else:
    if not os.path.exists(f'npz_checker/{FOLDER}'):
        os.mkdir(f'npz_checker/{FOLDER}')
#     default_names = ["normalRGB","enphasis","edge","hsv","red","blue","green","purple","emerald","yellow"]
    default_names = ["normalRGB","enphasis","edge","vari","rgbvi","grvi","ior","hsv","red","blue","green","purple","emerald","yellow"]
    for name in default_names:
        for win in [4,5,6]:
            FVH = FetureValueHistory(FOLDER)
            FVH.set_params(name=name, win=win)
            FVH.data_logger()

# print(dir_path)
# dir_path = dir_path[-1] #ここまでが最新時間のフォルダまでのパスとなる．
# print(dir_path)
# file_path = sorted(glob.glob(dir_path + "/camera_result/second_spm/learn1/*")) #planningフォルダ内のnpzファイルまでのパスを取得
# file_path = file_path[-1] #最新のnpzファイルのパスを取得
# print(file_path)

# file_path = sorted(glob.glob("/home/pi/Desktop/camera_result/planning/learn1/planning_npz/*"))
# file_path = file_path[-1]
# print(file_path)

