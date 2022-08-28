import numpy as np
import glob
import matplotlib.pyplot as plt

dir_path = sorted(glob.glob('pre_data_new_14/*'))
feature_name = 'normalRGB'
win = 5

class FetureValueHistory():

    def __init__(self, dir_path:str=None):
        self.file_path = sorted(glob.glob(dir_path+'/*'))
    
    def set_params(self, name:str='normalRGB', win:int=5):
        self.feature_name = name
        self.win = win

    def __set_box(self, data:dict) -> dict:
        self.values = data[self.feature_name][f'win_{self.win}']
        self.keys = [key for key in self.values.keys()]
        self.values_dict = {}
        for key in self.keys:
            self.values_dict[key] = []
        return self.values_dict
    
    def __load_file(self, path) -> dict:
        # for file_path in self.dir_path:
        raw_data = np.load(path,allow_pickle=True)
        all_data = raw_data['array_1'][0]
            # data = all_data[self.feature_name][f"win_{self.win}"]
        return all_data
    
    def __get_data(self, data:dict) -> dict:
        for key in self.keys:
            self.values_dict[key].append(data[key])
        return self.values_dict

    
    def __values_list(self):
        self.frame = [k for k in range(len(self.file_path))]
        for k, path in enumerate(self.file_path):
            all_data = self.__load_file(path=path)
            if k == 0:
                self.__set_box(data=all_data)
            self.__get_data(data=all_data)
    
    def data_logger(self):
        self.__values_list()
        for i, key in enumerate(self.keys):
            plt.figure(figsize=(10,5))
            plt.subplot(int(f'23{i+1}'))
            plt.plot(self.values_dict[key], self.frame)
        plt.show()



# print(dir_path)
# dir_path = dir_path[-1] #ここまでが最新時間のフォルダまでのパスとなる．
# print(dir_path)
# file_path = sorted(glob.glob(dir_path + "/camera_result/second_spm/learn1/*")) #planningフォルダ内のnpzファイルまでのパスを取得
# file_path = file_path[-1] #最新のnpzファイルのパスを取得
# print(file_path)

# file_path = sorted(glob.glob("/home/pi/Desktop/camera_result/planning/learn1/planning_npz/*"))
# file_path = file_path[-1]
# print(file_path)

