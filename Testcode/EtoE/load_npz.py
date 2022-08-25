import numpy as np
import glob

dir_path = sorted(glob.glob("results/*"))
print(dir_path)
dir_path = dir_path[-1] #ここまでが最新時間のフォルダまでのパスとなる．
print(dir_path)
file_path = sorted(glob.glob(dir_path + "/camera_result/second_spm/learn1/*")) #planningフォルダ内のnpzファイルまでのパスを取得
file_path = file_path[-1] #最新のnpzファイルのパスを取得
print(file_path)

# file_path = sorted(glob.glob("/home/pi/Desktop/camera_result/planning/learn1/planning_npz/*"))
# file_path = file_path[-1]
# print(file_path)

data = np.load(file_path,allow_pickle=True)
data = data['array_1']
print(data)
print(len(data[0]))