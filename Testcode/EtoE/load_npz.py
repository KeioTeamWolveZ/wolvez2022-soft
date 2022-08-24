import numpy as np
import glob

dir_path = sorted(glob.glob("results/*"))
dir_path = dir_path[-1] #ここまでが最新時間のフォルダまでのパスとなる．
file_path = sorted(glob.glob(dir_path+"/planning/learn1/planning_npz/*")) #planningフォルダ内のnpzファイルまでのパスを取得
file_path = file_path[-1] #最新のnpzファイルのパスを取得

data = np.load(file_path[-1],allow_pickle=True)
print(data['array_1'])