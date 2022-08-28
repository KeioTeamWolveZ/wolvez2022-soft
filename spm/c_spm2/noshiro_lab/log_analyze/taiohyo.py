import os
from glob import glob
from pprint import pprint

current_dir=os.getcwd()

path_0827=current_dir+"/spm/c_spm2/noshiro_lab/log_analyze/log/0827"

log_paths=sorted(glob(path_0827+"/*/planning_result.txt"))

for i,log_path in enumerate(log_paths):
    i+=1
    fig_path=current_dir+f"/spm/c_spm2/noshiro_lab/log_analyze/results/0827/graph_bu/risk_{i}_thre_bottom_up.jpg"
    print(log_path[log_path.find("0827/")+len("0827:"):],"  -->>  ",fig_path[fig_path.find("graph_bu/")+len("graph_bu/"):])