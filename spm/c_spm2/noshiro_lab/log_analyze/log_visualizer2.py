import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pprint import pprint

current_dir=os.getcwd()


csv_paths=sorted(glob(current_dir+f"/spm/c_spm2/noshiro_lab/log_analyze/results/0827/csv/*"))
path_0827=current_dir+"/spm/c_spm2/noshiro_lab/log_analyze/log/0827"
log_paths=sorted(glob(path_0827+"/*/planning_result.txt"))
pprint(csv_paths)

def get_threshold_nd(csv_data_master):
    threshold_nd=[]
    for i in range(len(csv_data_master.T)):
        i+=1
        roi=csv_data_master.T[:i]
        threshold_nd.append(np.average(np.array(roi))+np.std(np.array(roi)))
    return threshold_nd

def get_threshold_nd2(csv_data_master):
    threshold_nd2=[]
    for i in range(len(csv_data_master.T)):
        i+=1
        roi=csv_data_master.T[:i]
        threshold_nd2.append(np.average(np.array(roi))+2*np.std(np.array(roi)))
    return threshold_nd2

def get_q75(csv_data_master):
    q75s=[]
    for i in range(len(csv_data_master.T)):
        i+=1
        roi=csv_data_master.T[:i]
        q75, q25 = np.percentile(roi, [75 ,25])
        q75s.append(q75)
    return q75s

def get_average(csv_data_master):
    aves=[]
    for i in range(len(csv_data_master.T)):
        ave=np.average(csv_data_master.T[i])
        aves.append(ave)
    return aves

def get_average_accum(csv_data_master):
    accum_aves=[]
    for i in range(len(csv_data_master.T)):
        i+=1
        roi=csv_data_master.T[:i]
        accum_ave = np.average(roi)
        accum_aves.append(accum_ave)
    return accum_aves

for (csv_path, txt_path) in zip(csv_paths,log_paths):
    no=re.sub(r"\D","",csv_path).replace("2019202220827","")
    txt_name=txt_path[txt_path.find("2022-08-27_")+len("2022-08-27_"):txt_path.find("/planning_result.txt")]
    fig_path=current_dir+f"/spm/c_spm2/noshiro_lab/log_analyze/results/0827/graph_geta_thre/risk_{txt_name}_geta_thre.jpeg"

    csv_data_master=np.loadtxt(csv_path, delimiter=',')
    # 下駄を履かせる
    first_numbers=csv_data_master[0]
    csv_data_master[2:]+=np.abs(first_numbers)

    csv_data_master=csv_data_master.T
    for i, window in enumerate(csv_data_master[:6]):
        if i>=3:
            plt.plot(np.arange(0,len(window)),window,label=f"win_{i+1}")

    # thresholdを再現計算
    accum_ave=get_average_accum(csv_data_master)
    plt.plot(np.arange(0,len(csv_data_master[6])),accum_ave,label="thre ave (estimated)",lw=4)
    threshold_nd=get_threshold_nd(csv_data_master)
    plt.plot(np.arange(0,len(csv_data_master[6])),threshold_nd,label="thre mu+sigma (estimated)",lw=4)
    threshold_nd2=get_threshold_nd2(csv_data_master)
    plt.plot(np.arange(0,len(csv_data_master[6])),threshold_nd2,label="thre mu+2sigma (estimated)",lw=4)
    q75=get_q75(csv_data_master)
    plt.plot(np.arange(0,len(csv_data_master[6])),q75,label="thre q75 (estimated)",lw=4)
    ave=get_average(csv_data_master)
    plt.plot(np.arange(0,len(csv_data_master[6])),ave,label="thre temporal_ave (estimated)",lw=4)

    # plt.plot(np.arange(0,len(csv_data_master[6])),csv_data_master[6],label="threshold_risk",lw=4)
    # plt.plot(np.arange(0,len(csv_data_master[7])),csv_data_master[7],label="max_risk",lw=4)
    plt.title(f"experiment {txt_name} with geta")
    plt.xlabel("frames")
    plt.ylabel("risk")
    plt.legend()
    plt.savefig(fig_path,dpi=500)
    plt.cla()