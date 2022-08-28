import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pprint import pprint

current_dir=os.getcwd()


csv_paths=sorted(glob(current_dir+f"/spm/c_spm2/noshiro_lab/log_analyze/results/0827/csv/*"))
pprint(csv_paths)

for csv_path in csv_paths:
    no=re.sub(r"\D","",csv_path).replace("2019202220827","")

    fig_path=current_dir+f"/spm/c_spm2/noshiro_lab/log_analyze/results/0827/graph/risk_{no}_thre.jpg"

    csv_data_master=np.loadtxt(csv_path, delimiter=',')
    csv_data_master=csv_data_master.T

    for i, window in enumerate(csv_data_master[:6]):
        if i>=3:
            plt.plot(np.arange(0,len(window)),window,label=f"win_{i+1}")
    plt.plot(np.arange(0,len(csv_data_master[6])),csv_data_master[6],label="threshold_risk",lw=4)
    plt.plot(np.arange(0,len(csv_data_master[7])),csv_data_master[7],label="max_risk",lw=4)
    plt.title(f"experiment no.{no}")
    plt.xlabel("frames")
    plt.ylabel("risk")
    plt.legend()
    plt.savefig(fig_path,dpi=500)
    plt.cla()