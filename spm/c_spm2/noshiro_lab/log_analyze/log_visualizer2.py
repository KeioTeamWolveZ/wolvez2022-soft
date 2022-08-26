import os
import csv
import numpy as np
import matplotlib.pyplot as plt
no=5
current_dir=os.getcwd()
csv_path=current_dir+f"/spm/c_spm2/noshiro_lab/log_analyze/results/risk_no{no}_thre.csv"
fig_path=current_dir+f"/spm/c_spm2/noshiro_lab/log_analyze/results/risk_{no}_thre.jpg"

csv_data_master=np.loadtxt(csv_path, delimiter=',')
csv_data_master=csv_data_master.T

for i, window in enumerate(csv_data_master[:6]):
    plt.plot(np.arange(0,len(window)),window,label=f"win_{i+1}")
plt.plot(np.arange(0,len(csv_data_master[6])),csv_data_master[6],label="threshold_risk",lw=4)
plt.plot(np.arange(0,len(csv_data_master[7])),csv_data_master[7],label="max_risk",lw=4)
plt.title(f"experiment no.{no}")
plt.xlabel("frames")
plt.ylabel("risk")
plt.legend()
plt.savefig(fig_path,dpi=500)