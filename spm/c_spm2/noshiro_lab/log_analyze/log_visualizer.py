import os
import csv
import numpy as np
import matplotlib.pyplot as plt
no=5
current_dir=os.getcwd()
csv_path=current_dir+f"/spm/c_spm2/noshiro_lab/log_analyze/results/risk_no{no}_thre.csv"
fig_path=current_dir+f"/spm/c_spm2/noshiro_lab/log_analyze/results/risk_{no}_thre_bottom_up456.jpg"
# with open(csv_path) as f:
#     reader = csv.reader(f)
#     for row in reader:
#         csv_data_master.append(row)

# csv_data_master=np.array(csv_data_master)

csv_data_master=np.loadtxt(csv_path, delimiter=',')
first_numbers=csv_data_master[0]
csv_data_master[2:]+=np.abs(first_numbers)
csv_data_master=csv_data_master.T
for i, window in enumerate(csv_data_master):
    if i>=3 and i<6:
        plt.plot(np.arange(0,len(window))[5:],window[5:],label=f"win_{i+1}")
plt.title(f"experiment no.{no}")
plt.xlabel("frames")
plt.ylabel("risk")
plt.legend()
plt.savefig(fig_path,dpi=500)