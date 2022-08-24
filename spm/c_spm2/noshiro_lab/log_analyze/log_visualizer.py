import csv
import numpy as np
import matplotlib.pyplot as plt
no=1
csv_path=f"/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/noshiro_lab/log_analyze/results/risk_{no}.csv"
fig_path=f"/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/noshiro_lab/log_analyze/results/risk_{no}.jpg"
# with open(csv_path) as f:
#     reader = csv.reader(f)
#     for row in reader:
#         csv_data_master.append(row)

# csv_data_master=np.array(csv_data_master)

csv_data_master=np.loadtxt(csv_path, delimiter=',')
csv_data_master=csv_data_master.T

for i, window in enumerate(csv_data_master):
    plt.plot(np.arange(0,len(window)),window,label=f"win_{i+1}")
plt.title(f"experiment no.{no}")
plt.xlabel("frames")
plt.ylabel("risk")
plt.legend()
plt.savefig(fig_path,dpi=500)