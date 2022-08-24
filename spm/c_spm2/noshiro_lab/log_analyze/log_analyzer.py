import numpy as np
import csv

no=1
log_path=f"/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/noshiro_lab/log_analyze/log/planning_no{no}.txt"
csv_path=f"/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/noshiro_lab/log_analyze/results/risk_no{no}.csv"

with open(log_path) as f:
    for line in f:
        start1="Risk:[["
        end1="]"
        start2="["
        end2="]"
        if line.find(start1)!=-1:
            idx_s=line.find(start1)+len(start1)
            idx_e=line.find(end1)
            data=line[idx_s:idx_e]
            data1=data.split(" ")
            try:
                data1.remove('')
            except ValueError:
                pass
            data_6win=[]
            data_6win.append(data1)
        if line.find(start1)==-1:
            idx_s=line.find(start2)+len(start2)
            idx_e=line.find(end2)
            data=line[idx_s:idx_e]
            data2=data.split(" ")
            try:
                data2.remove('')
            except ValueError:
                pass
            data_6win.append(data2)
        
        if len(data_6win)==2:
            data_6win=np.array(data_6win).flatten()
            print(data_6win)        
            with open(csv_path,'a') as f2:
                writer = csv.writer(f2)
                writer.writerow(data_6win)
