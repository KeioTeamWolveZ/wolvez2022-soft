import numpy as np
import csv
import os

no=5
current_dir=os.getcwd()
log_path=current_dir+f"/spm/c_spm2/noshiro_lab/log_analyze/log/planning_no{no}.txt"
csv_path=current_dir+f"/spm/c_spm2/noshiro_lab/log_analyze/results/risk_no{no}_thre.csv"

with open(log_path) as f:
    for i,line in enumerate(f):
        j=i+1
        if j%4==1 or j%4==2:
            data_win=None
            pass
        elif j%4==3:
            idx_s=line.find("Risk:[")+len("Risk:[")
            idx_e=line.find("]")
            # print("data : ",line[idx_s:idx_e])
            data=line[idx_s:idx_e]
            data_win=data.split(" ")
            thre_s=line.find("threadshold_risk:")+len("threadshold_risk:")
            thre_e=line.find(",max_risk:")
            thre_m_s=line.find(",max_risk:")+len(",max_risk:")
            thre_m_e=line.find(",boolean_risk:")
            print("threadshold_risk : ",line[thre_s:thre_e])
            print("max_risk : ",line[thre_m_s:thre_m_e])
            data_win.append(line[thre_s:thre_e])
            data_win.append(line[thre_m_s:thre_m_e])
            try:
                data_win.remove("")
            except ValueError:
                pass
        elif j%4==0:
            with open(csv_path,'a') as f2:
                writer = csv.writer(f2)
                writer.writerow(data_win)