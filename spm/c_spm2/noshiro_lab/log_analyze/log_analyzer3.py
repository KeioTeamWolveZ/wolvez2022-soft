import numpy as np
import csv

no=4
log_path=f"/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/noshiro_lab/log_analyze/log/planning_no{no}.txt"
csv_path=f"/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/noshiro_lab/log_analyze/results/risk_no{no}_thre.csv"

with open(log_path) as f:
    for i,line in enumerate(f):
        j=i+1
        if j%3==1:
            data_win=None
            pass
        elif j%3==2:
            idx_s=line.find("Risk:[[")+len("Risk:[[")
            idx_e=len(line)-1
            # print("data : ",line[idx_s:idx_e])
            data=line[idx_s:idx_e]
            data_win=data.split(" ")
            try:
                data_win.remove("")
            except ValueError:
                pass
        elif j%3==0:
            idx_s=2
            idx_e=line.find("]]")
            thre_s=line.find("threadshold_risk:")+len("threadshold_risk:")
            thre_e=line.find(",max_risk:")
            thre_m_s=line.find(",max_risk:")+len(",max_risk:")
            thre_m_e=line.find(",boolean_risk:")
            print("threadshold_risk : ",line[thre_s:thre_e])
            print("max_risk : ",line[thre_m_s:thre_m_e])
            # print("data : ",line[idx_s:idx_e])
            data_win.append(line[idx_s:idx_e])
            data_win.append(line[thre_s:thre_e])
            data_win.append(line[thre_m_s:thre_m_e])
            with open(csv_path,'a') as f2:
                writer = csv.writer(f2)
                writer.writerow(data_win)