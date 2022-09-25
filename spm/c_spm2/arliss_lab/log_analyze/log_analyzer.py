import numpy as np
import csv

no=1
log_path="/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/arliss_lab/log_analyze/planning_result.txt"
csv_path="/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/arliss_lab/log_analyze/risk.csv"

"""
収集する情報
(4n+1行目)
・タイムスタンプ
・緯度Lat
・経度Lng
(4n+2行目)
・Goal Distance
・Goal Angle
・q
(4n+3行目)
・Risk
・Threshold_risk
・Boolean risk
(4n+4行目)
空欄
"""

with open(log_path) as f:
    for i,line in enumerate(f):
        i+=1
        if i%4==1:
            timestamp=line[:line.find(",Time")]
            Lat=line[line.find("Lat:")+len("Lat:"):line.find(",Lng")]
            Lng=line[line.find("Lng:")+len("Lat:"):-2]

        elif i%4==2:
            goalDistance=line[line.find("Goal Distance:")+len("Goal Distance:"):line.find(",Goal Angle:")]
            goalAngle=line[line.find("Goal Angle:")+len("Goal Angle:"):line.find(",    q:")]
            q=line[line.find("q:")+len("q:"):-2]

        elif i%4==3:
            risk_raw=line[line.find("Risk:[")+len("Risk:["):line.find("],threadshold_risk:")]
            risk_list=risk_raw.split(" ")
            risk=[]
            for r in risk_list:
                if r != "":
                    risk.append(r)

            thre_raw=line[line.find(",threadshold_risk:[")+len(",threadshold_risk:["):line.find("],max_risk:")]
            thre_list=thre_raw.split(" ")
            thre=[]
            for t in thre_list:
                if t != "":
                    thre.append(t)

            bool_raw=line[line.find("boolean_risk:[")+len("boolean_risk:["):-3]
            bool_list=bool_raw.split(" ")
            bool=[]
            for b in bool_list:
                if b != "":
                    bool.append(b)

        elif i%4==0:
            result=[timestamp,Lat,Lng,goalDistance,goalAngle,q,risk[0],risk[1],risk[2],thre[0],thre[1],thre[2],bool[0],bool[1],bool[2]]
            result_no_space=[]
            for element in result:
                result_no_space.append(element.replace(" ",""))
            result=result_no_space
            with open(csv_path,'a') as f2:
                writer = csv.writer(f2)
                try:
                    result.remove(" ").remove("")
                except ValueError:
                    pass
                writer.writerow(result)