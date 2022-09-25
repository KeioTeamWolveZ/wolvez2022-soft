import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read csv
csv_path="/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/arliss_lab/log_analyze/risk.csv"
csv_data=pd.read_csv(csv_path)
csv_data=csv_data.set_axis(["timestamp","Lat","Lng","gDist","gAng","q","r_w1","r_w2","r_w3","th_w1","th_w2","th_w3","bo_w1","bo_w2","bo_w3"],axis=1)

# 緯度経度（移動履歴）
plt.scatter(-csv_data["Lng"],csv_data["Lat"],label="path",cmap="jet",s=3)
plt.scatter(-119.09875500,40.88215833,s=16,c="red",label="landing point")
plt.scatter(-119.09871500,40.88220500,s=16,c="green",label="sequence start")
plt.scatter(-119.09569000,40.89076833,s=16,c="black",label="dead point")
plt.title("Position of CanSat")
plt.xlabel("Lng [deg]")
plt.ylabel("Lat [deg]")
plt.legend()
plt.savefig("/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/arliss_lab/log_analyze/results/result_01_LatLng.png")
plt.cla()

# Goal Distance
plt.scatter(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["gDist"],s=3,label="distance")
plt.title("distance between CanSat & Goal")
plt.xlabel("time [s]")
plt.ylabel("distance [m]")
plt.legend()
plt.savefig("/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/arliss_lab/log_analyze/results/result_02_gDist.png")
plt.cla()
# Goal Angle
plt.scatter(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["gAng"],s=3,label="angle")
plt.title("goal Angle")
plt.xlabel("time [s]")
plt.ylabel("Angle [deg]")
plt.legend()
plt.savefig("/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/arliss_lab/log_analyze/results/result_03_gAng.png")
plt.cla()

# q
plt.scatter(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["q"],s=3,label="q")
plt.title("q")
plt.xlabel("time [s]")
plt.ylabel("q")
plt.legend()
plt.savefig("/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/arliss_lab/log_analyze/results/result_04_q.png")
plt.cla()

# risk
# plt.plot(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["r_w1"],label="win_1")
# plt.plot(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["r_w2"],label="win_2")
# plt.plot(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["r_w3"],label="win_3")
plt.scatter(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["r_w1"],s=2,label="win_1")
plt.scatter(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["r_w2"],s=2,label="win_2")
plt.scatter(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["r_w3"],s=2,label="win_3")
plt.title("risk")
plt.xlabel("time [s]")
plt.ylabel("risk")
plt.legend()
plt.savefig("/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/arliss_lab/log_analyze/results/result_05_risk.png",dpi=3000)
plt.cla()

# threshold_risk
plt.scatter(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["th_w1"],s=2,label="win_1")
plt.scatter(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["th_w2"],s=2,label="win_2")
plt.scatter(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["th_w3"],s=2,label="win_3")
plt.title("threshold_risk")
plt.xlabel("time [s]")
plt.ylabel("threshold_risk")
plt.legend()
plt.savefig("/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/arliss_lab/log_analyze/results/result_06_thre.png",dpi=3000)
plt.cla()
# bool_risk
plt.plot(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["bo_w1"],label="win_1")
plt.plot(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["bo_w2"],label="win_2")
plt.plot(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["bo_w3"],label="win_3")
# plt.scatter(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["bo_w1"],s=2,label="win_1")
# plt.scatter(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["bo_w2"],s=2,label="win_2")
# plt.scatter(csv_data["timestamp"]-csv_data["timestamp"][0],csv_data["bo_w3"],s=2,label="win_3")
plt.title("boolean_risk")
plt.xlabel("time [s]")
plt.ylabel("boolean_risk")
plt.legend()
plt.savefig("/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/c_spm2/arliss_lab/log_analyze/results/result_07_bool.png",dpi=3000)
plt.cla()