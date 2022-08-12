import numpy as np

risk_list_below=[[100,150,200],[150,200,250],[200,250,300],[250,300,350],[300,350,400],[350,400,450]]
def safe_or_not(lower_risk):
    """
    ・入力：下半分のwindowのリスク行列（3*1または1*3？ここはロバストに作ります）
    ・出力：危険=1、安全=0の(入力と同じ次元)
    """
    global risk_list_below
    threshold_risk = np.average(np.array(risk_list_below))+2*np.std(np.array(risk_list_below))
    max_risk=np.max(np.array(risk_list_below))
    print(threshold_risk,max_risk)
    answer_mtx=np.zeros_like(lower_risk)
    for i, risk_scaler in enumerate(lower_risk):
        print(risk_scaler)
        if risk_scaler >= threshold_risk or risk_scaler >= max_risk:
            answer_mtx[i]=1
    return answer_mtx

print(risk_list_below[-1])
print(safe_or_not(risk_list_below[-1]))