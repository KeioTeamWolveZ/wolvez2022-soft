import numpy as np

risk_list=[np.array([[0,1,2],[3,4,5]])]#[np.array([[1,2,3],[4,5,6]]),np.array([[1,2,3],[4,5,6]]),np.array([[1,2,3],[4,5,6]]),np.array([[1,2,3],[4,5,6]])]
risk=np.array([[1,2,3],[4,5,6]])
risk_list.append(risk)
print(risk_list)
risk_list=np.array(risk_list)
print(risk_list.shape)
ave=risk_list.mean(axis=0)
print(ave)