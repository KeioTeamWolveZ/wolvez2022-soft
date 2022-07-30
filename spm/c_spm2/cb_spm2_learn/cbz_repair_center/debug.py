import glob
import numpy as np

from debug_second_spm import SPM2Open_npz,SPM2Learn,SPM2Evaluate
import debug_constant as ct



train_npz=sorted(glob.glob("/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/b_spm1/b-data/bcca_secondinput/bccp/*"))[:50]
spm2_prepare = SPM2Open_npz()
data_list_all_win,label_list_all_win = spm2_prepare.unpack(train_npz)
spm2_learn = SPM2Learn()

f1 = 10# ct.const.STUCK_START # 11
f2 = 20# ct.const.STUCK_END # 13
planning_npz=sorted(glob.glob("/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/b_spm1/b-data/bcca_secondinput/bccp/2022-07-14_00-35-35.npz"))
f1f2_array_window_custom = None
spm2_learn.start(data_list_all_win,label_list_all_win,f1, f2,alpha=5.0,f1f2_array_window_custom=f1f2_array_window_custom)#どっちかは外すのがいいのか
model_master,label_list_all_win,scaler_master=spm2_learn.get_data()
nonzero_w, nonzero_w_label, nonzero_w_num = spm2_learn.get_nonzero_w()
print("feature_names",np.array(nonzero_w_label,dtype=object).reshape(6,1))
feature_names = nonzero_w_label