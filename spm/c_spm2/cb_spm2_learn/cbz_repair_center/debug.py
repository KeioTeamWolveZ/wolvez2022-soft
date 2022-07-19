import glob
import numpy as np

from debug_second_spm import SPM2Open_npz,SPM2Learn,SPM2Evaluate
import debug_constant as ct



train_npz=sorted(glob.glob("/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/b_spm1/b-data/bcca_secondinput/bccp/*"))
spm2_prepare = SPM2Open_npz()
data_list_all_win,label_list_all_win = spm2_prepare.unpack(train_npz)
spm2_learn = SPM2Learn()

stack_start = ct.const.STUCK_START # 11
stack_end = ct.const.STUCK_END # 13
planning_npz=sorted(glob.glob("/Users/hayashidekazuyuki/Desktop/Git_Win_Air/wolvez2022/spm/b_spm1/b-data/bcca_secondinput/bccp/2022-07-14_00-35-35.npz"))
stack_info = None
spm2_learn.start(data_list_all_win,label_list_all_win,fps=30,alpha=5.0,stack_appear=stack_start,stack_disappear=stack_end,stack_info=stack_info)#どっちかは外すのがいいのか
model_master,label_list_all_win,scaler_master=spm2_learn.get_data()
nonzero_w, nonzero_w_label, nonzero_w_num = spm2_learn.get_nonzero_w()
print("feature_names",np.array(nonzero_w_label,dtype=object).reshape(6,1))
feature_names = nonzero_w_label