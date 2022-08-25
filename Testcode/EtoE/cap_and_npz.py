from tempfile import TemporaryDirectory
from xml.dom.pulldom import default_bufsize
from pandas import IndexSlice
import RPi.GPIO as GPIO
import sys
import cv2
import time
import numpy as np
import os
import re
import math
from datetime import datetime
from glob import glob
import shutil
# from math import prod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from first_spm import IntoWindow, LearnDict, EvaluateImg
from second_spm import SPM2Open_npz,SPM2Learn,SPM2Evaluate

from bno055 import BNO055
from motor import motor
from gps import GPS
from lora import lora
from led import led
import constant as ct

class Npz_maker():
    # def __init__(self):
        
    
    def spm_first(self, PIC_COUNT:int=1, relearning:dict=dict(relearn_state=False,f1=ct.const.f1,f3=ct.const.f3)): #ステート4。スパースモデリング第一段階実施。
        if self.spmfirstTime == 0: #時刻を取得してLEDをステートに合わせて光らせる
            self.spmfirstTime = time.time()
            self.RED_LED.led_on()
            self.BLUE_LED.led_on()
            self.GREEN_LED.led_off()
        '''
        CHECK POINT
        ---
        MUST READ:
            1: This IntelliSence
            2: line 331~359 for learn state
            3: line 393~425 for evaluate state
        ------
        EXPLANATION
        ---
        args:
            PIC_COUNT (int) : number of taking pics
            relearning (dict) : dict of informations for relearning; relearn_state (bool), f1 (int) for number not should be counted as stack-pos, f3 (int) for number should be counted include f1
        
        flow:
            First Learning:
                1: PIC_COUNT=1, (global)learn_state=True
                2: PIC_COUNT=50?, (global)learn_state=False
            ReLearning:
                3: PIC_COUNT=1, (global)learn_state=True, relearning={'relearn_state':True, 'f1':int, 'f3':int}\\
                4: PIC_COUNT=50?, (global)learn_state=False, relearning={'relearn_state':True, 'f1':int, 'f3':int}
        
        output:
            New npzs will be in the current "learncount" folder
        '''
        start_time = time.time() #学習用時間計測。学習開始時間
        
        #保存時のファイル名指定（現在は時間）
        now=str(datetime.now())[:19].replace(" ","_").replace(":","-")

        Save = True
        
        # Path that img will be read
        #importPath = path.replace("\\", "/")
        
        # This will change such as datetime
        # print("CURRENT FRAME: "+str(re.findall(".*/frame_(.*).jpg", importPath)[0]))
        
        iw_shape = (2, 3)  #ウィンドウのシェイプ
        D: any
        ksvd: any  # 最初に指定しないと怒られちゃうから
        feature_values = {}

        if self.learn_state:
            print(f"=====LEARNING PHASE{self.learncount}=====")
        else:
            print(f"=====EVALUATING PHASE{self.learncount}=====")
        
        if self.learn_state: #学習モデル獲得     
            if relearning['relearn_state']:  # 再学習に用いる画像パスの指定
                # 一つ前のlearncountファイルの-f3枚目を指定
                try:
                    importPath = sorted(glob(f"results/{self.startTime}/camera_result/planning/learn{self.learncount-1}/planning_pics/planningimg*.jpg"))[-relearning['f3']]
                except IndexError:
                    # ここで学習枚数足りなかったら動作指定（あきらめて１回目と同じ動きするのか、再学習をあきらめるか）
                    print('There are not enough number of pics for ReLearning.')
                    # relearning['relearn_state'] = False  # 再学習用に画像を1枚
            
            if not relearning['relearn_state']:
                #学習用画像を一枚撮影
                '''
                再学習の段階でcamerafirstの値を指定することで
                辞書再作成用の画像撮影の有無を決定
                '''
                T = time.time()
                if not os.path.exists(f"../npz_maker/{T:.0f}"):
                    os.mkdir(f"../npz_maker/{T:.0f}")
                if not os.path.exists(f"../npz_maker/{T:.0f}/dict_pic"):
                    os.mkdir(f"../npz_maker/{T:.0f}/dict_pic")
                if not os.path.exists(f"../npz_maker/{T:.0f}/processed_pic"):
                    os.mkdir(f"../npz_maker/{T:.0f}/processed_pic")
                if self.camerafirst == 0:
                    self.cap = cv2.VideoCapture(0)
                    ret, firstimg = self.cap.read()
                    cv2.imwrite(f"../npz_maker/{T:.0f}/dict_pic/first.jpg",firstimg)
                    self.camerastate = "captured!"
                    self.sensor()
                    self.camerastate = 0
                    self.firstlearnimgcount += 1
                    self.camerafirst = 1
                elif self.camerafirst == 2:
                    '''
                    再撮影をする場合はここに記載
                    '''
                
                importPath = f"../npz_maker/{T:.0f}/dict_pic/first.jpg"
            
            processed_Dir = f"../npz_maker/{T:.0f}/processed_pic"
            iw = IntoWindow(importPath, processed_Dir, Save) #画像の特徴抽出のインスタンス生成
            self.img=cv2.imread(importPath, 1)
            self.img = self.img[int(0.25*self.img.shape[0]):int(0.75*self.img.shape[0])]
            cv2.imwrite(importPath, self.img)
            
            # processing img
            fmg_list = iw.feature_img(frame_num=now) #特徴抽出。リストに特徴画像が入る 
            for fmg in fmg_list:#それぞれの特徴画像に対して処理
                # breakout by windows
                iw_list, window_size = iw.breakout(cv2.imread(fmg, cv2.IMREAD_GRAYSCALE)) #ブレイクアウト
                feature_name = str(re.findall(processed_Dir+"/(.*)_.*_", fmg)[0])
                # print("FEATURED BY: ",feature_name)
                
                
                
                

                for win in range(int(np.prod(iw_shape))): #それぞれのウィンドウに対して学習を実施
                    if win+1 == int((iw_shape[0]-1)*iw_shape[1]) + int(iw_shape[1]/2) + 1:
                        ld = LearnDict(iw_list[win])
                        D, ksvd = ld.generate() #辞書獲得
                        self.dict_list[feature_name] = [D, ksvd]
                        save_name = self.saveDir + f"/learn{self.learncount}/learnimg/{feature_name}_part_{win+1}_{now}.jpg"
                        # cv2.imwrite(save_name, iw_list[win])
            self.learn_state = False

        else:# PIC_COUNT枚撮影
            if self.state == 4:  # 再学習時にステート操作が必要なら追記
                self.spm_f_eval(PIC_COUNT=50, now=now, iw_shape=iw_shape, relearning=relearning) #第2段階用の画像を撮影
                self.state = 5
                self.laststate = 5
            else:
                self.spm_f_eval(PIC_COUNT=PIC_COUNT, now=now, iw_shape=iw_shape, relearning=relearning) #第2段階用の画像を撮影

    def spm_f_eval(self, PIC_COUNT=1, now="TEST", iw_shape=(2,3),feature_names=None, relearning:dict=dict(relearn_state=False,f1=ct.const.f1,f3=ct.const.f3)):#第一段階学習&評価。npzファイル作成が目的
        if relearning['relearn_state']:
            try:
                second_img_paths = sorted(glob(f"results/{self.startTime}/camera_result/first_spm/learn{self.learncount-1}/evaluate/evaluateimg*.jpg"))[-relearning['f3']+1:-relearning['f1']]
            except IndexError:
                # ここで学習枚数足りなかったら動作指定（あきらめて１回目と同じ動きするのか、再学習をあきらめるか）
                print('There are not enough number of pics for ReLearning.')
                # relearning['relearn_state'] = False  # 再学習用に画像を1枚
        
        if not relearning['relearn_state']:
        # self.cap = cv2.VideoCapture(0)
            for i in range(PIC_COUNT):
                try:
                    ret,self.secondimg = self.cap.read()
                    print("done:",i)
                except:
                    pass
                if self.state == 4:
                    save_file = f"results/{self.startTime}/camera_result/first_spm/learn{self.learncount}/evaluate/evaluateimg{time.time():.2f}.jpg"
                elif self.state == 6:
                    save_file = f"results/{self.startTime}/camera_result/planning/learn{self.learncount}/planning_pics/planningimg{time.time():.2f}.jpg"

                cv2.imwrite(save_file,self.secondimg)
                self.firstevalimgcount += 1
                
                if self.state == 4:
                    # self.MotorR.go(74)#走行
                    # self.MotorL.go(70)#走行
                    # # self.stuck_detection()
                    # time.sleep(0.1)
                    # self.MotorR.stop()
                    # self.MotorL.stop()
                    # if i%10 == 0: #10枚撮影する毎にセンサの値取得
                    #     self.camerastate = "captured!"
                    #     self.sensor()
                    #     self.camerastate = 0
                    # state4の学習時にもBNOベースで走行
                    self.sensor()
                    self.planning(np.array([0,0,0,0,0,0]))
                    self.stuck_detection()#ここは注意
#                     print(f"{fmg_list.index(fmg)} fmg evaluated")
                
            if not PIC_COUNT == 1:
                second_img_paths = sorted(glob(f"results/{self.startTime}/camera_result/first_spm/learn{self.learncount}/evaluate/evaluateimg*.jpg"))
            else:
                second_img_paths = [save_file]
        
        for importPath in second_img_paths:
            self.GREEN_LED.led_on()
            self.RED_LED.led_on()
            self.BLUE_LED.led_off()
        
            feature_values = {}

            default_names = ["normalRGB","enphasis","edge","hsv","red","blue","green","purple","emerald","yellow"]
            for keys in default_names:
                feature_values[keys] = {}
            
            self.tempDir = TemporaryDirectory()
            tempDir_name = self.tempDir.name
            
            iw = IntoWindow(importPath, tempDir_name, False) #画像の特徴抽出のインスタンス生成
            self.img=cv2.imread(importPath, 1)
            self.img = self.img[int(0.25*self.img.shape[0]):int(0.75*self.img.shape[0])]
            cv2.imwrite(importPath, self.img)
            
            #if self.state == 4: #ステートが4の場合はセンサの値取得
                #self.sensor()
            if feature_names == None: #第一段階学習モード
                self.camerastate = "captured!"
                fmg_list = iw.feature_img(frame_num=now,feature_names=feature_names) #特徴抽出。リストに特徴画像が入る
                
                for fmg in fmg_list:#それぞれの特徴画像に対して処理
                    iw_list, window_size = iw.breakout(cv2.imread(fmg,cv2.IMREAD_GRAYSCALE)) #ブレイクアウト
                    feature_name = str(re.findall(tempDir_name + f"/(.*)_.*_", fmg)[0])
                    
                    # print("FEATURED BY: ",feature_name)
                    
                    D, ksvd = self.dict_list[feature_name]
                    for win in range(int(np.prod(iw_shape))): #それぞれのウィンドウに対して評価を実施

                        # win_1~3は特徴量算出を行わない
                        if win not in [0,1,2]:
                            ei = EvaluateImg(iw_list[win])
                            img_rec = ei.reconstruct(D, ksvd, window_size)
                            saveName = self.saveDir + f"/{self.startTime}/camera_result/first_spm/learn{self.learncount}/processed/difference"
                            if not os.path.exists(saveName):
                                os.mkdir(saveName)
                            saveName = self.saveDir + f"/{self.startTime}/camera_result/first_spm/learn{self.learncount}/processed/difference/{now}"
                            if not os.path.exists(saveName):
                                os.mkdir(saveName)
                            ave, med, var, mode, kurt, skew = ei.evaluate(iw_list[win], img_rec, win+1, feature_name, now, self.saveDir)
                        else :
                            ave, med, var, mode, kurt, skew = 0, 0, 0, 0, 0, 0

                        feature_values[feature_name][f'win_{win+1}'] = {}
                        feature_values[feature_name][f'win_{win+1}']["var"] = ave  # 平均値
                        feature_values[feature_name][f'win_{win+1}']["med"] = med  # 中央値
                        feature_values[feature_name][f'win_{win+1}']["ave"] = var  # 分散値
                        feature_values[feature_name][f'win_{win+1}']["mode"] = mode  # 最頻値
                        feature_values[feature_name][f'win_{win+1}']["kurt"] = kurt  # 尖度
                        feature_values[feature_name][f'win_{win+1}']["skew"] = skew  # 歪度
                
                self.camerastate = 0
                       
            else: #第一段階評価モード。runningで使うための部 # ここ変える
                feature_list = ["normalRGB","enphasis","edge","vari","rgbvi","grvi","ior","hsv","red","blue","green","purple","emerald","yellow"]
                features = []
                
                for feature in feature_names:# windowgoto
                    features = features + feature
#                     print("feature_names:",feature_names)
                features = set(features)
                features = list(features) #windownosaishouchi
                print("features:",features)
                fmg_list = iw.feature_img(frame_num=now,feature_names=features) # 特徴抽出。リストに特徴画像が入る

                for fmg in fmg_list: #それぞれの特徴画像に対して処理
                    iw_list, window_size = iw.breakout(cv2.imread(fmg,cv2.IMREAD_GRAYSCALE)) # ブレイクアウトにより画像を6分割
                    feature_name = str(re.findall(tempDir_name + f"/(.*)_.*_", fmg)[0]) # 特徴処理のみ抽出
                    # print("FEATURED BY: ",feature_name)
                    for win in range(int(np.prod(iw_shape))): #それぞれのウィンドウに対して評価を実施
                        if feature_name in feature_names[win] and win not in [0,1,2]: #ウィンドウに含まれいていた場合
                            D, ksvd = self.dict_list[feature_name]
                            ei = EvaluateImg(iw_list[win])
                            img_rec = ei.reconstruct(D, ksvd, window_size)
                            saveName = self.saveDir + f"/{self.startTime}/camera_result/first_spm/learn{self.learncount}/processed/difference"
    #                         if not os.path.exists(saveName):
    #                             os.mkdir(saveName)
    #                         saveName = self.saveDir + f"/camera_result/first_spm/learn{self.learncount}/processed/difference/{now}"
    #                         if not os.path.exists(saveName):
    #                             os.mkdir(saveName)
                            ave, med, var, mode, kurt, skew = ei.evaluate(iw_list[win], img_rec, win+1, feature_name, now, self.saveDir)

                        else:
                            ave, med, var, mode, kurt, skew = 0, 0, 0, 0, 0, 0


                        feature_values[feature_name][f'win_{win+1}'] = {}
                        feature_values[feature_name][f'win_{win+1}']["var"] = ave  # 平均値
                        feature_values[feature_name][f'win_{win+1}']["med"] = med  # 中央値
                        feature_values[feature_name][f'win_{win+1}']["ave"] = var  # 分散値
                        feature_values[feature_name][f'win_{win+1}']["mode"] = mode  # 最頻値
                        feature_values[feature_name][f'win_{win+1}']["kurt"] = kurt  # 尖度
                        feature_values[feature_name][f'win_{win+1}']["skew"] = skew  # 歪度
                                
                    for feature_name in feature_list:
                        if feature_name not in features:
                            for win in range(int(np.prod(iw_shape))): #それぞれのウィンドウに対して評価を実施
                                feature_values[feature_name][f'win_{win+1}'] = {}
                                feature_values[feature_name][f'win_{win+1}']["var"] = 0  # 平均値
                                feature_values[feature_name][f'win_{win+1}']["med"] = 0  # 中央値
                                feature_values[feature_name][f'win_{win+1}']["ave"] = 0  # 分散値
                                feature_values[feature_name][f'win_{win+1}']["mode"] = 0  # 最頻値
                                feature_values[feature_name][f'win_{win+1}']["kurt"] = 0  # 尖度
                                feature_values[feature_name][f'win_{win+1}']["skew"] = 0  # 歪度

                    if fmg != fmg_list[-1] and type(self.risk) == np.ndarray:
                        self.sensor()
                        self.planning(self.risk)
                        self.stuck_detection()#ここは注意
#                     print(f"{fmg_list.index(fmg)} fmg evaluated")
                    
            self.BLUE_LED.led_on()
            # npzファイル形式で計算結果保存
            if self.state == 4:
                # self.savenpz_dir = "/home/pi/Desktop/wolvez2022/pre_data/"
                self.savenpz_dir = self.saveDir + f"/{self.startTime}/camera_result/second_spm/learn{self.learncount}/"
            elif self.state == 6:
                self.savenpz_dir = self.saveDir + f"/{self.startTime}/camera_result/planning/learn{self.learncount}/planning_npz/"
            
            # 保存時のファイル名指定（現在は時間）
            now=str(datetime.now())[:21].replace(" ","_").replace(":","-")
#             print("feature_values:",feature_values)
            # print("shape:",len(feature_values))
            np.savez_compressed(self.savenpz_dir + now,array_1=np.array([feature_values])) #npzファイル作成
            self.tempDir.cleanup()
        self.GREEN_LED.led_on()
        self.RED_LED.led_on()
        self.BLUE_LED.led_off()