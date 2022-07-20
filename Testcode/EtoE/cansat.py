#Last Update 2022/07/02
#Author : Toshiki Fukui

from tempfile import TemporaryDirectory
import RPi.GPIO as GPIO
import sys
import cv2
import time
import numpy as np
import os
import re
from datetime import datetime
from glob import glob
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

import math
from math import sqrt
from math import radians
from math import sin
from math import fabs
from turtle import distance

class Cansat():
    def __init__(self,state):
        # GPIO設定
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM) #GPIOの設定
        GPIO.setup(ct.const.FLIGHTPIN_PIN,GPIO.IN,pull_up_down=GPIO.PUD_UP) #フライトピン用。プルアップを有効化
        GPIO.setup(ct.const.SEPARATION_PIN,GPIO.OUT) #焼き切り用のピンの設定
        
        # インスタンス生成        
        self.bno055 = BNO055()
        self.MotorR = motor(ct.const.RIGHT_MOTOR_IN1_PIN,ct.const.RIGHT_MOTOR_IN2_PIN,ct.const.RIGHT_MOTOR_VREF_PIN)
        self.MotorL = motor(ct.const.LEFT_MOTOR_IN1_PIN,ct.const.LEFT_MOTOR_IN2_PIN, ct.const.LEFT_MOTOR_VREF_PIN)
        self.gps = GPS()
        self.lora = lora()
        self.RED_LED = led(ct.const.RED_LED_PIN)
        self.BLUE_LED = led(ct.const.BLUE_LED_PIN)
        self.GREEN_LED = led(ct.const.GREEN_LED_PIN)
        
        #初期パラメータ設定
        self.timer = 0
        self.state = state
        self.startTime = time.time()
        self.preparingTime = 0
        self.flyingTime = 0
        self.droppingTime = 0
        self.landingTime = 0
        self.runningTime = 0
        self.landstate = 0
        self.firstlearnimgcount = 0
        self.firstevalimgcount = 0
        self.camerastate = 0
        self.camerafirst = 0
        self.stuckTime = 0
        self.learncount = 1
        self.learn_state = True
        # self.pre_motorTime = 0
        # self.startingTime = 0
        # self.measureringTime = 0
        # self.runningTime = 0
        # self.positioningTime = 0
        # self.finishTime = 0
        
        #state管理用変数初期化
        self.gpscount=0
        self.startgps_lon=[]
        self.startgps_lat=[]
        
        self.countPreLoop = 0
        self.countFlyLoop = 0
        self.countDropLoop = 0
        self.countstuckLoop = 0

        self.dict_list = {}
        self.saveDir = "results"
        self.mkdir()
        
        self.goallon = 139.65578
        self.goallat = 35.55518

    def mkdir(self):
        #フォルダ作成部分
        folder_paths =[f"results/camera_result/first_spm",
                       f"results/camera_result/first_spm/learn{self.learncount}",
                       f"results/camera_result/first_spm/learn{self.learncount}/evaluate",
                       f"results/camera_result/first_spm/learn{self.learncount}/processed",
                       f"results/camera_result/second_spm",
                       f"results/camera_result/second_spm/learn{self.learncount}",
                       f"results/camera_result/planning",
                       f"results/camera_result/planning/learn{self.learncount}",
                       f"results/camera_result/planning/learn{self.learncount}/planning_npz",
                       f"results/camera_result/planning/learn{self.learncount}/planning_pics"]

        for folder_path in folder_paths:
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
    
    def writeData(self):
        #ログデータ作成。\マークを入れることで改行してもコードを続けて書くことができる
        print_datalog = str(self.timer) + ","\
                  + "state:"+str(self.state)+ ","\
                  + "Time:"+str(self.gps.Time) + ","\
                  + "Lat:"+str(self.gps.Lat).rjust(6) + ","\
                  + "Lng:"+str(self.gps.Lon).rjust(6) + ","\
                  + "ax:"+str(round(self.ax,6)).rjust(6) + ","\
                  + "ay:"+str(round(self.ay,6)).rjust(6) + ","\
                  + "az:"+str(round(self.az,6)).rjust(6) + ","\
                  + "q:" + str(self.ex).rjust(6) + ","\
                  + "rV:" + str(round(self.MotorR.velocity,2)).rjust(6) + ","\
                  + "lV:" + str(round(self.MotorL.velocity,2)).rjust(6) + ","\
                  + "Camera:" + str(self.camerastate)

        print(print_datalog)
        
        datalog = str(self.timer) + ","\
                  + "state:"+str(self.state) + ","\
                  + "Time:"+str(self.gps.Time) + ","\
                  + "Lat:"+str(self.gps.Lat).rjust(6) + ","\
                  + "Lng:"+str(self.gps.Lon).rjust(6) + ","\
                  + "ax:"+str(self.bno055.ax).rjust(6) + ","\
                  + "ay:"+str(self.bno055.ay).rjust(6) + ","\
                  + "az:"+str(self.bno055.az).rjust(6) + ","\
                  + "q:"+str(self.bno055.ex).rjust(6) + ","\
                  + "rV:"+str(round(self.MotorR.velocity,3)).rjust(6) + ","\
                  + "lV:"+str(round(self.MotorL.velocity,3)).rjust(6) + ","\
                  + "Camera:" + str(self.camerastate)
        
        with open('results/control_result.txt',"a")  as test: # [mode] x:ファイルの新規作成、r:ファイルの読み込み、w:ファイルへの書き込み、a:ファイルへの追記
            test.write(datalog + '\n')

    def sequence(self):
        if self.state == 0:#センサ系の準備を行う段階
            self.preparing()
        elif self.state == 1:#放出・降下を行う段階
            self.flying()
        elif self.state == 2:#着陸判定、パラ分離（焼き切り）
            self.dropping()
        elif self.state == 3:#パラシュートから離れる。カメラでの撮影行う
            self.landing()
        elif self.state == 4:#スパースモデリング第一段階
            self.spm_first(ct.const.SPMFIRST_PIC_COUNT)
        elif self.state == 5:#スパースモデリング第二段階
            self.model_master,self.scaler_master,self.feature_names = self.spm_second()
        elif self.state == 6:#経路計画段階
            self.running(self.model_master,self.scaler_master,self.feature_names)
        # elif self.state == 7:
        #     self.re_learning()
        # elif self.state == 8:#終了
        #     self.finish()
        else:
            self.state = self.laststate #どこにも引っかからない場合何かがおかしいのでlaststateに戻してあげる

    def setup(self):
        self.gps.setupGps()
        # os.system("sudo insmod LoRa_SOFT/soft_uart.ko")
        self.bno055.setupBno()
        self.lora.sendDevice.setup_lora()
        if self.bno055.begin() is not True:
            print("Error initializing device")
            exit()    

    def sensor(self):#セットアップ終了後
        self.timer = int(1000*(time.time() - self.startTime)) #経過時間 (ms)
        self.gps.gpsread()
        self.bno055.bnoread()
        self.ax=round(self.bno055.ax,3)
        self.ay=round(self.bno055.ay,3)
        self.az=round(self.bno055.az,3)
        self.ex=round(self.bno055.ex,3)
        
        self.writeData()#txtファイルへのログの保存
    
        if not self.state == 1: #preparingのときは電波を発しない
#             if not self.state ==5:#self.sendRadio()#LoRaでログを送信
            self.sendLoRa()
#             else:
#                 self.switchRadio()

    def preparing(self):#時間が立ったら移行
        if self.preparingTime == 0:
            self.preparingTime = time.time()#時刻を取得
            self.RED_LED.led_on()
            self.BLUE_LED.led_off()
            self.GREEN_LED.led_off()

        #self.countPreLoop+ = 1
        if not self.preparingTime == 0:
            if self.gpscount <= ct.const.PREPARING_GPS_COUNT_THRE:
                self.startgps_lon.append(float(self.gps.Lon))
                self.startgps_lat.append(float(self.gps.Lat))
                self.gpscount+=1
                
            else:
                print("GPS completed!!")
            
            if time.time() - self.preparingTime > ct.const.PREPARING_TIME_THRE:
                self.startlon=np.mean(self.startgps_lon)
                self.startlat=np.mean(self.startgps_lat)
                self.state = 1
                self.laststate = 1
    
    def flying(self):#フライトピンが外れる➡︎ボイド缶から放出されたことを検出するステート
        if self.flyingTime == 0:#時刻を取得してLEDをステートに合わせて光らせる
            self.flyingTime = time.time()
            self.RED_LED.led_off()
            self.BLUE_LED.led_off()
            self.GREEN_LED.led_off()

        if GPIO.input(ct.const.FLIGHTPIN_PIN) == GPIO.HIGH:#highかどうか＝フライトピンが外れているかチェック
            self.countFlyLoop+=1
            if self.countFlyLoop > ct.const.FLYING_FLIGHTPIN_COUNT_THRE:#一定時間HIGHだったらステート移行
                self.state = 2
                self.laststate = 2       
        else:
            self.countFlyLoop = 0 #何故かLOWだったときカウントをリセット
    
    def dropping(self): #着地判定ステート
        if self.droppingTime == 0:#時刻を取得してLEDをステートに合わせて光らせる
            self.droppingTime = time.time()
            self.RED_LED.led_off()
            self.BLUE_LED.led_on()
            self.GREEN_LED.led_off()
      
        #加速度が小さくなったら着地判定
        if (self.bno055.ax**2 + self.bno055.ay**2 + self.bno055.az**2) < ct.const.DROPPING_ACC_THRE**2:#加速度が閾値以下で着地判定
            self.countDropLoop+=1            
            if self.countDropLoop > ct.const.DROPPING_ACC_COUNT_THRE:#着地判定が複数回行われたらステート以降
                self.state = 3
                self.laststate = 3
        else:
            self.countDropLoop = 0 #初期化の必要あり

    def landing(self):
        if self.landingTime == 0:#時刻を取得してLEDをステートに合わせて光らせる
            self.landingTime = time.time()
            self.RED_LED.led_off()
            self.BLUE_LED.led_off()
            self.GREEN_LED.led_on()
            
        if not self.landingTime == 0:
            #焼き切りによるパラ分離
            if self.landstate == 0:
                GPIO.output(ct.const.SEPARATION_PIN,1) #電圧をHIGHにして焼き切りを行う
                if time.time()-self.landingTime > ct.const.SEPARATION_TIME_THRE:
                    GPIO.output(ct.const.SEPARATION_PIN,0) #焼き切りが危ないのでlowにしておく
                    self.landstate = 1
                    self.pre_motorTime = time.time()
            
            #分離シート離脱
            elif self.landstate == 1:
                self.MotorR.go(ct.const.LANDING_MOTOR_VREF)
                self.MotorL.go(ct.const.LANDING_MOTOR_VREF)

                self.stuck_detection()

                if time.time()-self.pre_motorTime > ct.const.LANDING_MOTOR_TIME_THRE: #5秒間モータ回して分離シートから十分離れる
                    self.MotorR.stop()
                    self.MotorL.stop()
                    self.state = 4
                    self.laststate = 4

    def spm_first(self, PIC_COUNT):
        start_time = time.time()#学習用時間計測。学習開始時間
        
        #保存時のファイル名指定（現在は時間）
        now=str(datetime.now())[:19].replace(" ","_").replace(":","-")

        Save = True
        
        # Path that img will be read
        #importPath = path.replace("\\", "/")
        
        # This will change such as datetime
        # print("CURRENT FRAME: "+str(re.findall(".*/frame_(.*).jpg", importPath)[0]))
        
        iw_shape = (2, 3)#ウィンドウのシェイプ
        D, ksvd = None, None #最初に指定しないと怒られちゃうから
        feature_values = {}

        if self.learn_state:
            print("=====LEARNING PHASE=====")
        else:
            print("=====EVALUATING PHASE=====")
            
        
        if self.learn_state:#学習モデル獲得
            
            #学習用画像を一枚撮影
            if self.camerafirst == 0:
                self.cap = cv2.VideoCapture(0)
                ret, firstimg = self.cap.read()
                cv2.imwrite(f"results/camera_result/first_spm/learn{self.learncount}/firstimg{self.firstlearnimgcount}.jpg",firstimg)
                self.camerastate = "captured!"
                self.firstlearnimgcount += 1
                self.camerafirst = 1
            else:
                self.camerastate = 0
            
            importPath = f"results/camera_result/first_spm/learn{self.learncount}/firstimg{self.firstlearnimgcount-1}.jpg"
            processed_Dir = f"results/camera_result/first_spm/learn{self.learncount}/processed"
            iw = IntoWindow(importPath, processed_Dir, Save) #画像の特徴抽出のインスタンス生成
            # processing img
            fmg_list = iw.feature_img(frame_num=now) #特徴抽出。リストに特徴画像が入る 
            for fmg in fmg_list:#それぞれの特徴画像に対して処理
                # breakout by windows
                iw_list, window_size = iw.breakout(iw.read_img(fmg)) #ブレイクアウト
                feature_name = str(re.findall(self.saveDir + f"/camera_result/first_spm/learn{self.learncount}/processed/(.*)_.*_", fmg)[0])
                print("FEATURED BY: ",feature_name)

                for win in range(int(np.prod(iw_shape))): #それぞれのウィンドウに対して学習を実施
                    if win+1 == int((iw_shape[0]-1)*iw_shape[1]) + int(iw_shape[1]/2) + 1:
                        ld = LearnDict(iw_list[win])
                        D, ksvd = ld.generate() #辞書獲得
                        self.dict_list[feature_name] = [D, ksvd]
                        save_name = self.saveDir + f"/learn{self.learncount}/learnimg/{feature_name}_part_{win+1}_{now}.jpg"
                        # cv2.imwrite(save_name, iw_list[win])
            self.learn_state = False

        else:#20枚撮影
            self.spm_f_eval(PIC_COUNT=PIC_COUNT, now=now, iw_shape=iw_shape) #第2段階用の画像を撮影
            if self.state == 4:
                self.state = 5
                self.laststate = 5
                    
        
        end_time = time.time()#計算終了
        print("Calc Time:",end_time-start_time)

    def spm_f_eval(self, PIC_COUNT=1, now="TEST", iw_shape=(2,3),feature_names = None):
#         self.cap = cv2.VideoCapture(0)
        for i in range(PIC_COUNT):
            print(i,"枚目")
            ret,self.secondimg = self.cap.read()
            if self.state == 4:
                save_file = f"results/camera_result/first_spm/learn{self.learncount}/evaluate/evaluateimg{i}.jpg"
            elif self.state == 6:
                save_file = f"results/camera_result/planning/learn{self.learncount}/planning_pics/planningimg{i}.jpg"

            cv2.imwrite(save_file,self.secondimg)
            self.firstevalimgcount += 1
            
            if self.state == 4:
                self.MotorR.go(70)#走行
                self.MotorL.go(50)#走行
                time.sleep(0.4)
                self.MotorR.stop()
                self.MotorL.stop()
            
        if not PIC_COUNT == 1:
            second_img_paths = sorted(glob(f"results/camera_result/first_spm/learn{self.learncount}/evaluate/evaluateimg*.jpg"))
        else:
            second_img_paths = [save_file]
        
        for importPath in second_img_paths:
        
            feature_values = {}
            
            self.tempDir = TemporaryDirectory()
            tempDir_name = self.tempDir.name
            
            iw = IntoWindow(importPath, tempDir_name, False) #画像の特徴抽出のインスタンス生成
            # processing img
            
            if feature_names == None:
                fmg_list = iw.feature_img(frame_num=now,feature_names=feature_names) #特徴抽出。リストに特徴画像が入る

                for fmg in fmg_list:#それぞれの特徴画像に対して処理
                    iw_list, window_size = iw.breakout(iw.read_img(fmg)) #ブレイクアウト
                    feature_name = str(re.findall(tempDir_name + f"/(.*)_.*_", fmg)[0])
                    
                    print("FEATURED BY: ",feature_name)
                    
                    for win in range(int(np.prod(iw_shape))): #それぞれのウィンドウに対して評価を実施
                        D, ksvd = self.dict_list[feature_name]

                        ei = EvaluateImg(iw_list[win])
                        img_rec = ei.reconstruct(D, ksvd, window_size)
                        saveName = self.saveDir + f"/camera_result/first_spm/learn{self.learncount}/processed/difference"
                        if not os.path.exists(saveName):
                            os.mkdir(saveName)
                        saveName = self.saveDir + f"/camera_result/first_spm/learn{self.learncount}/processed/difference/{now}"
                        if not os.path.exists(saveName):
                            os.mkdir(saveName)
                        ave, med, var, mode, kurt, skew = ei.evaluate(iw_list[win], img_rec, win+1, feature_name, now, self.saveDir)
                        
                        # 特徴量終結/1枚
                        if win == 0:
                            feature_values[feature_name] = {}

                        feature_values[feature_name][f'win_{win+1}'] = {}
                        feature_values[feature_name][f'win_{win+1}']["var"] = ave  # 平均値
                        feature_values[feature_name][f'win_{win+1}']["med"] = med  # 中央値
                        feature_values[feature_name][f'win_{win+1}']["ave"] = var  # 分散値
                        feature_values[feature_name][f'win_{win+1}']["mode"] = mode  # 最頻値
                        feature_values[feature_name][f'win_{win+1}']["kurt"] = kurt  # 尖度
                        feature_values[feature_name][f'win_{win+1}']["skew"] = skew  # 歪度
                
            else:
                i=0
                for win,feature in enumerate(feature_names):#tokuchougazounoyouso
                    fmg_list = iw.feature_img(frame_num=now,feature_names=feature) #特徴抽出。リストに特徴画像が入る
#                     print(f"win{win},feature{feature}")
#                     print("feature_names:",feature_names)
#                     print("fmg_list",fmg_list)
                    for fmg in fmg_list:#それぞれの特徴画像に対して処理
                        if fmg !=0:
                            iw_list, window_size = iw.breakout(iw.read_img(fmg)) #ブレイクアウト #hitotsunoshoriwojikkou
                            feature_name = str(re.findall(tempDir_name + f"/(.*)_.*_", fmg)[0])
                            print("FEATURED BY: ",feature_name)
                            
                            D, ksvd = self.dict_list[feature_name]
                            ei = EvaluateImg(iw_list[win])
                            img_rec = ei.reconstruct(D, ksvd, window_size)
                            saveName = self.saveDir + f"/camera_result/first_spm/learn{self.learncount}/processed/difference"
                            if not os.path.exists(saveName):
                                os.mkdir(saveName)
                            saveName = self.saveDir + f"/camera_result/first_spm/learn{self.learncount}/processed/difference/{now}"
                            if not os.path.exists(saveName):
                                os.mkdir(saveName)
                            ave, med, var, mode, kurt, skew = ei.evaluate(iw_list[win], img_rec, win+1, feature_name, now, self.saveDir)
                        
                        else:
                            feature_name = str(i)
                            ave, med, var, mode, kurt, skew = [0,0,0,0,0,0]
                            i+=1
                        
                        # 特徴量終結/1枚
                        if feature_name not in feature_values:
                            feature_values[feature_name] = {}

                        feature_values[feature_name][f'win_{win+1}'] = {}
                        feature_values[feature_name][f'win_{win+1}']["var"] = ave  # 平均値
                        feature_values[feature_name][f'win_{win+1}']["med"] = med  # 中央値
                        feature_values[feature_name][f'win_{win+1}']["ave"] = var  # 分散値
                        feature_values[feature_name][f'win_{win+1}']["mode"] = mode  # 最頻値
                        feature_values[feature_name][f'win_{win+1}']["kurt"] = kurt  # 尖度
                        feature_values[feature_name][f'win_{win+1}']["skew"] = skew  # 歪度
            
            #npzファイル形式で計算結果保存
            if self.state == 4:
                self.savenpz_dir = self.saveDir + f"/camera_result/second_spm/learn{self.learncount}/"
            elif self.state == 6:
                self.savenpz_dir = self.saveDir + f"/camera_result/planning/learn{self.learncount}/planning_npz/"
            
            #保存時のファイル名指定（現在は時間）
            now=str(datetime.now())[:19].replace(" ","_").replace(":","-")
#             print("feature_values:",feature_values)
#             print("shape:",len(feature_values))
            np.savez_compressed(self.savenpz_dir + now,array_1=np.array([feature_values]))
            self.tempDir.cleanup()
    
    def spm_second(self):
        npz_dir = f"results/camera_result/second_spm/learn{self.learncount}/*"
        # wolvez2022/spmで実行してください
        train_npz = sorted(glob(npz_dir))
        spm2_prepare = SPM2Open_npz()
        data_list_all_win,label_list_all_win = spm2_prepare.unpack(train_npz)
        spm2_learn = SPM2Learn()

        #ウィンドウによってスタックと教示する時間帯を変えず、一括とする場合
        stack_start = ct.const.STUCK_START
        stack_end = ct.const.STUCK_END

        #ウィンドウによってスタックすると教示する時間帯を変える場合はnp.arrayを定義
        stack_info = None
        """
            stack_info=np.array([[12., 18.],
                [12., 18.],
                [12., 18.],
                [12., 18.],
                [12., 18.],
                [12, 18.]])
            「stackした」と学習させるフレームの指定方法
            1. 全ウィンドウで一斉にラベリングする場合
                Learnの引数でstack_appearおよびstack_disappearを[s]で指定する。
            2. ウィンドウごとに個別にラベリングする場合
            stack_info=np.array(
                [
                    [win_1_stack_start,win_1_stack_end],
                    [win_2_stack_start,win_2_stack_end],
                    ...
                    [win_6_stack_start,win_6_stack_end],
                ]
            )
            t[s]で入力すること。
        """
        spm2_learn.start(data_list_all_win,label_list_all_win,fps=30,alpha=5.0,stack_appear=stack_start,stack_disappear=stack_end,stack_info=stack_info)#どっちかは外すのがいいのか
        model_master,label_list_all_win,scaler_master=spm2_learn.get_data()
        nonzero_w, nonzero_w_label, nonzero_w_num = spm2_learn.get_nonzero_w()
        print("feature_names",np.array(nonzero_w_label,dtype=object).shape)
        feature_names = nonzero_w_label
        

        
        """
            model_master: 各ウィンドウを学習したモデル（俗にいう"model.predict()"とかの"model.predict()"とかのmodelに相当するのがリストで入ってる）
            label_list_all_win: 重み行列の各成分を、その意味（ex.window_1のrgb画像のaverage）の説明で書き換えた配列
            scaler_master: 各ウィンドウを標準化した時のモデル（scaler.transform()の"scaler"に相当するのがリストで入って）
            feature_names: 特徴処理の名前をリストに格納
        """
        
        self.state = 6
        self.laststate = 6
        return model_master,scaler_master,feature_names

    def running(self,model_master,scaler_master,feature_names):
        planning_dir = f"results/camera_result/planning/learn{self.learncount}/planning_npz/*"
        planning_npz = sorted(glob(planning_dir))
        
        #保存時のファイル名指定（現在は時間）
        now=str(datetime.now())[:19].replace(" ","_").replace(":","-")
        
        self.spm_f_eval(now = now)#特徴的な処理を行ってnpzを作成
        if self.runningTime == 0:
            self.runningTime = time.time()
        else:
            SPM2_predict_prepare = SPM2Open_npz()#第一段階で作成したnpzを取得
            test_data_list_all_win,test_label_list_all_win = SPM2_predict_prepare.unpack([planning_npz[-1]])
            spm2_predict = SPM2Evaluate()
            spm2_predict.start(model_master,test_data_list_all_win,test_label_list_all_win,scaler_master)
            risk = np.array(spm2_predict.get_score()).reshape(2,3)#win1~win6の危険度マップができる
            print(risk)
#             self.state = 8
#             self.laststate =8
    #         # 走行
            self.planning(risk)
            # self.stuck_detection()#ここは注意


    def planning(self,risk):
#         self.gps.Lon = 140
#         self.gps.Lat = 36
        
#         self.goallon = 139.65578
#         self.goallat = 35.55518
        

        self.gps.vincenty_inverse(self.goallat,self.goallon,self.gps.Lat,self.gps.Lon) #距離:self.gps.gpsdis 方位角:self.gps.gpsdegrees
        self.x = self.gps.gpsdis*math.cos(math.radians(self.gps.gpsdegrees))
        self.y = self.gps.gpsdis*math.sin(math.radians(self.gps.gpsdegrees))
        theta_goal = self.gps.gpsdegrees
        phi = theta_goal-self.bno055.ex
        
        if phi < -180:
            phi += 360
        elif phi > 180:
            phi -= 360

        dir_run = self.calc_dir(risk,phi)
        if dir_run == 0:
            self.MotorR.go(80)
            self.MotorL.go(60)
        elif dir_run == 1:
            self.MotorR.go(70)
            self.MotorL.go(70)
        elif dir_run == 2:
            self.MotorR.go(60)
            self.MotorL.go(80)
        elif dir_run == 3:
            self.MotorR.stop()
            self.MotorL.stop()


    def decide_direction(self,phi):
        if phi >= 20:
            direction_goal = 2
            print("ゴール方向："+str(direction_goal)+" -> 右に曲がりたい")
        elif phi > -20 and phi < 20:
            direction_goal = 1
            print("ゴール方向："+str(direction_goal)+" -> 直進したい")
        else:
            direction_goal = 0
            print("ゴール方向："+str(direction_goal)+" -> 左に曲がりたい")
        return direction_goal


    def calc_dir(self,risk,phi):
        # 危険度の閾値を決定
        threshold_risk = 70
        lower_risk = risk[1,:]
        direction_goal = self.decide_direction(phi)
        
        if np.amin(lower_risk) >= threshold_risk:
            print("前方に安全なルートはありません。90度回転して新たな経路を探索します。")
            direction_real = 3
        else:
            if lower_risk[direction_goal] <= threshold_risk:   #ゴール方向の危険度が閾値以下の場合
                print("go for goal")
                direction_real = direction_goal
            else:
                print("ゴール方向が安全ではありません。別ルートを探索します。")
                if direction_goal == 0:
                    if lower_risk[1] <= lower_risk[2]:
                        direction_real = 1
                    else:
                        direction_real = 2
                elif direction_goal == 1:
                    if lower_risk[0] <= lower_risk[2]:
                        direction_real = 0
                    else:
                        direction_real = 2
                elif direction_goal == 2:
                    if lower_risk[0] <= lower_risk[1]:
                        direction_real = 0
                    else:
                        direction_real = 1
                        
        return direction_real

    
            
    def sendLoRa(self):
        datalog = str(self.state) + ","\
                  + str(self.gps.Time) + ","\
                  + str(self.gps.Lat) + ","\
                  + str(self.gps.Lon)

        self.lora.sendData(datalog) #データを送信
        
    def stuck_detection(self):        
        if (self.bno055.ax**2+self.bno055.ay**2+self.bno055.az**2) <= ct.const.STUCK_ACC_THRE**2:
            self.countstuckLoop+= 1
            if self.countstuckLoop > ct.const.STUCK_COUNT_THRE: #加速度が閾値以下になるケースがある程度続いたらスタックと判定
                if self.stuckTime == 0:
                    self.stuckTime = time.time()#スタック検知最初の時間計測
                #トルネード実施
                self.MotorR.go(ct.const.STUCK_MOTOR_VREF)
                self.MotorL.back(ct.const.STUCK_MOTOR_VREF)

                if time.time() - self.stuckTime > ct.const.STUCK_MOTOR_TIME_THRE:#閾値以上の時間モータを回転させたら
                    self.MotorR.stop()
                    self.MotorL.stop()
                    self.stuckTime = 0
                    self.countstuckLoop = 0

    def keyboardinterrupt(self):
        self.MotorR.stop()
        self.MotorL.stop()
        self.RED_LED.led_off()
        self.BLUE_LED.led_off()
        self.GREEN_LED.led_off()
#         self.cap.release()
#         cv2.destroyAllWindows()