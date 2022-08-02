import keyboard
import constant as ct
import cv2
import time
import tkinter as tk
import traceback
# import RPi.GPIO as GPIO
# from motor import motor

class Experiment():
    def __init__(self):
        self.state = 0
        self.start = 0.0
        self.end = 0.0
        self.root = tk.Tk()
        self.ela_time = self.end-self.start
        self.root.geometry('300x200')
        self.root.title("制限時間")
        self.root.configure(bg = "red")
        self.canvas=tk.Canvas(self.root,width=300,heigh=200)
        self.canvas.pack()#ここを書かないとcanvasがうまく入らない．

    def create_canvas(self):
        time =180.0-self.ela_time
        m = str(int(time/60)).rjust(2,"0")
        s = str(int(time%60)).rjust(2,"0")
        text = f"{m}:{s}"
        self.canvas.create_text(300/2,200/2,text=text,font=("Times New Roman",100),tag='Y') #タグを入れることで更新できるようにする．
        self.canvas.update()
        self.canvas.delete('Y')
    
    def key(self):
        keyboard.on_press_key("up", lambda _: self.straight())
        keyboard.on_press_key("down", lambda _: self.back())
        keyboard.on_press_key("right", lambda _: self.right())
        keyboard.on_press_key("left", lambda _: self.left())
        keyboard.on_press_key("c", lambda _: self.cam_start())
    
    def straight(self):
        print("Go straight")
        # MotorR.go(70)
        # MotorL.go(70)
    
    def back(self):
        print("Go backward")
        # MotorR.back(70)
        # MotorL.back(70)
    
    def right(self):
        print("Turn right")
        # MotorR.back(70)
        # MotorL.go(70)

    def left(self):
        print("Turn left")
        # MotorR.go(70)
        # MotorL.back(70)
    
    def cam_start(self):
        if self.state == 0:
            self.cap = cv2.VideoCapture(0)
            
            self.start = time.time()
            self.state = 1

    def cam(self):
        if self.state == 1:
            ret,img = self.cap.read()
            height = int(img.shape[0])
            width = int(img.shape[1])
            img = cv2.resize(img,dsize=(int(width/2), int(height/2)))
            cv2.imshow("image",img)
            cv2.moveWindow('window name', 300, 500)

            self.end = time.time()
            self.ela_time = self.end-self.start

    def keyboardinterrupt(self):
        traceback.print_exc()
        self.cap.release()


experiment = Experiment()  

try:
    while True:
        print("start")
        experiment.key()
        experiment.create_canvas()
        experiment.cam()
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

except:
    experiment.keyboardinterrupt()

# class Experiment():
#     def __init__(self):
#         GPIO.setwarnings(False)
#         GPIO.setmode(GPIO.BCM) #GPIOの設定
#         MotorR = motor(ct.const.RIGHT_MOTOR_IN1_PIN,ct.const.RIGHT_MOTOR_IN2_PIN,ct.const.RIGHT_MOTOR_VREF_PIN)
#         MotorL = motor(ct.const.LEFT_MOTOR_IN1_PIN,ct.const.LEFT_MOTOR_IN2_PIN, ct.const.LEFT_MOTOR_VREF_PIN)
#         self.camerastate = 0
    
#     def run(self):
#         if keyboard.read_key() == "A":
#             print("Go straight")
#             MotorR.go(70)
#             MotorL.go(70)
#         elif keyboard.read_key() == "B":
#             print("Back")
#             MotorR.back(70)
#             MotorL.back(70)
#         elif keyboard.read_key() == "C":
#             print("Turn left")
#             MotorR.go(70)
#             MotorL.back(70)
#         elif keyboard.read_key() == "D":
#             print("Turn right")
#             MotorR.back(70)
#             MotorL.back(70)
#         elif keyboard.read_key() == "c":
#             if self.camerastate == 0:
#                 self.cap = cv2.VideoCapture(0)
#                 self.camerastate == 1
#             ret,img = self.cap.read()
#             cv2.imshow("image",img)

#     def keyboardinterrupt(self):
#         MotorR.stop()
#         MotorL.stop()
#         GPIO.cleanup()
#         cap.release()
#         cv2.destroyAllWindows()

# import pygame
# class Experiment():
#     def __init__(self):
#         pygame.init()  

#     def run(self):
#         print("b")
#         self.pressed_key = pygame.key.get_pressed()
#         print("c")
#         if self.pressed_key[K_LEFT]:
#             print("Turn left")
#         elif self.pressed_key[K_RIGHT]:
#             print("Turn left")

#                 # イベント処理
#         for event in pygame.event.get():
#             # キーを押したとき
#             if event.type == KEYDOWN:
#                 # ESCキーなら終了
#                 if event.key == K_ESCAPE:
#                     pygame.quit()
#                     sys.exit()
    
#     def cam(self):
#         if self.state == 1:
#             ret,img = self.cap.read()
#             cv2.imshow("image",img)
    
#     def keyboardinterrupt(self):
#         # self.cap.release()
#         # cv2.destroyAllWindows()
#         return True