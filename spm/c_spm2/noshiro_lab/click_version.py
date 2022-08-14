import keyboard
import constant as ct
import cv2
import time
import tkinter as tk
import traceback
import RPi.GPIO as GPIO
from motor import motor

class Experiment():
    def __init__(self):
        self.camerastate = 0
        self.start = 0.0
        self.end = 0.0
        self.width = 300
        self.height = 200
        
        self.MotorR = motor(ct.const.RIGHT_MOTOR_IN1_PIN,ct.const.RIGHT_MOTOR_IN2_PIN,ct.const.RIGHT_MOTOR_VREF_PIN)
        self.MotorL = motor(ct.const.LEFT_MOTOR_IN1_PIN,ct.const.LEFT_MOTOR_IN2_PIN, ct.const.LEFT_MOTOR_VREF_PIN)
        
        self.root = tk.Tk()
        button_up = tk.Button(self.root, text="UP", command=self.straight)
        button_dw= tk.Button(self.root, text="DOWN", command=self.back)
        button_r= tk.Button(self.root, text="RIGHT", command=self.right)
        button_l= tk.Button(self.root, text="LEFT", command=self.left)

        button_up.place(x=150, y=20, width=100, height=50)
        button_dw.place(x=10, y=120, width=100, height=50)
        button_r.place(x=150, y=120, width=100, height=50)
        button_l.place(x=290, y=120, width=100, height=50)

        self.root.mainloop()

        self.ela_time = self.end-self.start
        self.root.geometry(f"{self.width}x{self.height}")
        self.root.title("制限時間")
        self.root.configure(bg = "red")

        # self.canvas.pack()

    def create_canvas(self):
        time =180.0-self.ela_time
        m = str(int(time/60)).rjust(2,"0")
        s = str(int(time%60)).rjust(2,"0") 
        text = f"{m}:{s}"
        self.canvas.create_text(self.width/2,self.height/2,text=text,font=("Times New Roman",100),tag='Y')
        self.canvas.update()
        self.canvas.delete('Y')
    
    def key(self):
        if keyboard.is_pressed("up") == True:
            self.straight()
        elif keyboard.is_pressed("down") == True:
            self.back()
        elif keyboard.is_pressed("right") == True:
            self.right()
        elif keyboard.is_pressed("left") == True:
            self.left()
        elif keyboard.is_pressed("c") == True:
            self.cam_start()
        else:
            self.motor_stop()
    
    def straight(self):
        print("Go straight")
        self.MotorR.go(70)
        self.MotorL.go(70)
    
    def back(self):
        print("Go backward")
        self.MotorR.back(70)
        self.MotorL.back(70)
    
    def right(self):
        print("Turn right")
        self.MotorR.back(70)
        self.MotorL.go(70)

    def left(self):
        print("Turn left")
        self.MotorR.go(70)
        self.MotorL.back(70)
        
    def motor_stop(self):
        self.MotorR.stop()
        self.MotorL.stop()
        
    def cam_start(self):
        if self.camerastate == 0:
            self.cap = cv2.VideoCapture(0)
            
            self.start = time.time()
            self.camerastate = 1

    def cam(self):
        if self.camerastate == 1:
            ret,img = self.cap.read()
            height = int(img.shape[0])
            width = int(img.shape[1])
#             img = cv2.resize(img,dsize=(int(width/2), int(height/2)))
            cv2.imshow("image",img)
            cv2.moveWindow('window name', 300, 500)

            self.end = time.time()
            self.ela_time = self.end-self.start

    def keyboardinterrupt(self):
        self.MotorR.stop()
        self.MotorL.stop()
        traceback.print_exc()
#         self.cap.release()


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
