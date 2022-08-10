import cv2
from datetime import datetime
cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
ret,frame = cap.read() # return a single frame in variable `frame`

while(True):
    ret,frame = cap.read() # return a single frame in variable `frame`
    cv2.imshow('img1',frame) #display the captured image
    print("here")
    if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
        print("saved")
        now= str(datetime.now())[:19].replace(" ","_").replace(":","-").replace(".","_")
        cv2.imwrite(f'/home/pi/Desktop/wolvez2022/Testcode/run_cam/stack_pic/{now}.png',frame)
        
cap.release()
cv2.destroyAllWindows()
        
