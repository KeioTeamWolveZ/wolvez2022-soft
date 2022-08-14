import cv2
import tkinter as tk
from PIL import Image, ImageTk

def btn_clicked():
	print("Button Clicked")

# ウィンドウ作成
root = tk.Tk()

# ボタンの作成と配置
button_up = tk.Button(root, text="UP", command=btn_clicked)
button_dw= tk.Button(root, text="DOWN", command=btn_clicked)
button_r= tk.Button(root, text="RIGHT", command=btn_clicked)
button_l= tk.Button(root, text="LEFT", command=btn_clicked)

button_up.place(x=150, y=20, width=100, height=50)
button_dw.place(x=10, y=120, width=100, height=50)
button_r.place(x=150, y=120, width=100, height=50)
button_l.place(x=290, y=120, width=100, height=50)


cap=cv2.VideoCapture(0)

# メインループ
while True:
	ret,frame=cap.read()
	cv2.imwrite('temp.jpg',frame)
	image_bgr = cv2.imread("temp.jpg")
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
	image_pil = Image.fromarray(image_rgb) # RGBからPILフォーマットへ変換
	image_tk  = ImageTk.PhotoImage(image_pil) # ImageTkフォーマットへ変換

	canvas = tk.Canvas(root, width=frame.shape[0], height=frame.shape[1]) # Canvas作成
	canvas.pack()
	canvas.create_image(0, 0, image=image_tk, anchor='nw') # ImageTk 画像配置
	root.mainloop()
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()