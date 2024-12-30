from ultralytics import YOLO
import cv2
import numpy as np
import tkinter as tk # Tkinter
from threading import Thread

from data_class import *
from KeyBoardPointData import KeyBoardPos
from KeyPUT import PressKey, ReleaseKey, SCAN_CODE

# Step 5: 학습된 모델로 추론
model_path = "MODEL\Finger_YOLO_n\\best.pt"
model = YOLO(model_path)

# Step 6: 실시간 추론
cap = cv2.VideoCapture(0)  # 웹캠 사용
assert cap.isOpened(), "카메라를 열 수 없습니다."
count = 0

KEYBOARD_FRAMECUT = 2

CAM_WIDTH   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
CAM_HEIGHT  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

DrowCanvers = []

# Keyboard_X_size = 0.068
# Keyboard_angle  = 10

# Keyboard_X_pos  = 0.55
# Keyboard_Y_pos  = 0.51

KeyBoardPoint = []

def KeyBoardPointF(Keyboard_X_size,
                    Keyboard_angle, 
                    Keyboard_X_pos, 
                    Keyboard_Y_pos):
    global KeyBoardPoint

    Keyboard_Y_size = Keyboard_X_size * np.sin(np.deg2rad(Keyboard_angle))

    KeyBoardPoint = []
    for key in KeyBoardPos.keys():
        KeyBoardPoint.append(
            KeyBoard(
                key, 
                KeyBoardPos[key], 
                Vector(
                    -CAM_WIDTH*(KeyBoardPos[key].x*Keyboard_X_size-Keyboard_X_pos),
                    -CAM_HEIGHT*(KeyBoardPos[key].y*Keyboard_Y_size-Keyboard_Y_pos),
                    CAM_WIDTH*KeyBoardPos[key].width*Keyboard_X_size,
                    CAM_HEIGHT*KeyBoardPos[key].height*Keyboard_Y_size,
                ),
                False, 
                KEYBOARD_FRAMECUT
                ))
    
def WinUI():
    Keyboard_X_size=0
    Keyboard_angle =0
    Keyboard_X_pos =0
    Keyboard_Y_pos =0

    # GUI 설계
    win = tk.Tk() # 인스턴스 생성
    win.title("set") # 제목 표시줄 추가
    win.geometry("320x240") # 지오메트리: 너비x높이+x좌표+y좌표
    win.resizable(False, False) # x축, y축 크기 조정 비활성화

    def select(self):
        KeyBoardPointF(select_Keyboard_X_size_.get(), 
                       select_Keyboard_angle_.get(), 
                       select_Keyboard_X_pos_.get(), 
                       select_Keyboard_Y_pos_.get())
        
    select_Keyboard_X_size_=tk.Scale(win, variable=tk.StringVar(), command=select,   orient="horizontal", showvalue=False, to=0.1, length=300, resolution=0.001)
    select_Keyboard_angle_=tk.Scale(win, variable=tk.StringVar(), command=select,    orient="horizontal", showvalue=False, to=120, length=300, resolution=0.1)
    select_Keyboard_X_pos_=tk.Scale(win, variable=tk.StringVar(), command=select,    orient="horizontal", showvalue=False, to=1, length=300, resolution=0.01)
    select_Keyboard_Y_pos_=tk.Scale(win, variable=tk.StringVar(), command=select,    orient="horizontal", showvalue=False, to=1, length=300, resolution=0.01)
    select_Keyboard_X_size_.pack()
    select_Keyboard_angle_.pack()
    select_Keyboard_X_pos_.pack()
    select_Keyboard_Y_pos_.pack()

    win.mainloop() #GUI 시작

winUI = Thread(target=WinUI, args=())
#winUI.daemon = True
    
winUI.start()
#winUI.join()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv11 모델 추론
    results = model.predict(
        source=frame, 
        #imgsz=640,
        stream=True,
        #device='cpu',
        verbose=False)

    for KeyPoint in KeyBoardPoint: 
        if KeyPoint.put: 
            if KeyPoint.framcut >= KEYBOARD_FRAMECUT:
                print("release")
                ReleaseKey(SCAN_CODE[KeyPoint.key])
                KeyPoint.put = False

            KeyPoint.framcut += 1

    # 박스 그리기
    for result in results:
        boxes = result.boxes.xyxy.numpy()  # [x1, y1, x2, y2]
        

        for box in boxes:
            point = Vector2((box[0]+box[2])/2, box[3])

            cv2.line(frame, (int(point.x-10), int(point.y)), (int(point.x+10), int(point.y)), (255,255,255), 1)

            for KeyPoint in KeyBoardPoint:
                if ((KeyPoint.localpos.x - point.x)**2 <= KeyPoint.localpos.width**2
                and (KeyPoint.localpos.y - point.y)**2 <= KeyPoint.localpos.height**2):
                    if not KeyPoint.put:
                        print("press", KeyPoint.key)
                        PressKey(SCAN_CODE[KeyPoint.key])
                        KeyPoint.put = True
                    KeyPoint.framcut = 0

                    cv2.rectangle(frame, 
                                (int(KeyPoint.localpos.x+KeyPoint.localpos.width), 
                                int(KeyPoint.localpos.y+KeyPoint.localpos.height)), 
                                (int(KeyPoint.localpos.x-KeyPoint.localpos.width), 
                                int(KeyPoint.localpos.y-KeyPoint.localpos.height)), 
                                (0,100,255),-1)
                    
                    

    # for p in DrowCanvers:
    #     cv2.line(frame, p, p, (0,0,0), 2)
    # # for i in range(64):
    # #     cv2.line(frame, (0, i*10), (640, i*10), (0,0,255))
    # # cv2.line(frame, (320, 0), (320, 500), (0,0,255))

    for value in KeyBoardPoint:
        cv2.rectangle(frame, 
                    (int(value.localpos.x+value.localpos.width), 
                     int(value.localpos.y+value.localpos.height)), 
                    (int(value.localpos.x-value.localpos.width), 
                     int(value.localpos.y-value.localpos.height)), 
                     (0,100,255))


    # 출력
    frame = cv2.flip(frame,1)
    cv2.imshow("YOLOv11 Finger Tip Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
