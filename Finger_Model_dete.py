import os
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from KeyBoardPointData import KeyBoardPoint
from KeyPUT import PressKey, ReleaseKey, SCAN_CODE


# Detectron2 모델 설정
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 클래스 수 (배경 제외)

cfg.MODEL.WEIGHTS = "Finger_Nomal\\model_final.pth"

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.98  # 예측 임계값
cfg.MODEL.DEVICE = "cpu"  # GPU 사용("cuda") 또는 CPU 사용("cpu")
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.AMP.ENABLED = True  # FP16 활성화

# Predictor 생성
predictor = DefaultPredictor(cfg)


# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠, 다른 번호는 추가 웹캠을 의미


CAM_WIDTH   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
CAM_HEIGHT  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

DrowCanvers = []

size = -62
pos = (320, 338)
Rotation = 1/8

for key in KeyBoardPoint.keys():
    KeyBoardPoint[key] = [(KeyBoardPoint[key][0]*size+pos[0], KeyBoardPoint[key][1]*size*Rotation+pos[1]), False]


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectron2로 예측 수행
    outputs = predictor(frame)
    
	# 감지된 객체의 박스 좌표 가져오기
    instances = outputs["instances"].to("cpu")  # CPU로 이동
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None

    # if boxes is not None:
    #     for box in boxes:
    #         point = ((box[0]+box[2])/2, box[3])

    #         cv2.line(frame, (int(point[0]-10), int(point[1])), (int(point[0]+10), int(point[1])), (255,255,255), 1)

    #         DrowCanvers.append((int(point[0]), int(point[1])))

    #         x_Min = (size/2)**2
    #         y_Min = (size*Rotation/2)**2
    #         for key in KeyBoardPoint.keys():
    #             if ((KeyBoardPoint[key][0][0] - point[0])**2 < x_Min 
    #             and (KeyBoardPoint[key][0][1] - point[1])**2 < y_Min):
                    
    #                 if not KeyBoardPoint[key][1]:
    #                     PressKey(SCAN_CODE[key])
    #                     KeyBoardPoint[key][1] = True

    #                 break

    #             elif KeyBoardPoint[key][1]: 
                    
    #                 ReleaseKey(SCAN_CODE[key])
    #                 KeyBoardPoint[key][1] = False

    # Detectron2 시각화 도구로 결과 표시
    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(instances)
    result_frame = v.get_image()[:, :, ::-1]

    # 결과 프레임 표시
    frame = cv2.flip(frame,1)
    cv2.imshow("Webcam Detection", result_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()