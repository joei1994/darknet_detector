import cv2
import os
import time

from cardetection.detector import Detector
from cardetection.captor import BoxCaptor, LineCaptor
from cardetection.tracker.sort import Sort

current_milli_time = lambda: int(round(time.time() * 10000))

def draw_bbox(frame, detections):
    for det in detections:
        xmin, ymin, xmax, ymax = det['xmin'], det['ymin'], det['xmax'], det['ymax']
        
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)

        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 1)
        
        cv2.putText(frame, f"{det['label']}", (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
    return frame

def create_roi_box(x, y, width, height):
    roi_box = {
        'xmin': int(x - width / 2),
        'ymin': int(y - height / 2),
        'xmax': int(x + width / 2),
        'ymax': int(y + height / 2)
    }

    return roi_box

def save_frame(frame, track, output_dir='./capture'):
    margin = 30
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cropped_frame = frame[track[1] - margin:track[3] + margin, track[0] - margin:track[2] + margin]
    ouptut_path = os.path.join(output_dir, f'{current_milli_time()}.jpg')
    if not os.path.exists(ouptut_path):
        cv2.imwrite(ouptut_path, cropped_frame)
    
def main():
    frameWidth, frameHeight = 1920, 1080  
    
    detector = Detector()
    #roi_box = create_roi_box(x=680, y=570, width=640, height=400)  
    roi_box = create_roi_box(x=680, y=520, width=560, height=310)
    captor = BoxCaptor(Sort(), ['car', 'truck'], roi_box)

    lines = [((2000, 550), (100, 550))]
    roi_box = create_roi_box(x=680, y=520, width=530, height=300)
    lineCaptor = LineCaptor(Sort(), ['car', 'truck'], lines, roi_box)

    #cap = cv2.VideoCapture('rtsp://admin:iapp2019@192.168.1.64/1')
    cap = cv2.VideoCapture('./1.mp4')
    cap.set(0, 90000)

    while True:
        _, frame = cap.read()
        
        detections = detector.detect(frame)
        capturedDetection = lineCaptor.capture(detections)
        if capturedDetection is not None:
            save_frame(frame, capturedDetection)
    
        frame = draw_bbox(frame, detections)
        cv2.rectangle(frame, (roi_box['xmin'], roi_box['ymin']), (roi_box['xmax'], roi_box['ymax']), (255, 0, 0), 1)
        cv2.line(frame, (lines[0][0]), (lines[0][1]), (255,0,0))
        cv2.imshow('Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


if __name__ == '__main__':
    main()