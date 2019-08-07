import sys
from ctypes import *
import math
import random
import os
import cv2
import numpy as numpy
import time
from pdb import set_trace
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from cardetection import darknet
from cardetection.utils import detection_mapper, constants

class Detector:
    def fix_names_path(self, metaPath, package_root_dir):
        try:
            with open(metaPath, 'r') as metaFH:
                lines = metaFH.read().split('\n')
                names_line_index = [i for i, line in enumerate(lines) if line.startswith('names')][0]
                coco_names = lines[names_line_index].split(' ')[-1].split('/')[-1]
                lines[names_line_index] = ('names = ' + os.path.join(package_root_dir, constants.DATA_DIR, coco_names.strip())).replace('\\','/')    
            
            with open(metaPath, 'w') as metaFH:
                metaFH.write('\n'.join(lines))
        except:
            pass

    def __init__(self, objectName, frameWidth, frameHeight, roiBox, onCapturedListener, model_name='yolov3-tiny', dataset_name='coco'):        
        PACKAGE_ROOT_DIR = os.path.dirname(__file__)
        CONFIG_PATH = os.path.join(PACKAGE_ROOT_DIR, constants.CONFIG_DIR, f'{model_name}.{constants.CONFIG_EXTENSION}')
        WEIGHT_PATH = os.path.join(PACKAGE_ROOT_DIR, f'{model_name}.{constants.WEIGHTS_EXTENSION}')
        META_PATH = os.path.join(PACKAGE_ROOT_DIR, constants.CONFIG_DIR, f'{dataset_name}.{constants.DATA_EXTENSION}')
    
        # setup network
        self.netMain =  darknet.load_net_custom(CONFIG_PATH.encode("ascii"), WEIGHT_PATH.encode("ascii"), 0, 1)
        self.fix_names_path(META_PATH, PACKAGE_ROOT_DIR)
        '''
        try:
            with open(META_PATH) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)

                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            self.altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
        '''
        self.metaMain =  darknet.load_meta(META_PATH.encode("ascii"))
        
        # setup input images
        self.frameWidth, self.frameHeight = frameWidth, frameHeight
        self.darknetImage = darknet.make_image(darknet.network_width(self.netMain), darknet.network_height(self.netMain), 3)
        self.objectName = objectName
        self.onCapturedListener = onCapturedListener

    
    def __resizeFrameAndBbox(self, frame, detections, width, heigth):
        bboxes = BoundingBoxesOnImage(
            [BoundingBox(x1=det['xmin'], y1=det['ymin'], x2=det['xmax'], y2=det['ymax']) for det in detections], 
            shape=frame.shape
        )

        resizeOp = iaa.Resize({'width': width, 'height': heigth})
        resizedFrame, bboxes = resizeOp(image=frame, bounding_boxes=bboxes)

        return resizedFrame, [detection_mapper.createDetection(bbox.x1, bbox.y1, bbox.x2, bbox.y2, detections[i]['confidence'], detections[i]['label'])
                                for i, bbox in enumerate(bboxes.bounding_boxes)]

    def detect(self, frame):
        originalFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        try:
            resizedFrame = cv2.resize(originalFrame,
                                    (darknet.network_width(self.netMain),
                                        darknet.network_height(self.netMain)),
                                    interpolation=cv2.INTER_LINEAR)
        except Exception:
            print("Error when resize frame")
            return
        
        darknet.copy_image_from_bytes(self.darknetImage, resizedFrame.tobytes())
        darkNetDetections = darknet.detect_image(self.netMain, self.metaMain, self.darknetImage, thresh=.25)

        detections = detection_mapper.convertDarknetDetections_to_Detections(darkNetDetections)
        _, detections = self.__resizeFrameAndBbox(resizedFrame, detections, self.frameWidth, self.frameHeight)
        return detections
        