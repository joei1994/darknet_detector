import numpy as np
import time

current_milli_time = lambda: int(round(time.time() * 1000))

class Captor:
    def __init__(self, tracker, targetLabels, roiBox, detectionByRoiBoxAreaRatio = .3, iouThreshold=.6, capturePeriodMillis = 1000):
        self.tracker = tracker
        self.targetLabels = targetLabels
        self.roiBox = roiBox
        self.detectionByRoiBoxAreaRatio = detectionByRoiBoxAreaRatio
        self.iouThreshold = iouThreshold

        self.capturedIds = set()
        self.lastestCapturedTime = current_milli_time()
        self.capturePeriodMillis = capturePeriodMillis
        
    def __bboxArea(self, xmin, ymin, xmax, ymax):
        diff_x = xmax - xmin
        diff_y = ymax - ymin

        return diff_x * diff_y

    def __iou(self, b1_xmin, b1_ymin, b1_xmax, b1_ymax, b2_xmin, b2_ymin, b2_xmax, b2_ymax):
        intersect_xmin = max(b1_xmin, b2_xmin)
        intersect_ymin = max(b1_ymin, b2_ymin)
        intersect_xmax = min(b1_xmax, b2_xmax)
        intersect_ymax = min(b1_ymax, b2_ymax)
    
        intersect_area = max(0, intersect_xmax - intersect_xmin + 1) * max(0, intersect_ymax - intersect_ymin + 1)
    
        box1_area = (b1_xmax - b1_xmin + 1) * (b1_ymax - b1_ymin + 1)
        box2_area = (b2_xmax - b2_xmin + 1) * (b2_ymax - b2_ymin + 1)
    
        total_area = box1_area + box2_area - intersect_area
        return intersect_area  / total_area
    
    def __filterDetections(self, detections, minArea, targetLabels):
        # Filter out non Vehicle 
        detections = [det for det in detections if det['label'] in targetLabels]

        # Filter out too small detections
        detections = [det for det in detections if self.__bboxArea(det['xmin'], det['ymin'], det['xmax'], det['ymax']) > minArea]
        return detections

    def __assignIds(self, detections, tracker):
        dets = np.asarray([[det['xmin'], det['ymin'], det['xmax'], det['ymax'], det['confidence']] for det in detections])
        tracks = tracker.update(dets)
    
        return np.asarray(tracks, dtype='int32')

    def __capture(self, tracks):
        self.capturedIds = set(track[4] for track in tracks).intersection(self.capturedIds)
        for track in tracks:
            objectId = track[4]
            if (
                objectId not in self.capturedIds and
                self.__iou(track[0], track[1], track[2], track[3], self.roiBox['xmin'], self.roiBox['ymin'], self.roiBox['xmax'], self.roiBox['ymax']) > self.iouThreshold
               ):
                currentCapturedTime = current_milli_time()
                timeDiff = currentCapturedTime - self.lastestCapturedTime
                if timeDiff > self.capturePeriodMillis:
                    self.capturedIds.add(objectId)
                    self.lastestCapturedTime = currentCapturedTime
                    return track
        return None

    def capture(self, detections):
        # Filter out some detections
        roiBoxArea = self.__bboxArea(self.roiBox['xmin'],
                                      self.roiBox['ymin'],
                                      self.roiBox['xmax'],
                                      self.roiBox['ymax'])
        detections = self.__filterDetections(detections, 
                                            minArea = roiBoxArea * self.detectionByRoiBoxAreaRatio,
                                            targetLabels = self.targetLabels)
        
        # Assign ID
        tracks = self.__assignIds(detections, self.tracker)
        capturedTrack = self.__capture(tracks)

        return capturedTrack

        
        




        
        
        