import numpy as np
import time

current_milli_time = lambda: int(round(time.time() * 1000))

class Captor:
    def __init__(self, tracker, targetLabels = ['car', 'truck'], capturePeriodMillis = 1000):
        self.tracker = tracker
        self.targetLabels = targetLabels
        self.capturePeriodMillis = capturePeriodMillis

        self.capturedIds = set()
        self.lastestCapturedTime = current_milli_time()
       
    def _bboxArea(self, xmin, ymin, xmax, ymax):
        diff_x = xmax - xmin
        diff_y = ymax - ymin

        return diff_x * diff_y
    
    def _filterDetections(self, detections, targetLabels):
        # Filter out non Vehicle 
        detections = [det for det in detections if det['label'] in targetLabels]
        return detections

    def _assignIds(self, detections, tracker):
        dets = np.asarray([[det['xmin'], det['ymin'], det['xmax'], det['ymax'], det['confidence']] for det in detections])
        tracks = tracker.update(dets)
    
        return np.asarray(tracks, dtype='int32')

    def _capture(self, tracks, captureCondition):
        self.capturedIds = set(track[4] for track in tracks).intersection(self.capturedIds)
        for track in tracks:
            objectId = track[4]
            if (
                objectId not in self.capturedIds and
                captureCondition(track) 
               ):
                currentCapturedTime = current_milli_time()
                timeDiff = currentCapturedTime - self.lastestCapturedTime
                if timeDiff > self.capturePeriodMillis:
                    self.capturedIds.add(objectId)
                    self.lastestCapturedTime = currentCapturedTime
                    return track
        return None
        
class BoxCaptor(Captor):
    def __init__(self, tracker, targetLabels, roiBox, filterDetectionByRoiBoxAreaRatio = .3, iouThreshold = .6, capturePeriodMillis = 1000):
        super().__init__(tracker, targetLabels, capturePeriodMillis)
        self.roiBox = roiBox
        self.iouThreshold = iouThreshold
        self.filterDetectionByRoiBoxAreaRatio = filterDetectionByRoiBoxAreaRatio
    
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
    
    def _filterDetections(self, detections, targetLabels, minArea):
         # Default filter 
        detections = super()._filterDetections(detections, targetLabels)

        # Filter out too small detections
        detections = [det for det in detections if super(BoxCaptor, self)._bboxArea(det['xmin'], det['ymin'], det['xmax'], det['ymax']) > minArea]
        return detections
    
    def capture(self, detections):
        # Filter out some detections
        roiBoxArea = super()._bboxArea(self.roiBox['xmin'],
                                        self.roiBox['ymin'],
                                        self.roiBox['xmax'],
                                        self.roiBox['ymax'])
        detections = self._filterDetections(detections,
                                            targetLabels = self.targetLabels,
                                            minArea = roiBoxArea * self.filterDetectionByRoiBoxAreaRatio)
        
        # Assign ID
        tracks = self._assignIds(detections, self.tracker)
        capturedTrack = self._capture(tracks, captureCondition = lambda track: self.__iou(track[0], track[1], track[2], track[3], self.roiBox['xmin'], self.roiBox['ymin'], self.roiBox['xmax'], self.roiBox['ymax']) > self.iouThreshold)

        return capturedTrack




        
        
        