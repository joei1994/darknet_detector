def __convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def createDetection(xmin, ymin, xmax, ymax, confidenc, label):
    det = {}
    det['xmin'] = int(xmin)
    det['ymin'] = int(ymin)
    det['xmax'] = int(xmax)
    det['ymax'] = int(ymax)
    det['confidence'] = confidenc
    det['label'] = label
    return det

def convertDarknetDetection_to_Detection(darknetDetection):
    x, y, w, h = darknetDetection[2][0],\
        darknetDetection[2][1],\
        darknetDetection[2][2],\
        darknetDetection[2][3]
    xmin, ymin, xmax, ymax = __convertBack(float(x), float(y), float(w), float(h))

    return  createDetection(xmin, ymin, xmax, ymax, darknetDetection[1], darknetDetection[0].decode())
        
def convertDarknetDetections_to_Detections(darknetdetections):
    return [convertDarknetDetection_to_Detection(ddet) for ddet in darknetdetections]