import cv2
import numpy as np

def detect_traffic_lights(image_path):
   
    image = cv2.imread(image_path)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=200, param2=35, minRadius=10, maxRadius=40)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0, :]:
    
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            
        
            roi = hsv_image[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]
            mean_color = cv2.mean(roi)
            
            # 색상 판별
            if mean_color[0] > 0 and mean_color[0] < 20: 
                cv2.putText(image, 'Red', (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            elif mean_color[0] > 60 and mean_color[0] < 80:  
                cv2.putText(image, 'Green', (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
  
    cv2.imshow('Detected Traffic Lights', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = 'C:\\Users\\user\\image\\light3.jpg'
detect_traffic_lights(image_path)
