1.import cv2
import numpy as np
import pytesseract
image = cv2.imread("C:\Users\ip103\Documents\카카오워크 받은 파일\ClipboardImage_2023-12-04_213526.png")
lower_red = np.array([0, 100, 0])
upper_red = np.array([100, 255, 100])
red_mask = cv2.inRange(image, lower_red, upper_red)
contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
largest_contour = max(contours, key=cv2.contourArea)
cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
plate_roi = image[largest_contour[0, 0]:largest_contour[-1, 0], largest_contour[0, 1]:largest_contour[-1, 1]]
plate_roi_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(plate_roi_gray, config="--psm 7")
print(text)
2. 
import cv2
import numpy as np
import pytesseract
image = cv2.imread("C:\Users\ip103\Downloads\test3.png")
lower_yellow = np.array([0, 100, 100])
upper_yellow = np.array([100, 255, 255])
yellow_mask = cv2.inRange(image, lower_yellow, upper_yellow
contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
largest_contour = max(contours, key=cv2.contourArea)
cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
plate_roi = image[largest_contour[0, 0]:largest_contour[-1, 0], largest_contour[0, 1]:largest_contour[-1, 1]]
plate_roi_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(plate_roi_gray, config="--psm 7")
print(text)
3.
import cv2
import numpy as np
import pytesseract
image = cv2.imread("C:\Users\ip103\Documents\카카오워크 받은 파일\ClipboardImage_2023-12-04_213533.png")
lower_green = np.array([0, 100, 0])
upper_green = np.array([100, 255, 100])
green_mask = cv2.inRange(image, lower_green, upper_green)
contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
largest_contour = max(contours, key=cv2.contourArea)
cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
plate_roi = image[largest_contour[0, 0]:largest_contour[-1, 0], largest_contour[0, 1]:largest_contour[-1, 1]]
plate_roi_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(plate_roi_gray, config="--psm 7").
print(text)
import cv2
import numpy as np
import pytesseract

