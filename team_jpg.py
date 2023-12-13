import cv2
import numpy as np
import pytesseract

# Tesseract OCR 경로 설정 (설치된 경로에 맞게 수정하세요)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\user\tesseract\tesseract.exe'

def find_square_sign(image, min_contour_length=100):
    # 이미지를 HSV 색 공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 초록색 표지판의 색상 범위를 정의 (조정 가능)
    lower_green = np.array([55, 55, 55])
    upper_green = np.array([90, 255, 255])

    # 색상 범위에 해당하는 부분을 이진화
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 모폴로지 연산을 사용하여 노이즈 제거
    kernel = np.ones((3, 4), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 이미지에서 초록색 표지판의 윤곽을 찾음
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선을 그려서 표시하고 네모 모양 확인
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 꼭지점이 4개이면 네모 모양으로 판단
        if len(approx) == 4:
            # 사각형의 둘레가 min_contour_length 이상인 경우에만 표시
            if cv2.arcLength(contour, True) >= min_contour_length:
                cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)

    # 네모 모양의 표지판이 있는지 여부를 반환
    return len([contour for contour in contours if len(cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)) == 4 and cv2.arcLength(contour, True) >= min_contour_length]) > 0

def detect_text_traffic_lights_and_square_sign(image_path):
    image = cv2.imread(image_path)

    # Text detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preprocessed_image = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_image = cv2.threshold(preprocessed_image, 180, 255, cv2.THRESH_BINARY_INV)
    detected_text = pytesseract.image_to_string(binary_image, lang='eng')

    # Square sign detection
    square_sign_found = find_square_sign(image, min_contour_length=95)

    # Traffic light detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=80,
                               param1=181, param2=20, minRadius=10, maxRadius=40)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

            roi = hsv_image[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]
            mean_color = cv2.mean(roi)

            # 색상 판별
            if mean_color[0] > 105 and mean_color[0] < 120:
                cv2.putText(image, 'Red', (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            elif mean_color[0] > 80 and mean_color[0] < 90:
                cv2.putText(image, 'Green', (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 결과 반환
    return detected_text, image, square_sign_found

def main():
    # 이미지 경로 설정
    image_path = r'C:\Users\user\image\test7.jpg'

    # 텍스트, 교통 신호등, 그리고 네모 모양의 표지판 감지
    detected_text, combined_image, square_sign_found = detect_text_traffic_lights_and_square_sign(image_path)

    if detected_text and combined_image is not None:
        # 결과 출력
        print("Detected Text:")
        print(detected_text)

        # 네모 모양의 표지판이 있으면 메시지를 표시
        if square_sign_found:
            cv2.putText(combined_image, 'Square Sign Found', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 이미지 표시
        cv2.imshow('Detect Text, Traffic Lights, and Square Sign', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Detection failed.")

if __name__ == "__main__":
    main()
