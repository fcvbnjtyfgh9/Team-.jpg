import cv2
import numpy as np
import pytesseract

def detect_traffic_lights(image_path):

    image = cv2.imread(image_path)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=80, param1=200, param2=11, minRadius=14, maxRadius=40)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0, :]:
    
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            
        
            roi = hsv_image[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]
            mean_color = cv2.mean(roi)
            
            # 색상 판별
            if mean_color[0] > 130 and mean_color[0] < 180: 
                cv2.putText(image, 'Red', (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            elif mean_color[0] > 80 and mean_color[0] < 90:  
                cv2.putText(image, 'Green', (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imshow('Detected Traffic Lights', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = r'C:\Users\user\image\test3.png'
detect_traffic_lights(image_path)

# 도로의 초록색 표지판을 기준으로 작성
# 초록색 사각형 표지판 인식 시 출력되는 이미지에 표지판을 표시하고, 'Square Sign Found' 출력
# 이미지를 로드하지 못할 경우, 콘솔창에 'Fail to load image.' 출력
# 사각형 표지판만 인식하기 위해 일정 크기 이상의 사각형만 표시

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

# 이미지 파일을 읽어옴
image_path = './image/test6.webp'
image = cv2.imread(image_path)

if image is not None:
    # 네모 모양의 표지판을 찾음 (최소 둘레 길이를 100으로 설정)
    square_sign_found = find_square_sign(image, min_contour_length=95)

    # 네모 모양의 표지판이 있으면 메시지를 표시
    if square_sign_found:
        cv2.putText(image, 'Square Sign Found', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 이미지를 표시
    cv2.imshow('Square Sign Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to load image.")


# Detect Text in Image
# Tesseract OCR 경로 설정 (설치된 경로에 맞게 수정하세요)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\user\tesseract\tesseract.exe'

def detect_text(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load the image at path {image_path}")
        return "", None  # 이미지 로드 실패 시 빈 문자열과 None 반환

    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지 전처리 (예: 가우시안 블러 적용)
    preprocessed_image = cv2.GaussianBlur(gray, (5, 5), 0)

    # 흰색 글씨 추출을 위한 이진화 (thresholding)
    _, binary_image = cv2.threshold(preprocessed_image, 200, 255, cv2.THRESH_BINARY_INV)

    # 텍스트 감지 (언어 설정 추가)
    text = pytesseract.image_to_string(binary_image, lang='eng')

    return text, image

def main():
    # 표지판 이미지 경로 설정
    image_path = r'C:\Users\user\image\test31.jp'

    # 텍스트 감지
    detected_text, image = detect_text(image_path)

    if detected_text and image is not None:
        # 결과 출력
        print("Detected Text:")
        print(detected_text)

        # 이미지 표시
        cv2.imshow('detect_text', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Text detection failed.")

if __name__ == "__main__":
    main()

