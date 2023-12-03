import cv2
import pytesseract

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

    # 텍스트 감지 (언어 설정 추가)
    text = pytesseract.image_to_string(preprocessed_image, lang='eng')

    return text, image

def main():
    # 표지판 이미지 경로 설정
    image_path = r'C:\Users\user\image\test3.png'

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
