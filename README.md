# Team .jpg 팀 프로젝트

### 신호등과 표지판을 인식하고, 신호등의 색과 표지판의 글씨를 출력하는 Open CV 프로그램

**주제 선정 과정**: 첫 회의 당시 팀원들이 가져왔던 의견이 각각 신호등, 표지판, 차선 인식이었고 둘 다 한 이미지에서 확인할 수 있는 요소들이었기 때문에 의견들을 모아 한 프로젝트로 만들자고 결론지음.

**결과**: ![result](https://i.ibb.co/1md9qd1/Clipboard-Image-2023-12-13-170154.png)


**실행 방법**: opencv와 tesseract를 설치한  뒤 team_jpg.py 코드를 실행.

**메인 소스 코드**: team_jpg.py 코드가 최종 완성 코드이며, 세부 코드들은 조원끼리 파트를 나누어 작업함. 세부 코드들은 Details 폴더 속에 기재되어 있음.

**사용한 이미지**: image 폴더 속에 저장되어 있음.

**사용한 프로그램**: opencv, python, tesseract(글자 인식) - https://github.com/UB-Mannheim/tesseract.git

**사용 패키지 및 버전**: python 3.11.5, opencv-python 4.8.1.78, tesseract 0,1,3

---
### 현재 발생하는 오류 사항
1. 텍스트 인식의 불완정성

    - 이미지나 tesseract 자체의 문제로 판단.
2. 신호등 인식 시 신호등이 아닌 것들도 동시에 원으로 인식됨
   
   - color range, circles 내부 요소들을 수정해보았으나 실패함.
---
### 프로젝트를 진행하며 생겼던 문제들
1. 텍스트가 출력되지 않음:

    chat gpt 등의 open ai를 참조하여 작성한 기존의 코드는 '하얀 바탕' 위의 '검은 글씨'를 출력하는 원리인데, 팀에서 인식하고자 한 글씨는 '초록색 표지판' 위의 '하얀 글씨'였기 때문에 텍스트가 검출되지 않았음. 아래의 코드를 추가하여 보완함.
    ```
    $ _, binary_image = cv2.threshold(preprocessed_image, 200, 255, cv2.THRESH_BINARY_INV)
    ```
    그러나 텍스트 인식의 정확도가 떨어짐.(상기한 1번 오류 문항)

2. 강제 푸시로 인한 커밋 오류

    - 기존 과제 제출을 위해 main으로 설정해둔 branch들을 잊고서 원활히 push 되지 않아 -f를 이용해 강제 푸시 하였더니 branch의 내용들이 날아감. 아래 코드를 이용해 복구 했었음.
        ```
        $ git reset --hard HEAD~n  //n is some integer number  
        ```
    - 마찬가지로 push를 위해 -f를 사용했다가 commit 내역이 날아감. 위의 reset 명령어로 복구를 실패하여 당시 마지막으로 commit 했던 다른 팀원의 branch를 push하여 복구함. 
    - **결론점**: 강제 푸시는 가능한 최대한 사용하지 않을 것. 사용하더라도 git clone등의 다른 가상환경에서 test 해보고서 push할 것.

3. 하나의 창에서 실행되지 않음
    - 각자가 만든 코드를 그대로 이어붙이기만 했기때문으로 추측. 이후 chat gpt의 도움을 받아 하나의코드로 합쳤다. 유의점은 녹색표지판을 인식할 때는 컬러 이미지인 채로 인식해야하기 때문에 제일 첫번째로 인식하고 결과 사항을 저장할 것, 그리고 text를 읽을 때 방해가 되서는 안되기 때문에 신호등의 결과는 detect text보다 이후에 저장되어야 한다.
 
---
### 참고 내역
- chat gpt: 코드 작성 도움
- 이하 링크 기재

https://jupiny.com/2019/03/19/revert-commits-in-remote-repository/

https://gr-st-dev.tistory.com/909

https://wandukong.tistory.com/9

https://velog.io/@jiyuninha/Toy-Project1-%EC%98%81%EC%83%81%EC%B2%98%EB%A6%AC%EB%A1%9C-%EC%8B%A0%ED%98%B8%EB%93%B1-%EC%9D%B8%EC%8B%9D%ED%95%98%EA%B8%B02
