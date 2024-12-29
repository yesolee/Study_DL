# step1: 모듈 불러오기
import easyocr

# step2: 추론기 만들기
reader = easyocr.Reader(['ko','en']) # this needs to run only once to load the model into memory

# ste3: 데이터 불러오기
data = 'menu.jpg'

# step4: 추론하기
result = reader.readtext(data, detail=0) # detail이 0: 위치 정보 출력 안함
print(result)
# step5: 후처리 post processing (예: pdf에서 개인정보 찾기)
# if dddd = "주민등록등본":