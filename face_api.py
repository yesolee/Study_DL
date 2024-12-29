# insightface 깃허브 > examples/demo_analysis.py

# step1: 필요한 모듈 불러오기
import cv2
# 비트맵 2차원 배열을 위해 넘파이 import
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# step2: 추론기 만들기
face = FaceAnalysis()
# 추론기에 옵션넣기
face.prepare(ctx_id=0, det_size=(640,640))

from fastapi import FastAPI, UploadFile

app = FastAPI()

import cv2
import numpy as np
@app.post("/uploadfile/")
async def create_upload_file(file1: UploadFile, file2: UploadFile):

    # STEP 3: Load the input image.
    contents1 = await file1.read()
    contents2 = await file2.read()

    nparr1 = np.fromstring(contents1, np.uint8)
    nparr2 = np.fromstring(contents2, np.uint8)

    img1= cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    # step4: 추론하기 inference
    # 얼굴 위치 찾고, 랜드마크(눈,코,입) 찾기, 성별, 나이, 얼굴 고유 특징 임베딩 5가지 model이 내부에서 동작함(옵션으로 조정 가능)
    faces1 = face.get(img1)
    faces2 = face.get(img2)

    # 평가(확인)
    assert len(faces1)==1
    assert len(faces2)==1

    # step5:

    # 이미지 저장
    # rimg = app.draw_on(img, faces)
    # cv2.imwrite("./t1_output.jpg", rimg)

    # 얼굴간의 상호 유사도 계산 then print all-to-all face similarity
    # 정규화 시킨 임베딩 사용
    face_feat1 = faces1[0].normed_embedding
    face_feat2 = faces2[0].normed_embedding

    face_feat1 = np.array(face_feat1, dtype=np.float32)
    face_feat2 = np.array(face_feat2, dtype=np.float32)

    # 행렬 곱 (코사인 유사도 계산)
    # 모델이 정상적으로 작동하는지 확인하기 위해 같은 사진을 비교해 1이 나오는지 검사
    sims = np.dot(face_feat1, face_feat2.T)
    # -1 ~ 1 사이의 값, 0.4 미만이면 그레이 존이 있다. 0.4 이상이면 동일인 0.2 ~ 0 의심, 0미만은 타인
    print(sims)

    if sims >= 0.4:
        return {"result": "동일인입니다."}
    else:
        return {"result": "동일인이 아닙니다"}
    
# fastapi dev face_api.py 