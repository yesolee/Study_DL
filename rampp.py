'''
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
'''
# 사진을 넣으면 글자가 나오고 글자를 가지고 사진을 해석할 수 있다. (이미지 해시태그 뽑는거)
# tag2text가 문장으로 만들어 주기 때문에 더 유용함
# tag2text inference.py 파일도 가져와 step1~5 재구성해주면 됨

# step0
# 1. install git
# 2. pip install git+https://github.com/xinyu1205/recognize-anything.git 실행하면 아래 import문 활성화됨
# 3. 모델 다운로드 https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth

# step1: import modules
import numpy as np
import random
import torch
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

# step2: create inference object
model_path = "ram_plus_swin_large_14m.pth"
model = ram_plus(pretrained=model_path,
                            image_size=384,
                            vit='swin_l')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# step3: load data
image_path = "1641173_2291260800.jpg"
transform = get_transform(image_size=384)
image = transform(Image.open(image_path)).unsqueeze(0).to(device)

# step4: inference
res = inference(image, model)

# step5: post processing
print("Image Tags: ", res[0])
