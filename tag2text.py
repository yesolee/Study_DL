# tag2text

# step1: import moduels
import argparse
import numpy as np
import random
import torch
from PIL import Image
from ram.models import tag2text
from ram import inference_tag2text as inference
from ram import get_transform

# step2: create inference object
model_path = 'tag2text_swin_14m.pth'
# delete some tags that may disturb captioning 물체가 아닌 태그를 지워는 것
# 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
delete_tag_index = [127,2961, 3351, 3265, 3338, 3355, 3359]

model = tag2text(pretrained=model_path,
                            image_size=384,
                            vit='swin_b',
                            delete_tag_index=delete_tag_index)
model.threshold = 0.68  # threshold for tagging
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# step3: Load data 
img_path = '1641173_2291260800.jpg'
transform = get_transform(image_size=384)
image = transform(Image.open(img_path)).unsqueeze(0).to(device)

# step4: inference
res = inference(image, model, 'None')

# step5: post processing
print("Model Identified Tags: ", res[0])
print("User Specified Tags: ", res[1])
print("Image Caption: ", res[2])