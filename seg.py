# import urllib.request

IMAGE_FILENAMES = ['segmentation_input_rotation0.jpg']

# for name in IMAGE_FILENAMES:
#   url = f'https://storage.googleapis.com/mediapipe-assets/{name}'
#   urllib.request.urlretrieve(url, name)

import cv2
# from google.colab.patches import cv2_imshow
import math

# Height and width that will be used by the model
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# Performs resizing and showing the image
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow("test",img)
  cv2.waitKey(0)


# Preview the image(s)
images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
for name, image in images.items():
  print(name)
  resize_and_show(image)

# step1: 모듈 가져오기
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white

# step2: 추론 객체 만들기
# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(model_asset_path='models\deeplab_v3.tflite') # 모델 경로 수정
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)
# 리소스 해제 때문에 with 씀
with vision.ImageSegmenter.create_from_options(options) as segmenter:

  # Loop through demo image(s)
  for image_file_name in IMAGE_FILENAMES:
    
    # step3: 데이터 불러오기 
    # Create the MediaPipe image file that will be segmented
    image = mp.Image.create_from_file(image_file_name)

    # step4: 추론
    # Retrieve the masks for the segmented image
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask

    # step5: 화면에 그려줌
    # Generate solid color images for showing the output segmentation mask.
    image_data = image.numpy_view()
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
    output_image = np.where(condition, fg_image, bg_image)

    print(f'Segmentation mask of {name}:')
    resize_and_show(output_image)