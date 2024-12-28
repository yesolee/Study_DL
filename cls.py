# 필요한 이미지 다운로드
# import urllib.request

IMAGE_FILENAMES = ['burger.jpg', 'cat.jpg']

# for name in IMAGE_FILENAMES:
#   url = f'https://storage.googleapis.com/mediapipe-tasks/image_classifier/{name}'
#   urllib.request.urlretrieve(url, name)

# # 다운로드 받은 이미지 확인
# import cv2
# # from google.colab.patches import cv2_imshow
# import math

# DESIRED_HEIGHT = 480
# DESIRED_WIDTH = 480

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
# #   cv2_imshow(img)
#   cv2.imshow("test", img)
#   cv2.waitKey(0)

# # Preview the images.

# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='models\\efficientnet_lite2.tflite')
options = vision.ImageClassifierOptions(base_options=base_options, max_results=3) # 3등까지 출력
classifier = vision.ImageClassifier.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file('cat.jpg')

# STEP 4: Classify the input image.
classification_result = classifier.classify(image) # forward(), inference(), get() 등 추론하는 함수 이름 다양함

# print(classification_result)
# STEP 5: Process the classification result. In this case, visualize it. 
top_category = classification_result.classifications[0].categories[0] # 1등만 출력
print(f"{top_category.category_name} ({top_category.score:.2f})")

