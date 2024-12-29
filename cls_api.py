# step1: import modules
from transformers import pipeline

# step2: create inference object
# model 부분에 user이름/모델명 이 부분만 수정해주면 됨
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")

from typing import Annotated
from fastapi import FastAPI, Form

app = FastAPI()

@app.post("/inference/")
async def inference(text: Annotated[str, Form()]):
    # step3: X

    # step4: inference
    result = classifier(text)

    # step5: post processing
    print(result)
    return {"result": result}