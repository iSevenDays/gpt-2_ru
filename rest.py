import regex as re
import threading

from os import environ
device = environ.get('DEVICE', 'cpu')
model_path = environ.get('MODEL_PATH')
if model_path is None:
    print("MODEL_PATH env variable is required!")
    exit(0)

import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename=f"logs.log", level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Russian GPT-2", version="0.1",)
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

from evaluate_model import ModelEvaluator

model = ModelEvaluator(model_path, device=device)

class SamplePrompt(BaseModel):
    prompt:str = Field(..., max_length=10000, title='Model prompt')
    length:int = Field(15, ge=1, le=1000, title='Number of tokens generated in each sample')
    temperature:float = Field(0.7, ge=0.01, le=1.0, title="Temperature of neural network.")
    top_k:int = Field(0, ge=0, le=50, title="Number of next possible tokens taked into account.")
    top_p:float = Field(0.9, ge=0.00, le=1.0, title="Total sum of possibilities of next tokens. Only first P tokens with total possibility sum will be included.")
    allow_linebreak:bool = Field(True, title='Allow linebreak in a sample')

class SampleTillTokenPrompt(BaseModel):
    prompt:str = Field(..., max_length=10000, title='Model prompt.')
    stop_token:int = Field(...,  title='Token to stop on.')
    length:int = Field(15, ge=1, le=1000, title='Maximum tokens allowed.')
    temperature:float = Field(0.7, ge=0.01, le=1.0, title="Temperature of neural network.")
    top_k:int = Field(0, ge=0, le=50, title="Number of next possible tokens taked into account.")
    top_p:float = Field(0.9, ge=0.00, le=1.0, title="Total sum of possibilities of next tokens. Only first P tokens with total possibility sum will be included.")
    allow_linebreak:bool = Field(True, title='Allow linebreak in a sample')


lock = threading.RLock()


@app.post("/sample")
def gen_sample(prompt: SamplePrompt):
    with lock:
        model.temperature = prompt.temperature
        model.top_k = prompt.top_k
        model.top_p = prompt.top_p
        sample = model.sample(prompt.prompt, prompt.length, 1)
        return {"input": prompt, "reply": sample[0], "status": "OK"}


@app.post("/sample_stop")
def gen_sample_stop(prompt: SampleTillTokenPrompt):
    with lock:
        model.temperature = prompt.temperature
        model.top_k = prompt.top_k
        model.top_p = prompt.top_p
        return {"input": prompt, "reply": "some sampled data.", "status": "OK"}


@app.get("/health")
def healthcheck():
    return True
