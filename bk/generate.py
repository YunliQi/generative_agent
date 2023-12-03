from langchain.llms import HuggingFacePipeline
import networkx as nx
from transformers import pipeline
import re
from world_creater import build_world

pipeline = pipeline(model="meta-llama/Llama-2-7b-chat-hf", device_map="auto")
model = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0.7})

world = build_world()

def generate(prompt):
  output = model(prompt, do_sample=True, min_length=10, max_length=len(prompt)+128)
  out = output[0]['generated_text']
  if '### Response:' in out:
    out = out.split('### Response:')[1]
  if '### Instruction:' in out:
    out = out.split('### Instruction:')[0]
  return out