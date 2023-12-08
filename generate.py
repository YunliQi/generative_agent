import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TRANSFORMERS_CACHE'] = '/mnt/iMVR/junde/.cache/huggingface/hub'

from langchain.llms import HuggingFacePipeline
import networkx as nx
from transformers import pipeline
import re
from creat import creat_world
from prompt import *

pipeline = pipeline(model="meta-llama/Llama-2-7b-chat-hf", device_map="auto")
model = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0.7})

locations, agents = creat_world(model)

global_time = 0
for repeats in range(5):
  global_time += 1
  print("In global time", global_time)
  action_prompts = {}
  action_results = {}
  for location in town_areas.keys():
    people = []
    for i in town_people.keys():
      if locations[i] == location:
        people.append(i)
    
    # for name in people:
      # prompt = "You are {}. Your plans are: {}. You are currently in {} with the following description: {}. Your memories are: {}. It is currently {}:00. The following people are in this area: {}. You can interact with them.".format(name, plans[name], location, town_areas[location], '\n'.join(compressed_memories_all[name][-5:]), str(global_time), ', '.join(people))
    people_description = []
    for i in people: 
      if i not in action_results: # initialize action results as the observations
        action_results[i] = agents[i].get_summary(force_refresh=False)
      people_description.append(i+': '+ action_results[i])


    for i in people: # add observation to memory and react
      print("Mind Tree of people: ", i)

      others = [x for x in people if x != i]
      observation = "You are {}.You are currently in {} with the following description: {}. \
      It is currently {}:00. The following people are in this area: {}. You can interact with them.". \
      format(i, location, town_areas[location], str(global_time), ', '.join(others))

      others_des = [x for x in people_description if i+': ' not in x]
      observation += ' You know the following about people: ' + '. '.join(others_des)

      # print("for people", i)
      # print("the observation is:", observation)

      agents[i].memory.add_memory(observation)
      _, reaction = agents[i].generate_reaction(observation)
      action_results[i] = reaction

      print("action result is:")
      print(reaction)