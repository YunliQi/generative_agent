from langchain.llms import HuggingFacePipeline
import networkx as nx
from transformers import pipeline
import re
from world_creater import build_world
from .prompt import *


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


memories = {}
for i in town_people.keys():
  memories[i] = []
plans = {}
for i in town_people.keys():
  plans[i] = []

global_time = 8
def generate_description_of_area(x):
  text = "It is "+str(global_time)+":00. The location is "+x+"."
  people = []
  for i in locations.keys():
    if locations[i] == x:
      people.append(i)

compressed_memories_all = {}
for name in town_people.keys():
  compressed_memories_all[name] = []

for name in town_people.keys():
  prompt = "You are {}. {} You just woke up in the town of Phandalin and went out to the Town Square. The following people live in the town: {}. What is your goal for today? Be brief, and use at most 20 words and answer from your perspective.".format(name, town_people[name], ', '.join(list(town_people.keys())) )
  plans[name] = generate(prompt_meta.format(prompt))
  print(name, plans[name])

action_prompts = {}
for location in town_areas.keys():
  people = []
  for i in town_people.keys():
    if locations[i] == location:
      people.append(i)
  
  for name in people:
    prompt = "You are {}. {} You are planning to: {}. You are currently in {} with the following description: {}. It is currently {}:00. The following people are in this area: {}. You can interact with them.".format(name, town_people[name], plans[name], location, town_areas[location], str(global_time), ', '.join(people))
    people_description = []
    for i in people:
      people_description.append(i+': '+town_people[i])
    prompt += ' You know the following about people: ' + '. '.join(people_description)
    memory_text = '. '.join(memories[name][-10:])
    prompt += "What do you do in the next hour? Use at most 10 words to explain."
    action_prompts[name] = prompt

action_results = {}
for name in town_people.keys():
  action_results[name] = generate(prompt_meta.format(action_prompts[name]))
  # Now clean the action
  prompt = """
  Convert the following paragraph to first person past tense:
  "{}"
  """.format(action_results[name])
  action_results[name] = generate(prompt_meta.format(prompt)).replace('"', '').replace("'", '')
  print(name, action_results[name])

action_prompts = {}
for location in town_areas.keys():
  people = []
  for i in town_people.keys():
    if locations[i] == location:
      people.append(i)
  
  for name in people:
    for name_two in people:
      memories[name].append('[Time: {}. Person: {}. Memory: {}]\n'.format(str(global_time), name_two, action_results[name_two]))


def get_rating(x):
  nums = [int(i) for i in re.findall(r'\d+', x)]
  if len(nums)>0:
    return min(nums)
  else:
    return None

memory_ratings = {}
for name in town_people.keys():
  memory_ratings[name] = []
  for i, memory in enumerate(memories[name]):
    prompt = "You are {}. Your plans are: {}. You are currently in {}. It is currently {}:00. You observe the following: {}. Give a rating, between 1 and 5, to how much you care about this.".format(name, plans[name], locations[name], str(global_time), memory)
    res = generate(prompt_meta.format(prompt))
    rating = get_rating(res)
    max_attempts = 2
    current_attempt = 0
    while rating is None and current_attempt<max_attempts:
      rating = get_rating(res)
      current_attempt += 1
    if rating is None:
      rating = 0
    memory_ratings[name].append((res, rating))
  print(memory_ratings[name])

MEMORY_LIMIT = 10
compressed_memories = {}
for name in town_people.keys():
  memories_sorted = sorted(
        memory_ratings[name], 
        key=lambda x: x[1]
    )[::-1]
  relevant_memories = memories_sorted[:MEMORY_LIMIT]
  # print(name, relevant_memories)
  memory_string_to_compress = '.'.join([a[0] for a in relevant_memories])
  prompt = "You are {}. Your plans are: {}. You are currently in {}. It is currently {}:00. You observe the following: {}. Summarize these memories in one sentence.".format(name, plans[name], locations[name], str(global_time), memory_string_to_compress)
  res = generate(prompt_meta.format(prompt))
  compressed_memories[name] = '[Recollection at Time {}:00: {}]'.format(str(global_time), res)
  compressed_memories_all[name].append(compressed_memories[name])

place_ratings = {}

for name in town_people.keys():
  place_ratings[name] = []
  for area in town_areas.keys():
    prompt = "You are {}. Your plans are: {}. You are currently in {}. It is currently {}:00. You have the following memories: {}. Give a rating, between 1 and 5, to how likely you are likely to be at {} the next hour.".format(name, plans[name], locations[name], str(global_time), compressed_memories[name], area)
    res = generate(prompt_meta.format(prompt))
    rating = get_rating(res)
    max_attempts = 2
    current_attempt = 0
    while rating is None and current_attempt<max_attempts:
      rating = get_rating(res)
      current_attempt += 1
    if rating is None:
      rating = 0
    place_ratings[name].append((area, rating, res))
  place_ratings_sorted = sorted(
      place_ratings[name], 
      key=lambda x: x[1]
  )[::-1]
  if place_ratings_sorted[0][0] != locations[name]:
    new_recollection = '[Recollection at Time {}:00: {}]'.format(str(global_time), 'I then moved to {}.'.format(place_ratings_sorted[0][0]))
    compressed_memories_all[name].append(new_recollection)
  locations[name] = place_ratings_sorted[0][0]

for repeats in range(5):
  global_time += 1
  action_prompts = {}
  action_results = {}
  for location in town_areas.keys():
    people = []
    for i in town_people.keys():
      if locations[i] == location:
        people.append(i)
    
    for name in people:
      prompt = "You are {}. Your plans are: {}. You are currently in {} with the following description: {}. Your memories are: {}. It is currently {}:00. The following people are in this area: {}. You can interact with them.".format(name, plans[name], location, town_areas[location], '\n'.join(compressed_memories_all[name][-5:]), str(global_time), ', '.join(people))
      people_description = []
      for i in people:
        people_description.append(i+': '+ agents[i].get_summary(force_refresh=True))
      observation += ' You know the following about people: ' + '. '.join(people_description)
      for i in people:
        agents[i].memory.add_memory(observation)
        _, reaction = agents[i].generate_reaction(observation)
        

  for name in town_people.keys():
    action_results[name] = generate(prompt_meta.format(action_prompts[name]))
    # Now clean the action
    prompt = """
    Convert the following paragraph to first person past tense:
    "{}"
    """.format(action_results[name])
    action_results[name] = generate(prompt_meta.format(prompt)).replace('"', '').replace("'", '')
    print(name, locations[name], global_time, action_results[name])
  action_emojis = {}
  for name in town_people.keys():
    prompt = """
    Convert the following paragraph to a tuple (Action, Object):
    "{}"
    """.format(action_results[name])
    action_emojis[name] = generate(prompt_meta.format(prompt)).replace('"', '').replace("'", '')
    print('    - Emoji Representation:', name, locations[name], global_time, action_emojis[name])
  action_prompts = {}
  for location in town_areas.keys():
    people = []
    for i in town_people.keys():
      if locations[i] == location:
        people.append(i)
    
    for name in people:
      for name_two in people:
        memories[name].append('[Time: {}. Person: {}. Memory: {}]\n'.format(str(global_time), name_two, action_results[name_two]))

  memory_ratings = {}
  for name in town_people.keys():
    memory_ratings[name] = []
    for i, memory in enumerate(memories[name]):
      prompt = "You are {}. Your plans are: {}. Your memories are: {}. You are currently in {}. It is currently {}:00. You observe the following: {}. Give a rating, between 1 and 5, to how much you care about this.".format(name, plans[name], '\n'.join(compressed_memories_all[name][-5:]), locations[name], str(global_time), memory)
      res = generate(prompt_meta.format(prompt))
      rating = get_rating(res)
      max_attempts = 2
      current_attempt = 0
      while rating is None and current_attempt<max_attempts:
        rating = get_rating(res)
        current_attempt += 1
      if rating is None:
        rating = 0
      memory_ratings[name].append((res, rating))

  compressed_memories = {}
  for name in town_people.keys():
    memories_sorted = sorted(
          memory_ratings[name], 
          key=lambda x: x[1]
      )[::-1]
    relevant_memories = memories_sorted[:MEMORY_LIMIT]
    memory_string_to_compress = '.'.join([a[0] for a in relevant_memories])
    prompt = "You are {}. Your plans are: {}. You are currently in {}. It is currently {}:00. You observe the following: {}. Summarize these memories in one sentence.".format(name, plans[name], locations[name], str(global_time), memory_string_to_compress)
    res = generate(prompt_meta.format(prompt))
    compressed_memories[name] = '[Recollection at Time {}:00: {}]'.format(str(global_time), res)
    compressed_memories_all[name].append(compressed_memories[name])

  place_ratings = {}

  for name in town_people.keys():
    place_ratings[name] = []
    for area in town_areas.keys():
      prompt = "You are {}. Your plans are: {}. You are currently in {}. It is currently {}:00. You have the following memories: {}. Give a rating, between 1 and 5, to how likely you are likely to be at {} the next hour.".format(name, plans[name], locations[name], str(global_time), compressed_memories[name], area)
      res = generate(prompt_meta.format(prompt))
      rating = get_rating(res)
      max_attempts = 2
      current_attempt = 0
      while rating is None and current_attempt<max_attempts:
        rating = get_rating(res)
        current_attempt += 1
      if rating is None:
        rating = 0
      place_ratings[name].append((area, rating, res))
    place_ratings_sorted = sorted(
        place_ratings[name], 
        key=lambda x: x[1] )[::-1]
    if place_ratings_sorted[0][0] != locations[name]:
      new_recollection = '[Recollection at Time {}:00: {}]'.format(str(global_time), 'I then moved to {}.'.format(place_ratings_sorted[0][0]))
      compressed_memories_all[name].append(new_recollection)
    locations[name] = place_ratings_sorted[0][0]
