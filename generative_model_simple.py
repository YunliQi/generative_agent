import networkx as nx
from transformers import pipeline
import re

# prompt = "Write an email about an alpaca that likes flan"
model = pipeline(model="meta-llama/Llama-2-7b-chat-hf", device_map="auto")
# print(model(prompt, max_length=128, do_sample=True))

prompt = '''### Instruction:
Who is the first person on the moon?

### Response:'''
def generate(prompt):
  output = model(prompt, do_sample=True, min_length=10, max_length=len(prompt)+128)
  out = output[0]['generated_text']
  if '### Response:' in out:
    out = out.split('### Response:')[1]
  if '### Instruction:' in out:
    out = out.split('### Instruction:')[0]
  return out

world_graph = nx.Graph()

prompt_meta = '''### Instruction:
{}

### Response:'''

town_areas = ["Barthen's Provisions", "Lionshield Coster", "Stonehill Inn", "Phandalin Town Square"]
town_areas = {"Phandalin Town Square": 'Town square of the town of Phandalin.',
              'Stonehill Inn': "In the center of town stands a large, newly built roadhouse of fieldstone and rough-hewn timbers. The common room is filled with locals nursing mugs of ale or cider, all of them eyeing you with curiosity.",
              "Barthen's Provisions": "Barthen’s is the biggest trading post in Phandalin. Its shelves stock most ordinary goods and supplies, including backpacks, bedrolls, rope, and rations. The place is open from sunup to sundown.",
              "Edermath Orchard": "A tidy little cottage beside an apple orchard.",
              "Lionshield Coster": "Hanging above the front door of this modest trading post is a sign shaped like a wooden shield with a blue lion painted on it. This building is owned by the Lionshields, a merchant company based in the city of Yartar, over a hundred miles to the east. They ship finished goods to Phandalin and other small settlements throughout the region, but this outpost has been hard hit by banditry. The most recent Lionshield caravan due in Phandalin never arrived.",
              "Phandalin Miner's Exchange": "The Miner’s Exchange is a trading post where local miners have their valuable finds weighed, measured, and paid out. In the absence of any local lord or authority, the exchange also serves as an unofficial records office, registering claims to various streams and excavations around the area. There isn’t any real gold rush in Phandalin, but enough wealth is hidden in the nearby streams and valleys to support a good number of independent prospectors. The exchange is a great place to meet people who spend a lot of time out and about in the countryside surrounding Phandalin. The guildmaster is an ambitious and calculating human woman named Halia Thornton.",
              "Alderleaf Farm": "A farm owned by the helpful halfling farmer, Qelline Alderleaf.",
              "Shrine of Luck": "Phandalin's only temple is a small shrine made of stones taken from the nearby ruins. It is dedicated to Tymora, goddess of luck and good fortune.",
              "The Sleeping Giant": "This rundown tap house is a dirty, dangerous watering hole at the end of Phandalin’s main street. It is frequented by Redbrand thugs and operated by a surly female dwarf named Grista.",
              "Townmaster’s Hall": "The townmaster’s hall has sturdy stone walls, a pitched wooden roof, and a bell tower at the back. Posted on a board next to the front door is a notice written in Common. It reads: “REWARD — Orcs near Wyvern Tor! Those of a mind to face the orc menace should inquire within.” The notice bears the town’s seal and an indecipherable signature.",
              "Tresendar Manor": "A ruined manor. The Redbrands’ base in Phandalin is a dungeon complex under Tresendar Manor. Before the manor was ruined, its cellars served as safe storage for food and water in the event that the estate was attacked, while an adjoining crypt provided a resting place for the deceased members of the Tresendar family. The Redbrands have since expanded the cellars to suit their own purposes, adding slave pens, workshops, and barracks."
              }
town_people = {"Toblen Stonehill": "Toblen owns a trading post.", 
               "Daran Edermath": "Daran is a retired adventurer who lives in a tidy little cottage beside an apple orchard. A fit, silver-haired half-elf well over a hundred years old, Daran is a fighter who served as a marshal and herald for many years in the lands of the Dragon Coast, far to the southeast. Upon retiring, he returned to the Neverwinter region, his original home.",  
               "Linene Graywind": "Linene runs a trading post.",  
               "Halia Thornton": "Halia is an ambitious and calculating human woman. She is the guildmaster of Phandalin Miner’s Exchange, a trading post where local miners have their valuable finds weighed, measured, and paid out. In her attempts to establish the Miner's Exchange as the closest thing the town has to a governing authority, she acts as more than a simple merchant.",  
               "Qelline Alderleaf": "Qelline is a wise female halfling of forty-five, and is a pragmatic farmer who seems to know everything that goes on in town. She is a kind host, and is willing to let the characters stay in her hayloft if they don't want to stay at the Stonehill Inn.",  
               "Sister Garaele": "Sister Garaele is an elf cleric of Tymora and a Harper agent.", 
               "Harbin Wester": "Harbin is the townmaster of Phandalin. A pompous, old food. Phandalin has no functioning government, but the townsfolk elect someone to serve as townmaster each year. The townmaster serves as a judge in minor disputes and keeps any records that need to be kept.",
               "Terrill Bloodscar": "Terrill is a human ruffian. He wears a grimy scarlet cloak. He is a member of the Redbrand ruffians. He doesn't like adventurers, and wants to rob and kill them.",
               "Conrad Scarface": "Conrad is a human ruffian. He wears a grimy scarlet cloak. He is a member of the Redbrand ruffians. He doesn't like adventurers, and wants to rob and kill them.",
               "Nellie Starsmith": "Nellie is a human ruffian. She wears a grimy scarlet cloak. She is a member of the Redbrand ruffians. She doesn't like adventurers, and wants to rob and kill them.",
               "Valerie Grinblade": "Valerie is a human ruffian. She wears a grimy scarlet cloak. She is a member of the Redbrand ruffians. She doesn't like adventurers, and wants to rob and kill them.",
               }
for town_area in town_areas.keys():
  world_graph.add_node(town_area)
  world_graph.add_edge(town_area, town_area)
for town_area in town_areas.keys():
  world_graph.add_edge(town_area, "Phandalin Town Square")
locations = {}
for i in town_people.keys():
  locations[i] = "Phandalin Town Square"


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
  for location in town_areas.keys():
    people = []
    for i in town_people.keys():
      if locations[i] == location:
        people.append(i)
    
    for name in people:
      prompt = "You are {}. Your plans are: {}. You are currently in {} with the following description: {}. Your memories are: {}. It is currently {}:00. The following people are in this area: {}. You can interact with them.".format(name, plans[name], location, town_areas[location], '\n'.join(compressed_memories_all[name][-5:]), str(global_time), ', '.join(people))
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
