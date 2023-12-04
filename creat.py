import networkx as nx
import math
import faiss

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from datetime import datetime, timedelta
from typing import List


from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)

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

world_graph = nx.Graph()

### Response:'''
def creat_world(model):

    for town_area in town_areas.keys():
        world_graph.add_node(town_area)
        world_graph.add_edge(town_area, town_area)
    for town_area in town_areas.keys():
        world_graph.add_edge(town_area, "Phandalin Town Square")
    locations = {}
    for i in town_people.keys():
        locations[i] = "Phandalin Town Square"
        create_agent(model, name, age, traits, status)

    return locations


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = HuggingFaceEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )

def create_agent(model, name, age, traits, status):
    tommies_memory = GenerativeAgentMemory(
        llm=model,
        memory_retriever=create_new_memory_retriever(),
        verbose=False,
        reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
    )

    tommie = GenerativeAgent(
        name=name,
        age=age,
        traits=traits,  # You can add more persistent traits here
        status=status,  # When connected to a virtual world, we can have the characters update their status
        memory_retriever=create_new_memory_retriever(),
        llm=model,
        memory=tommies_memory,
    )

    print("You are creating a new agent: " + str(name))
    print(tommie.get_summary())

    return tommie