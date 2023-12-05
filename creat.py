import networkx as nx
import math
import faiss

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from datetime import datetime, timedelta
from typing import List
from prompt import *


from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)

world_graph = nx.Graph()

### Response:'''
def creat_world(model):

    for town_area in town_areas.keys():
        world_graph.add_node(town_area)
        world_graph.add_edge(town_area, town_area)
    for town_area in town_areas.keys():
        world_graph.add_edge(town_area, "Phandalin Town Square")
    locations = {}
    agents = {}
    for i in town_people.keys():
        locations[i] = "Phandalin Town Square"
        agents[i] = create_agent(model, i, town_people[i]["age"], town_people[i]["traits"], town_people[i]["status"], town_people[i]["sum"])

    return locations, agents


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
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = HuggingFaceEmbeddings(model_name = model_name, model_kwargs=model_kwargs)
    # Initialize the vectorstore as empty
    embedding_size = 768 # it is a value only for llama-2-7b-chat model
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

def create_agent(model, name, age, traits, status, summary):
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
        summary = summary,
        memory_retriever=create_new_memory_retriever(),
        llm=model,
        memory=tommies_memory,
    )

    print("You are creating a new agent: " + str(name))
    print(tommie.get_summary())

    return tommie