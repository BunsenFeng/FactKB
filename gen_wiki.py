import os
import json
import argparse
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-k", "--kg", help="which knowledge graph to use (in the \kg folder))")

    args = argParser.parse_args()
    kg_name = args.kg

    graph = json.load(open("kg/" + kg_name + "/graph.json", "r"))

    f = open("kg/" + kg_name + "_wiki.txt", "w")

    concepts = []
    fc = open("kg/" + kg_name + "/entity.txt", "r")
    for line in fc:
        concepts.append(line.strip())
    fc.close()

    relations = []
    fr = open("kg/" + kg_name + "/relation.txt", "r")
    for line in fr:
        relations.append(line.strip())
    fr.close()

    for i in tqdm(range(len(concepts))):
        try:
            wiki = graph[str(i)]
        except:
            continue
        text = ""
        for choice in wiki:
            text += (concepts[i] + " " + relations[choice[0]] + " " + concepts[choice[1]] + ", ")
        if kg_name == "atomic":
            text = text.replace("PersonX", "person").replace("PersonY", "person").replace("PersonZ", "person").replace("PersonA", "person").replace("PersonB", "person")
        f.write(text[:-2] + ".\n")
    
    f.close()