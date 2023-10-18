import os
import json
import argparse
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-k", "--kg", help="which knowledge graph to use (in the \kg folder))")
    argParser.add_argument("-n", "--num", default = 100000, help="how many (entity, evidence) to employ")

    args = argParser.parse_args()
    kg_name = args.kg
    num_path = int(args.num)

    graph = json.load(open("kg/" + kg_name + "/graph.json", "r"))

    f = open("kg/" + kg_name + "_evidence_num" + str(num_path) + ".txt", "w")

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

    descriptions = []
    fd = open("kg/" + kg_name + "/entity_desc.txt", "r")
    for line in fd:
        descriptions.append(line.strip())
    fd.close()

    for i in tqdm(range(num_path)):
        while 1:
            try:
                start = np.random.randint(0, len(concepts))
                choices = graph[str(start)]
                choice = choices[np.random.randint(0, len(choices))]
                if len(descriptions[choice[1]]) < 10:
                    continue
            except:
                continue
            break
        text = concepts[start] + " " + relations[choice[0]] + " "
        tail_token = len(concepts[choice[1]].strip().split(" "))
        for k in range(tail_token):
            text += "<mask> "
        text += descriptions[choice[1]]
        text = text.strip() + "\n"
        if kg_name == "atomic":
            text = text.replace("PersonX", "person").replace("PersonY", "person").replace("PersonZ", "person").replace("PersonA", "person").replace("PersonB", "person")
        f.write(text)
    f.close()
