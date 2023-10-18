import os
import json
import argparse
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-k", "--kg", help="which knowledge graph to use (in the \kg folder))")
    argParser.add_argument("-n", "--num", default = 100000, help="how many paths to generate")
    argParser.add_argument("-l", "--len", default = 5, help="how long each path is")

    args = argParser.parse_args()
    kg_name = args.kg
    num_path = int(args.num)
    len_path = int(args.len)

    graph = json.load(open("kg/" + kg_name + "/graph.json", "r"))
    f = open("kg/" + kg_name + "_walk_len" + str(len_path) + "_num" + str(num_path) + "_corpus.txt", "w")

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

    for i in tqdm(range(num_path)):
        while 1:
            try:
                start = np.random.randint(0, len(concepts))
                path = [start]
                for j in range(len_path):
                    choices = graph[str(path[-1])]
                    choice = choices[np.random.randint(0, len(choices))]
                    path += list(choice)
                break
            except:
                continue
        # print(path)
        text = ""
        for i in range(len(path)):
            if i%2 == 0:
                text += concepts[path[i]] + " "
            else:
                text += relations[path[i]] + " "
        text = text.strip() + "\n"
        if kg_name == "atomic":
            text = text.replace("PersonX", "Person").replace("PersonY", "Person").replace("PersonZ", "Person").replace("PersonA", "Person").replace("PersonB", "Person")
        f.write(text)
    f.close()