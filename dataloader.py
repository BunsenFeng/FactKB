import torch
import torch.nn as nn
import torch_geometric
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelWithLMHead, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
import json

def get_dataloaders(dataset, batch_size, model, filter = "none"):
    train_dataset = KCDataset(dataset, "train", model, filter)
    dev_dataset = KCDataset(dataset, "dev", model, filter)
    test_dataset = KCDataset(dataset, "test", model, filter)

    # print("train size: ", len(train_dataset))
    # print("dev size: ", len(dev_dataset))
    # print("test size: ", len(test_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 4, collate_fn = train_dataset.pad_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers = 4, collate_fn = dev_dataset.pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 4, collate_fn = test_dataset.pad_collate)
    return train_loader, dev_loader, test_loader

def get_frank_dataloader(batch_size, model, filter = "none"):
    frank_dataset = KCDataset("fact", "frank", model, filter)
    frank_loader = DataLoader(frank_dataset, batch_size=8, shuffle=True, num_workers = 4, collate_fn = frank_dataset.pad_collate)
    return frank_loader

# def pad_collate(batch):
#     texts = []
#     labels = []
#     ids = []
#     for sample in batch:
#         texts.append(sample["input"])
#         labels.append(sample["label"])
#         ids.append(sample["id"])
    

class KCDataset(Dataset):
    def __init__(self, dataset, tdt, model, filter = "none"): # filter: cnndm, xsum
        self.dataset = dataset
        self.tdt = tdt
        self.model = model
        if model == "facebook/bart-base":
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base", padding="max_length", truncation=True)
        elif model == "roberta-base":
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True)
        elif model == "google/electra-base-discriminator":
            self.tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator", padding="max_length", truncation=True)
        elif model == "microsoft/deberta-v3-base":
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", padding="max_length", truncation=True, model_max_length=512)
        elif model == "albert-base-v2":
            self.tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", padding="max_length", truncation=True)
        elif model == "distilroberta-base":
            self.tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", padding="max_length", truncation=True)
        else:
            raise Exception("Invalid model, choose from facebook/bart-base, roberta-base, google/electra-base-discriminator, microsoft/deberta-v3-base, albert-base-v2, distilroberta-base")
        self.data = []
        if tdt == "train":
            temp = open("data/" + dataset + "/" + dataset + "_train.json")
        elif tdt == "dev":
            temp = open("data/" + dataset + "/" + dataset + "_dev.json")
        elif tdt == "test":
            temp = open("data/" + dataset + "/" + dataset + "_test.json")
        elif tdt == "frank":
            temp = open("data/fact/frank_test.json")
        else:
            raise Exception("Invalid tdt")
        # if dataset == "fact":
        if tdt == "frank":
            objs = json.load(temp)
            for obj in objs:
                if obj["domain"] == filter or filter == "none":
                    self.data.append(obj)
        else:
            for line in temp:
                obj = json.loads(line)
                if filter == "none" or not self.tdt == "test" or filter == obj["domain"]:
                    self.data.append(obj)
        random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)

    def pad_collate(self, batch):
        texts = []
        labels = []
        ids = []
        if self.tdt == "frank":
            hashs = []
            model_names = []
        for sample in batch:
            texts.append(sample["input"])
            labels.append(sample["label"])
            ids.append(sample["id"])
            if self.tdt == "frank":
                hashs.append(sample["hash"])
                model_names.append(sample["model_name"])
        if self.tdt == "frank":
            return {
                "input": self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True),
                "label": torch.tensor(labels).long(),
                "id": ids,
                "hash": hashs,
                "model_names": model_names
            }
        else:
            return {
            "input": self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True),
            "label": torch.tensor(labels).long(),
            "id": ids
            }
    
    def __getitem__(self, idx):
        if self.tdt == "frank":
            return {
            "input": [self.data[idx]["summary"], self.data[idx]["article"]],
            "label": 1 if self.data[idx]["label"] == "CORRECT" else 0,
            "id": self.data[idx]["id"],
            "hash": self.data[idx]["hash"],
            "model_name": self.data[idx]["model_name"]
            }
        else:
            return {
            "input": [self.data[idx]["summary"], self.data[idx]["article"]],
            "label": 1 if self.data[idx]["label"] == "CORRECT" else 0,
            "id": self.data[idx]["id"]
            }

# model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
# frank_dataset = KCDataset("train", "roberta")
# frank_loader = DataLoader(frank_dataset, batch_size=8, shuffle=True, num_workers = 4, collate_fn = frank_dataset.pad_collate)
# for batch in frank_loader:
#     result = model(**batch["input"], labels=batch["label"])
#     print(result) # "loss", "logits"
#     break

# model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
# train_loader, dev_loader, test_loader = get_dataloaders("scifact", 8, "roberta")
# for batch in train_loader:
#     result = model(**batch["input"], labels=batch["label"])
#     print(result) # "loss", "logits"
#     break

# frank_loader = get_frank_dataloader(8, "roberta")
# for batch in frank_loader:
#     print(batch)
#     break