import sys
import torch
import torch.nn as nn
import dataloader
from tqdm import tqdm
import os
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelWithLMHead, AutoModel, AutoModelForSequenceClassification, BartForSequenceClassification
import numpy as np
import json
import random
import argparse

class FactClassifier(pl.LightningModule):
    def __init__(self, train_dataset_name, test_dataset_name, learning_rate, weight_decay, batch_size, num_epochs, final_name, filter):
        super().__init__()
        self.train_dataset = train_dataset_name
        self.test_dataset = test_dataset_name
        if kg_name == "none":
            self.model = AutoModelForSequenceClassification.from_pretrained(final_name, num_labels=2)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained("./models/" + final_name, num_labels=2)
        self.truth = []
        self.pred = []
        self.id = []
        self.score = []

    def forward(self, input):
        result = self.model(**input["input"], labels=input["label"])
        return result

    def configure_optimizers(self):
        return torch.optim.RAdam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def training_step(self, batch, batch_idx):
        result = self.forward(batch)
        self.log("train_loss", result.loss)
        return result.loss

    def validation_step(self, batch, batch_idx):
        result = self.forward(batch)
        self.truth += batch["label"].tolist()
        self.pred += torch.argmax(result.logits, dim=1).tolist()
        bacc = balanced_accuracy_score(self.truth, self.pred)
        self.log("val_bacc", bacc)
        self.truth = []
        self.pred = []
        self.log("val_loss", result.loss)
        return result.loss

    def test_step(self, batch, batch_idx):
        result = self.forward(batch)
        self.log("test_loss", result.loss)
        self.truth += batch["label"].tolist()
        self.pred += torch.argmax(result.logits, dim=1).tolist()
        self.id += batch["id"]
        self.score += result.logits[:,1].tolist()
        return result.loss

    def on_test_end(self):
        print(self.train_dataset + " " + self.test_dataset + " " + final_name + " filter: " + filter + " " + str(datetime.now()) + "\n") # normal
        # f.write(final_name + "train_cnndm_test_xsum\n")
        # f.write(final_name + "train_xsum_test_cnndm\n")
        if self.test_dataset == "fact":
            print("F1: " + str(f1_score(self.truth, self.pred, average = 'micro')) + "\n") # micro f1 score reported in FactCollect paper
        elif self.test_dataset in ["covidfact", "healthver", "scifact"]:
            print("F1: " + str(f1_score(self.truth, self.pred)) + "\n") # binary F1 score
        print("Accuracy: " + str(accuracy_score(self.truth, self.pred)) + "\n")
        print("Precision: " + str(precision_score(self.truth, self.pred)) + "\n")
        print("Recall: " + str(recall_score(self.truth, self.pred)) + "\n")
        print("Balanced Accuracy: " + str(balanced_accuracy_score(self.truth, self.pred)) + "\n")

        # specific = []
        # for i in range(len(self.truth)):
        #     specific.append({"id": self.id[i], "score": self.score[i]})
        # # with open("logs/" + final_name + " "+ self.test_dataset + ".json", "w") as f:
        # #     json.dump(specific, f)

        # global_acc.append(accuracy_score(self.truth, self.pred))
        # if self.test_dataset == "fact":
        #     global_f1.append(f1_score(self.truth, self.pred, average = 'micro'))
        # elif self.test_dataset in ["covidfact", "healthver", "scifact"]:
        #     global_f1.append(f1_score(self.truth, self.pred))
        # global_bacc.append(balanced_accuracy_score(self.truth, self.pred))

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use as backbone (in huggingface format)")
    argParser.add_argument("-c", "--corpus", help="which KG-based synthetic corpus to use for training, none for vanilla LM")
    argParser.add_argument("-t", "--train_dataset", help="which dataset to train on in data/")
    argParser.add_argument("-s", "--test_dataset", help="which dataset to test on in data/")
    argParser.add_argument("-f", "--filter", default = "none", help="which test set filter to use, cnndm/xsum, when testing on FactCollect")
    argParser.add_argument("-b", "--batch_size", default = 32, help="batch size")
    argParser.add_argument("-e", "--epochs", default = 50, help="number of epochs")
    argParser.add_argument("-l", "--learning_rate", default = 1e-4, help="learning rate")
    argParser.add_argument("-w", "--weight_decay", default = 1e-5, help="weight decay")

    args = argParser.parse_args()
    model_name = args.model
    kg_name = args.corpus
    train_dataset_name = args.train_dataset
    test_dataset_name = args.test_dataset
    filter = args.filter
    batch_size = int(args.batch_size)
    num_epochs = int(args.epochs)
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay)

    if kg_name == "none":
        final_name = model_name
    else:
        final_name = model_name + "-retrained-" + kg_name

    train_loader, dev_loader, _ = dataloader.get_dataloaders(train_dataset_name, batch_size, model_name, filter = filter) # normal
    _, _, test_loader = dataloader.get_dataloaders(test_dataset_name, batch_size, model_name, filter = filter) # normal
    # print(len(train_loader))
    # print(len(dev_loader))
    # print(len(test_loader))

    seed = random.randint(0, 1000000)
    seed_everything(seed)

    model = FactClassifier(train_dataset_name=train_dataset_name, test_dataset_name = test_dataset_name, learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size, num_epochs=num_epochs, final_name=final_name, filter = filter)
    trainer = pl.Trainer(max_epochs=num_epochs, gpus=1, gradient_clip_val=1, precision=16, callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min"), ModelCheckpoint(monitor="val_bacc", mode="max", save_top_k=1, save_last=True, filename="{epoch}-{val_bacc:.2f}")])
    trainer.fit(model, train_loader, dev_loader)

    # saving weights
    model.model.save_pretrained("weights/" + final_name + " " + train_dataset_name + " " + test_dataset_name + " " + str(datetime.now()))

    trainer.test(ckpt_path="best", dataloaders = test_loader)