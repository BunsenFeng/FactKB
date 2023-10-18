## FactKB

[FactKB: Generalizable Factuality Evaluation using Language Models Enhanced with Factual Knowledge](https://arxiv.org/abs/2305.08281)

EMNLP 2023

### Quick Start

A quick demo of FactKB is available at: https://huggingface.co/bunsenfeng/FactKB.

```
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

summary = "An elephant has been hit by a stone at a zoo in western france after it was hit by a tree."
article = "The stone got past the elephant's fence and a ditch separating the animal and visitors, the zoo said in a statement.The girl was taken to hospital and died within a few hours, the zoo added.The zoo statement said the enclosure met international standards and said 'this kind of accident is rare, unpredictable and unusual'.Africa Live: More on this and other storiesThe statement went on (in French) to point out two other recent incidents in the US:Phyllis Lee, Scientific Director of the Amboseli Trust for Elephants, says that targeted throwing of stones and branches by elephants is very unusual.'It can happen when elephants are frustrated or bored. In my opinion, it's unlikely the elephant was directly targeting the girl - but exhibiting frustration. You can't predict what animals in captivity will do.'The moments after the girl was struck at Rabat Zoo on Tuesday were filmed by a bystander and uploaded onto YouTube.The video shows the elephant waving its trunk behind a fence and swerves round to show a stone on the ground.Metres away people are gathered around the girl, holding her head and stroking her leg."
input = [[summary, article]]

tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True)
factkb = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels = 2)

tokens = tokenizer(input, return_tensors="pt", padding="max_length", truncation=True)
result = torch.softmax(factkb(**tokens).logits, dim = 1)

print('The factuality score (0-1, 1 as factual) is: ', float(result[0][1]))
```

### Data

Download KG data (kg.zip) and training/evaluation data (data.zip) at TBD. Unzip so that there are two directories: `kg` and `data`.

Six folders in `kg/' represent each knowledge graph, four folders in `data/` represent each dataset (`fact` for FactCollect, the others you could probably tell which is which).

### Step 1: synthetic KG-based factuality pretraining data generation

Generate with the three factuality pretraining strategies:

1) Entity Wiki `gen_wiki.py`

```
usage: gen_wiki.py [-h] [-k KG]

optional arguments:
  -h, --help      show this help message and exit
  -k KG, --kg KG  which knowledge graph to use (in the \kg folder))
```

2) Evidence Extraction `gen_evidence.py`

```
usage: gen_evidence.py [-h] [-k KG] [-n NUM]

optional arguments:
  -h, --help         show this help message and exit
  -k KG, --kg KG     which knowledge graph to use (in the \kg folder))
  -n NUM, --num NUM  how many (entity, evidence) to employ
```

3) Knowledge Walk `gen_walk.py`

```
usage: gen_walk.py [-h] [-k KG] [-n NUM] [-l LEN]

optional arguments:
  -h, --help         show this help message and exit
  -k KG, --kg KG     which knowledge graph to use (in the \kg folder))
  -n NUM, --num NUM  how many paths to generate
  -l LEN, --len LEN  how long each path is
```

A text file should appear in `kg/` with the strategy you chose and hyperparameters. It is a line-by-line text file with each line representing one created synthetic knowledge instance.

### Step 2: Factuality Pretraining

Train the model with your choice on the factuality pretraining corpora of your choice.

```
usage: pretrain_cont.py [-h] [-m MODEL] [-c CORPUS] [-b BATCH_SIZE] [-e EPOCHS] [-l LEARNING_RATE] [-w WEIGHT_DECAY]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        which language model to use as the backbone (in hugging face format, e.g. roberta-base)
  -c CORPUS, --corpus CORPUS
                        which KG-based synthetic corpus to use for training
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size
  -e EPOCHS, --epochs EPOCHS
                        number of epochs
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate
  -w WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        weight decay
```

### Step 3: Train it as a Factuality Evaluation Model

Grab the factuality pretrained model, train it on FactCollect alone / a combination of FactCollect and several fact-checking datasets / whatever you see fit.

Note: in `data/fact`, the unfiltered versions are the full of FactCollect; the filtered (without "filtered" in names) are the ones that removed FRANK instances (for experiments in Table 2 and a few other papers).

### Data Pointers

This work wouldn't be possible without the following datasets and resources:

[FactCollect dataset](https://aclanthology.org/2022.naacl-main.236/)

[FRANK benchmark](https://arxiv.org/abs/2104.13346)

[CovidFact, Healthver, SciFact](https://arxiv.org/abs/2112.01640)

### Reference

```
@article{feng2023factkb,
  title={Factkb: Generalizable factuality evaluation using language models enhanced with factual knowledge},
  author={Feng, Shangbin and Balachandran, Vidhisha and Bai, Yuyang and Tsvetkov, Yulia},
  journal={arXiv preprint arXiv:2305.08281},
  year={2023}
}
```
