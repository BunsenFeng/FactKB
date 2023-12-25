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

### Step 0: Data

Download KG data (kg.zip) and training/evaluation data (data.zip) at [link](https://drive.google.com/drive/folders/1xjXTeBV3ijHE4bfqUyBf_68OsCWgOBwG?usp=sharing). Unzip so that there are two directories: `kg` and `data`.

Six folders in `kg/` represent each knowledge graph, four folders in `data/` represent each dataset (`fact` for FactCollect, the others you could probably tell which is which).

`mkdir models` and `mkdir weights` to create two directories to store pretrained and trained models.

### Step 1: synthetic KG-based factuality pretraining data generation

Generate with the three factuality pretraining strategies:

1) Entity Wiki `gen_wiki.py`

```
usage: gen_wiki.py [-h] [-k KG]

optional arguments:
  -h, --help      show this help message and exit
  -k KG, --kg KG  which knowledge graph to use (in the kg/ folder))
```

2) Evidence Extraction `gen_evidence.py`

```
usage: gen_evidence.py [-h] [-k KG] [-n NUM]

optional arguments:
  -h, --help         show this help message and exit
  -k KG, --kg KG     which knowledge graph to use (in the kg/ folder))
  -n NUM, --num NUM  how many (entity, evidence) to employ (default 100k)
```

3) Knowledge Walk `gen_walk.py`

```
usage: gen_walk.py [-h] [-k KG] [-n NUM] [-l LEN]

optional arguments:
  -h, --help         show this help message and exit
  -k KG, --kg KG     which knowledge graph to use (in the kg/ folder))
  -n NUM, --num NUM  how many paths to generate (default 100k)
  -l LEN, --len LEN  how long each path is (default 5)
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
                        batch size (default 32)
  -e EPOCHS, --epochs EPOCHS
                        number of epochs (default 5)
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate (default 2e-5)
  -w WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        weight decay (default 1e-5)
```

### Step 3: Train it as a Factuality Evaluation Model

Grab the factuality pretrained model, train it on FactCollect alone / a combination of FactCollect and several fact-checking datasets / whatever you see fit.

Note: in `data/fact`, the unfiltered versions are the full of FactCollect; the filtered (with "filtered" in names) are the ones that removed FRANK instances (for experiments in Table 2 and a few other papers).

```
usage: train.py [-h] [-m MODEL] [-c CORPUS] [-t TRAIN_DATASET] [-s TEST_DATASET] [-f FILTER] [-b BATCH_SIZE] [-e EPOCHS] [-l LEARNING_RATE] [-w WEIGHT_DECAY]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        which language model to use as backbone (in huggingface format)
  -c CORPUS, --corpus CORPUS
                        which KG-based synthetic corpus to use for training, none for vanilla LM
  -t TRAIN_DATASET, --train_dataset TRAIN_DATASET
                        which dataset to train on in data/
  -s TEST_DATASET, --test_dataset TEST_DATASET
                        which dataset to test on in data/
  -f FILTER, --filter FILTER
                        which test set filter to use, cnndm/xsum, when testing on FactCollect
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size
  -e EPOCHS, --epochs EPOCHS
                        number of epochs
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate
  -w WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        weight decay
```

It will be automatically saved and evaluated on the test split of your specified test dataset.

### Data Pointers

This work wouldn't be possible without the following datasets and resources:

[FactCollect dataset](https://aclanthology.org/2022.naacl-main.236/)

[FRANK benchmark](https://arxiv.org/abs/2104.13346)

[CovidFact, Healthver, SciFact](https://arxiv.org/abs/2112.01640)

### Reference

```
@inproceedings{feng-etal-2023-factkb,
    title = "{F}act{KB}: Generalizable Factuality Evaluation using Language Models Enhanced with Factual Knowledge",
    author = "Feng, Shangbin  and
      Balachandran, Vidhisha  and
      Bai, Yuyang  and
      Tsvetkov, Yulia",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.59",
    doi = "10.18653/v1/2023.emnlp-main.59",
    pages = "933--952",
    abstract = "Evaluating the factual consistency of automatically generated summaries is essential for the progress and adoption of reliable summarization systems. Despite recent advances, existing factuality evaluation models are not robust, being especially prone to entity and relation errors in new domains. We propose FactKB{---}a simple new approach to factuality evaluation that is generalizable across domains, in particular with respect to entities and relations. FactKB is based on language models pretrained using facts extracted from external knowledge bases. We introduce three types of complementary factuality pretraining objectives based on entity-specific facts, facts extracted from auxiliary knowledge about entities, and facts constructed compositionally through knowledge base walks. The resulting factuality evaluation model achieves state-of-the-art performance on two in-domain news summarization benchmarks as well as on three out-of-domain scientific literature datasets. Further analysis of FactKB shows improved ability to detect erroneous entities and relations in summaries and is robust and easily generalizable across domains.",
}
```
