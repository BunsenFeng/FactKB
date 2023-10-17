## FactKB

[FACTKB: Generalizable Factuality Evaluation using Language Models Enhanced with Factual Knowledge](https://arxiv.org/abs/2305.08281). EMNLP 2023.

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

### Data Pointers
[FactCollect dataset](https://aclanthology.org/2022.naacl-main.236/)

[FRANK benchmark](https://arxiv.org/abs/2104.13346)

[CovidFact, Healthver, SciFact](https://arxiv.org/abs/2112.01640)
