import argparse
from transformers import LineByLineTextDataset, Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForMaskedLM, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling

# model_name = "roberta" # "roberta" or "electra" or "bart" or "deberta" or "albert" or "distilroberta"
# kg_name = "yago_wiki" # {kg type}_{text type}_{amount} # "yago", "wikidata", "cpnet", "atomic", "kgap", "ddb"
# batch_size = 32
# epochs = 5
# resume = False

# if model_name == "roberta":
#     model_path = "roberta-base"
# elif model_name == "electra":
#     model_path = "google/electra-base-discriminator"
# elif model_name == "bart":
#     model_path = "facebook/bart-base"
# elif model_name == "gpt2":
#     model_path = "gpt2"
# elif model_name == "albert":
#     model_path = "albert-base-v2"
# elif model_name == "deberta":
#     model_path = "microsoft/deberta-v3-base"
# elif model_name == "distilroberta":
#     model_path = "distilroberta-base"

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# if model_name == "gpt2":
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # add pad token

# if model_name == "gpt2":
#     model = AutoModelForCausalLM.from_pretrained(model_path)
#     model.resize_token_embeddings(len(tokenizer)) # resize token embeddings
# else:
#     model = AutoModelForMaskedLM.from_pretrained(model_path)

# dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path="kg/" + kg_name + "_corpus.txt",
#     block_size=512,
# )

# if model_name == "gpt2":
#     data_collator = DataCollatorForLanguageModeling(
#         # tokenizer=tokenizer, mlm=True, mlm_probability=0.15
#         tokenizer=tokenizer, mlm=False
#     )
# else:
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer, mlm=True, mlm_probability=0.15
#     )

# training_args = TrainingArguments(
#     output_dir="./models/" + model_name + "-retrained-" + kg_name + "-epoch-" + str(epochs),
#     #output_dir="./models/" + model_name + "-retrained-" + kg_name,
#     overwrite_output_dir=True,
#     learning_rate=2e-5,
#     weight_decay=1e-5,
#     num_train_epochs=epochs,
#     per_device_train_batch_size=batch_size,
#     save_steps=500,
#     save_total_limit=1,
#     lr_scheduler_type='linear',
#     warmup_ratio = 0.06,
#     adam_epsilon = 1e-6,
#     adam_beta2 = 0.98
#     #seed=1
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=dataset
# )

# trainer.train(resume_from_checkpoint = resume)

# trainer.save_model("./models/" + model_name + "-retrained-" + kg_name + "-epoch-" + str(epochs))
# #trainer.save_model("./models/" + model_name + "-retrained-" + kg_name)



if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use as backbone (in huggingface format)")
    argParser.add_argument("-c", "--corpus", help="which KG-based synthetic corpus to use for training")
    argParser.add_argument("-b", "--batch_size", default = 32, help="batch size")
    argParser.add_argument("-e", "--epochs", default = 5, help="number of epochs")
    argParser.add_argument("-l", "--learning_rate", default = 2e-5, help="learning rate")
    argParser.add_argument("-w", "--weight_decay", default = 1e-5, help="weight decay")

    args = argParser.parse_args()
    model_name = args.model
    kg_name = args.corpus
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="kg/" + kg_name,
    block_size=512,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./models/" + model_name + "-retrained-" + kg_name + "-epoch-" + str(epochs),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=1,
        lr_scheduler_type='linear',
        warmup_ratio = 0.06,
        adam_epsilon = 1e-6,
        adam_beta2 = 0.98
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()

    trainer.save_model("./models/" + model_name + "-retrained-" + kg_name + "-epoch-" + str(epochs))