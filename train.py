import numpy as np
import datasets
from collections import Counter
import evaluate
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

mode = 2 # Mode 1 is the manual PyTorch training, mode 2 is the Trainer training
pre_trained_model = "google/bert_uncased_L-2_H-128_A-2"

torch.cuda.empty_cache()

#load a dataset form the HuggingFace Hub
hate = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')

train_testvalid = hate['train'].train_test_split(test_size=0.2)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
hate = datasets.DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

hate_small = hate.select_columns(["text", "status"])
hate_small = hate_small.rename_column("status", "label")

tokenizer = AutoTokenizer.from_pretrained(pre_trained_model, model_max_length=512)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_hate_small = hate_small.map(tokenize_function, batched=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

if mode == 1:
    
    print('-- Manual PyTorch training --')
        
    tokenized_hate_small = tokenized_hate_small.remove_columns(["text"])
    tokenized_hate_small = tokenized_hate_small.rename_column("label", "labels")
    tokenized_hate_small.set_format("torch")
        
    #tokenized_hate_small_train = tokenized_hate_small['train'].shuffle(seed=42)
    #tokenized_hate_small_test = tokenized_hate_small['test'].shuffle(seed=42)
    
    tokenized_hate_small_train = tokenized_hate_small['train'].shuffle(seed=42).select(range(1000))
    tokenized_hate_small_test = tokenized_hate_small['test'].shuffle(seed=42).select(range(1000))

    train_dataloader = DataLoader(tokenized_hate_small_train, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(tokenized_hate_small_test, batch_size=8)

    model = AutoModelForSequenceClassification.from_pretrained(pre_trained_model, num_labels=1)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=1, num_training_steps=num_training_steps
    )

    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            

    metric = evaluate.load("accuracy")
    progress_bar = tqdm(range(len(eval_dataloader)))
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar.update(1)
    metric.compute()
    
else:
    
    print('-- Trainer training --')
    
    model = AutoModelForSequenceClassification.from_pretrained(pre_trained_model, num_labels=1)
    
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_hate_small["train"],
    eval_dataset=tokenized_hate_small["test"],
    compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
## Inference

print('-- Inference --')

text_pos = "The weather is great in Israel today."
text_neg = "Kill all the fucking jews over there"
text_pos_neg = [text_pos, text_neg]

inputs = tokenizer(text_pos, return_tensors="pt",  padding = True, truncation = True)
inputs = inputs.to(device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    
predicted_class_id = logits.argmax().item()
print("predicted_class_id: ", predicted_class_id)

print(model.config.id2label[predicted_class_id])

inputs = tokenizer(text_neg, return_tensors="pt",  padding = True, truncation = True)

with torch.no_grad():
    logits = model(**inputs.to(device)).logits

predicted_class_id = logits.argmax().item()
print("predicted_class_id: ", predicted_class_id)

print(model.config.id2label[predicted_class_id])

inputs = tokenizer(text_pos_neg, return_tensors="pt", padding=True, truncation = True)

with torch.no_grad():
    logits = model(**inputs.to(device)).logits

print("logits: ", logits)

predicted_class_id = logits.argmax(dim=1)
print(predicted_class_id)

for pred in predicted_class_id:
  print(model.config.id2label[pred.item()])