import numpy as np
import datasets
import evaluate
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import params

pre_trained_model = params.model

tokenized_hate_small = datasets.load_from_disk('./datasets/tokenized_hate_small')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
   
#tokenized_hate_small_train = tokenized_hate_small['train'].shuffle(seed=42)
#tokenized_hate_small_test = tokenized_hate_small['test'].shuffle(seed=42)

tokenized_hate_small_train = tokenized_hate_small['train'].shuffle(seed=42).select(range(10000))
tokenized_hate_small_test = tokenized_hate_small['test'].shuffle(seed=42).select(range(10000))

train_dataloader = DataLoader(tokenized_hate_small_train, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(tokenized_hate_small_test, batch_size=8)

model = AutoModelForSequenceClassification.from_pretrained(pre_trained_model, num_labels=params.num_classes)

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
        batch["labels"] = batch["labels"].long()  # Convert labels to LongTensor
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
    batch["labels"] = batch["labels"].long()  # Convert labels to LongTensor
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
    progress_bar.update(1)
metric.compute()
    
# Save the model
model.save_pretrained("./model")