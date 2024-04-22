import datasets
from transformers import AutoModel
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import params

pre_trained_model = params.model

tokenized_hate_small = datasets.load_from_disk('./datasets/tokenized_hate_small_regression')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

tokenized_hate_small_train = tokenized_hate_small['train'].shuffle(seed=42).select(range(1000))
tokenized_hate_small_test = tokenized_hate_small['test'].shuffle(seed=42).select(range(1000))

train_dataloader = DataLoader(tokenized_hate_small_train, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(tokenized_hate_small_test, batch_size=8)

# Load the base model
base_model = AutoModel.from_pretrained(pre_trained_model)

# Replace the classifier with a new one that outputs a single value
base_model.classifier = torch.nn.Linear(base_model.config.hidden_size, 1)

# Apply tanh activation and scale to [-5, 5]
model = torch.nn.Sequential(
    base_model,
    torch.nn.Tanh(),
    torch.nn.Linear(1, 1, bias=False)
)
model[-1].weight.data.fill_(5.0)  # Set the weight to 5.0 to scale the output

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
        labels = batch.pop("labels").float()  # Remove labels from batch and convert to float
        outputs = model(**batch)
        logits = outputs[-1].squeeze(-1)  # Get the output of the model and remove the last dimension
        loss = torch.nn.functional.mse_loss(logits, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    
# Save the model
model.save_pretrained("./model")