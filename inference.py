from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

pre_trained_model = "google/bert_uncased_L-2_H-128_A-2"
tokenizer = AutoTokenizer.from_pretrained(pre_trained_model, model_max_length=512)

# Load the saved model
model = AutoModelForSequenceClassification.from_pretrained("./model")
model.to(device)

text = input("Enter a sentence: ")

inputs = tokenizer(text, return_tensors="pt",  padding = True, truncation = True)
inputs = inputs.to(device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    
predicted_class_id = logits.argmax().item()
print("predicted_class_id: ", predicted_class_id)