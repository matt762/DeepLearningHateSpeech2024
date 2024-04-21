from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import params

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

pre_trained_model = params.model
tokenizer = AutoTokenizer.from_pretrained(pre_trained_model, model_max_length=512)

# Load the saved model
print(params.model_to_load)
model = AutoModelForSequenceClassification.from_pretrained(params.model_to_load)
model.to(device)

text = input("Enter a sentence: ")

inputs = tokenizer(text, return_tensors="pt",  padding = True, truncation = True)
inputs = inputs.to(device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    
predicted_class_id = logits.argmax().item()
print("predicted_class_id: ", predicted_class_id)