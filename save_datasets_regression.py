import datasets
from transformers import AutoTokenizer
import params

#Tokenizer
tokenizer  = AutoTokenizer.from_pretrained(params.model, model_max_length=512)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Load the dataset
hate = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')

# Split the dataset
train_testvalid = hate['train'].train_test_split(test_size=0.2)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)

# Create a DatasetDict
hate = datasets.DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

# Select the columns and rename the label column
hate_small = hate.select_columns(["text", params.feature])
hate_small = hate_small.rename_column(params.feature, "label")

# Tokenize the text
tokenized_hate_small = hate_small.map(tokenize_function, batched=True)

# Remove the original text column and rename the label column
tokenized_hate_small = tokenized_hate_small.remove_columns(["text"])
tokenized_hate_small = tokenized_hate_small.rename_column("label", "labels")

# Set the format to torch
tokenized_hate_small.set_format("torch")

# Save the tokenized dataset
tokenized_hate_small.save_to_disk('./datasets/tokenized_hate_small_regression')