import datasets
from transformers import AutoTokenizer
import params

#Tokenizer
tokenizer  = AutoTokenizer.from_pretrained(params.model, model_max_length=512)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


## Hate dataset
hate = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')

train_testvalid = hate['train'].train_test_split(test_size=0.2)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
hate = datasets.DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})
hate_small = hate.select_columns(["text", params.feature])
hate_small = hate_small.rename_column(params.feature, "label")

new_features = hate_small['train'].features.copy()
new_features["label"] = datasets.ClassLabel(num_classes=params.num_classes)
hate_small['train'] = hate_small['train'].cast(new_features)

tokenized_hate_small = hate_small.map(tokenize_function, batched=True)
tokenized_hate_small = tokenized_hate_small.remove_columns(["text"])
tokenized_hate_small = tokenized_hate_small.rename_column("label", "labels")

tokenized_hate_small.set_format("torch")

# Save tokenized datasets
tokenized_hate_small.save_to_disk('./datasets/tokenized_hate_small')

""" #IMBD
imbd = datasets.load_dataset('imdb')
tokenized_imdb = imbd.map(tokenize_function, batched=True)
tokenized_imdb = tokenized_imdb.remove_columns(["text"])
tokenized_imdb = tokenized_imdb.rename_column("label", "labels")
tokenized_imdb.set_format("torch")
tokenized_imdb.save_to_disk('./datasets/tokenized_imdb') """

""" #Yelp
yelp = dataset = datasets.load_dataset("yelp_review_full")
tokenized_yelp = yelp.map(tokenize_function, batched=True)
tokenized_yelp = tokenized_yelp.remove_columns(["text"])
tokenized_yelp = tokenized_yelp.rename_column("label", "labels")
tokenized_yelp.set_format("torch")
tokenized_yelp.save_to_disk('./datasets/tokenized_yelp') """