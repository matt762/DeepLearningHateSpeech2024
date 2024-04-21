import datasets

# Import datase
tokenized_hate_small = datasets.load_from_disk('./datasets/tokenized_hate_small')

# Print
print(tokenized_hate_small['train'].features)