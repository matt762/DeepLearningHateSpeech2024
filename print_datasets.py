import datasets

# Import datase
tokenized_hate_small = datasets.load_from_disk('./datasets/tokenized_hate_small')
#tokenized_imdb = datasets.load_from_disk('./datasets/tokenized_imdb')

# Print
print(tokenized_hate_small['train'].features)
print(tokenized_hate_small['train'][520]['labels'].type())
#print(tokenized_imdb['train'].features)