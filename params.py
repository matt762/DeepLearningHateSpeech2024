#model = "google/bert_uncased_L-8_H-512_A-8"
model = "google/bert_uncased_L-2_H-128_A-2" # Pretrained model
model_to_load = "./model_violence_L-12_H-768_A-12" # Model to load for inference
feature = "violence"
num_classes = 5