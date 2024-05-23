import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_from_disk
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
from torch.utils.data import random_split

def get_paths(local=True):
    model_paths = {
        'hateval': 'HateBERT_hateval' if local else '/home/maetz/DL/CV_metrics/HateBERT_hateval',
        'abuseval': 'HateBERT_abuseval' if local else '/home/maetz/DL/CV_metrics/HateBERT_abuseval',
        'offenseval': 'HateBERT_offenseval' if local else '/home/maetz/DL/CV_metrics/HateBERT_offenseval'
    }

    dataset_paths = {
        'hateval': 'train_datasets/hateval_dataset' if local else '/home/maetz/DL/CV_metrics/train_datasets/hateval_dataset',
        'abuseval': 'train_datasets/abuseval_dataset' if local else '/home/maetz/DL/CV_metrics/train_datasets/abuseval_dataset',
        'offenseval': 'train_datasets/offenseval_dataset' if local else '/home/maetz/DL/CV_metrics/train_datasets/offenseval_dataset'
    }
    
    return model_paths, dataset_paths

def load_datasets(dataset_paths):
    return {name: load_from_disk(path) for name, path in dataset_paths.items()}

def tokenize_and_format_datasets(datasets, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    print("Tokenizing datasets...")
    tokenized_datasets = {
        name: {
            split: dataset.map(tokenize_function, batched=True)
            for split, dataset in dataset_dict.items()
        }
        for name, dataset_dict in datasets.items()
    }
    
    for dataset_dict in tokenized_datasets.values():
        for split in dataset_dict.values():
            split.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    print("Done tokenizing")
    return tokenized_datasets

def compute_metrics(predictions, labels):
    preds = np.argmax(predictions, axis=-1)
    macro_f1 = f1_score(labels, preds, average='macro')
    pos_f1 = f1_score(labels, preds, pos_label=1)
    return {'macro_f1': macro_f1, 'pos_f1': pos_f1}

def evaluate_model(model, dataset, device):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
    all_predictions, all_labels = [], []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        all_predictions.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return compute_metrics(all_predictions, all_labels)

def main():
    local = False # Switch when evaluating locally or on izar
    num_runs = 5
    model_paths, dataset_paths = get_paths(local)
    
    # Load datasets
    datasets = load_datasets(dataset_paths)
    
    # Load HateBERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('GroNLP/hateBERT')
    
    # Tokenize and format datasets
    tokenized_datasets = tokenize_and_format_datasets(datasets, tokenizer)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results = {}
    
    for dataset_name, model_path in model_paths.items():
        print(f"Evaluating model: {model_path} on dataset: {dataset_name}")
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval().to(device)
        
        dataset = tokenized_datasets[dataset_name]['train']  # Get the train split
        train_size = len(dataset)
        test_size = int(train_size * 0.2)  # Assuming 80-20 train-test split

        macro_f1_scores, pos_f1_scores = [], []
        
        for _ in range(num_runs):
            # Randomly split dataset into train and test sets
            train_dataset, test_dataset = random_split(dataset, [train_size - test_size, test_size])
            metrics = evaluate_model(model, test_dataset, device)
            macro_f1_scores.append(metrics['macro_f1'])
            pos_f1_scores.append(metrics['pos_f1'])
        
        results[dataset_name] = {
            'macro_f1_mean': np.mean(macro_f1_scores),
            'macro_f1_std': np.std(macro_f1_scores),
            'pos_f1_mean': np.mean(pos_f1_scores),
            'pos_f1_std': np.std(pos_f1_scores)
        }

    # Print results
    for key, value in results.items():
        print(f"{key}: Macro F1: {value['macro_f1_mean']} ± {value['macro_f1_std']}, Pos F1: {value['pos_f1_mean']} ± {value['pos_f1_std']}")

if __name__ == "__main__":
    main()
