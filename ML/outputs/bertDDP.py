import os
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

class IMDbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(train_texts, train_labels, test_texts, test_labels, tokenizer, batch_size, world_size, rank):
    train_dataset = IMDbDataset(train_texts, train_labels, tokenizer)
    test_dataset = IMDbDataset(test_texts, test_labels, tokenizer)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    return train_loader, test_loader

def train_epoch(model , data_loader , loss_fn , optimizer , device , scheduler):
    model.train()
    total_loss = 0
    correct_pred = 0

    for batch in data_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids , attention_mask = attention_mask , labels = labels)
        loss = outputs.loss
        logits = outputs.logits
        
        _,preds = torch.max(logits , dim=1)
        correct_pred += torch.sum(preds == labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    scheduler.step()

    return correct_pred.double() / len(data_loader.dataset) , total_loss/len(data_loader)

def eval_model(model , data_loader , loss_fn , device):
    model = model.eval()
    total_loss = 0
    correct_pred = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids = input_ids , attention_mask = attention_mask , labels = labels)
            loss = output.loss
            logits = output.logits

            _,pred = torch.max(logits , dim=1)
            correct_pred += torch.sum(pred == labels)
            total_loss += loss.item()

    return correct_pred.double()/ len(data_loader.dataset) , total_loss / len(data_loader)

def preprocess_data(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank. Necessary for using the torch.distributed.launch utility.')
    args = parser.parse_args()

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    # Load the dataset from CSV
    train_df = pd.read_csv('/home/rafael/Downloads/cnn_dailymail/train.csv')
    test_df = pd.read_csv('/home/rafael/Downloads/cnn_dailymail/test.csv')

    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create data loaders
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    batch_size = 8
    train_loader, test_loader = create_data_loaders(train_texts, train_labels, test_texts, test_labels, tokenizer, batch_size, world_size, rank)

    # Load the pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Define the optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*3)

    # Define the loss function
    loss_fn = CrossEntropyLoss()

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        if rank == 0:
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            local_rank,
            scheduler
        )

        if rank == 0:
            print(f'Train loss {train_loss} accuracy {train_acc}')

        test_acc, test_loss = eval_model(
            model,
            test_loader,
            loss_fn,
            local_rank
        )

        if rank == 0:
            print(f'Test loss {test_loss} accuracy {test_acc}')

    dist.destroy_process_group()

if __name__ == "__main__":
    main()