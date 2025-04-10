import pandas as pd 
import re
from tqdm import tqdm 
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np 


train_df = pd.read_csv("path")
test_df = pd.read_csv("path")

sentiment_map = {
    "Extremely Negative": "Negative",
    "Negative": "Negative",
    "Neutral": "Neutral",
    "Positive": "Positive",
    "Extremely Positive": "Positive"
}

train_df["Sentiment"] = train_df["Sentiment"].map(sentiment_map)
test_df["Sentiment"] = test_df["Sentiment"].map(sentiment_map)

def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

train_df["clean_text"] = train_df["OriginalTweet"].apply(clean_text)
test_df["clean_text"] = test_df["OriginalTweet"].apply(clean_text)

# Label encoding: map sentiments to numeric values
sentiment_mapping = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

train_df["label"] = train_df["Sentiment"].map(sentiment_mapping)
test_df["label"] = test_df["Sentiment"].map(sentiment_mapping)

X = train_df["clean_text"]
Y = train_df["label"]

X_train , X_val , y_train , y_val = train_test_split(
    X , Y , test_size=0.2 , random_state=40 , stratify=Y
)

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")


class MyData(Dataset):
    def __init__(self , texts , labels, tokenizer , max_len):
        self.encoding = tokenizer(
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors='pt'
        )
        
        self.labels = labels 
    
    def __getitem__(self, index):
        item = {key: val[index] for key , val in self.encoding.items()}
        
        item['labels'] = torch.tensor(self.labels[index])
        
        return item 
    
    def __len__(self):
        return len(self.labels)
    
train_dataset = MyData(X_train.tolist() , y_train.tolist() , tokenizer , 128)
val_dataset = MyData(X_val.tolist() , y_val.tolist() , tokenizer , 128)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)