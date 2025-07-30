import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, concatenate_datasets
import os

tinystories = True
fw = False
fw_train = None
fw_test = None
urdu = False
urdu_train = None
urdu_test = None

def is_urdu(text):
    if not text or not isinstance(text, str):
        return False
    for c in text:
        if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\uFB50' <= c <= '\uFC3F' or '\uFE70' <= c <= '\uFEFF':
            return True
    return False

def clean_urdu_text(text):
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if not is_urdu(text):
        return None
    if len(text) < 10:
        return None
    return text

def clean_urdu_dataset(dataset, field="document_body"):
    return dataset.filter(lambda x: clean_urdu_text(x[field]) is not None).map(lambda x: {field: clean_urdu_text(x[field])})

def prepare_dataset(split, device, batch_size, tokenizer, block_size, hf_token=None, dataset='tinystories'):
    print("Device is: ", device)
    global fw_train, fw_test, urdu_train, urdu_test
    if dataset == 'urdu':
        urdu = True
    else:
        urdu = False
    if urdu_train is None and urdu_test is None and urdu:
        urdu_train = load_dataset('El-chapoo/Urdu-1M-news-text', split='train')
        urdu_train = clean_urdu_dataset(urdu_train, field='News Text')
        urdu_train = urdu_train.train_test_split(test_size=0.01)
        urdu_test = urdu_train['test']
        urdu_train = urdu_train['train']
    if fw_train is None and fw_test is None and not urdu:
        if tinystories:
            fw_train = load_dataset("roneneldan/TinyStories", split="train")
            fw_test = load_dataset("roneneldan/TinyStories", split="validation")
        elif fw:   
            fw_train = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", token=hf_token)
            fw_train = fw_train.train_test_split(test_size=0.01)
    def collate_fn(batch):
        if urdu:
            texts = [item["News Text"] for item in batch]
        else:
        texts = [item["text"] for item in batch]
        input_encodings = tokenizer(texts, padding='max_length', max_length=block_size, truncation=True, return_tensors="pt")
        input_encodings["labels"] = input_encodings["input_ids"].clone()
        input_encodings["labels"][:, :-1] = input_encodings["input_ids"][:, 1:]
        input_encodings["labels"][:, -1] = tokenizer.eos_token_id
        return input_encodings
    data_loader = None
    if urdu:
        if split == 'train':
            data_loader = DataLoader(
                urdu_train,
                batch_size=batch_size,
                collate_fn=collate_fn,
                drop_last=True,
                shuffle=True,
            )
        elif split == 'val':
            data_loader = DataLoader(
                urdu_test,
                batch_size=batch_size,
                collate_fn=collate_fn,
                drop_last=True,
                shuffle=False,
            )
    elif tinystories:
        if split == 'train':
            data_loader = DataLoader(
                fw_train,
                batch_size=batch_size,
                collate_fn=collate_fn,
                drop_last=True,
                shuffle=True,
            )
        elif split == 'val':
            data_loader = DataLoader(
                fw_test,
                batch_size=batch_size,
                collate_fn=collate_fn,
                drop_last=True,
                shuffle=False, 
            )
    elif fw:
        if split == 'train':
            data_loader = DataLoader(
                fw_train['train'],
                batch_size=batch_size,
                collate_fn=collate_fn,
                drop_last=True,
                shuffle=True,
                num_workers=min(4, os.cpu_count()//2) if os.cpu_count() else 2,
                prefetch_factor=2,
                pin_memory=True,
                persistent_workers=True
            )
        elif split == 'val':
            data_loader = DataLoader(
                fw_train['test'],
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=min(4, os.cpu_count()//2) if os.cpu_count() else 2,
                prefetch_factor=2,
                drop_last=True,
                shuffle=False,
                pin_memory=True,
                persistent_workers=True
            )
    return data_loader

# Character-based encoding (legacy from original file)
def get_batch(split, train_data, val_data, block_size, batch_size, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['text'],
        padding='longest',
        truncation=True,
        return_overflowing_tokens=True,
        return_tensors='pt'
    )
