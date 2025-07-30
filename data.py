# data.py  (Urdu-only)
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import os

def is_urdu(text: str):
    if not text:
        return False
    for c in text:
        if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' \
           or '\uFB50' <= c <= '\uFC3F' or '\uFE70' <= c <= '\uFEFF':
            return True
    return False

def clean_urdu_text(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    return text if (is_urdu(text) and len(text) >= 10) else None

############3 main entry ###################
_URDU_TRAIN = None
_URDU_TEST  = None

def prepare_dataset(split, device, batch_size, tokenizer, block_size,
                    hf_token=None, dataset='urdu'):
    global _URDU_TRAIN, _URDU_TEST

    if _URDU_TRAIN is None or _URDU_TEST is None:
        ds = load_dataset('El-chapoo/Urdu-1M-news-text', split='train')
        ds = ds.filter(lambda x: clean_urdu_text(x['News Text']) is not None)
        ds = ds.map(lambda x: {'text': clean_urdu_text(x['News Text'])})
        ds = ds.train_test_split(test_size=0.01)
        _URDU_TRAIN = ds['train']
        _URDU_TEST  = ds['test']

    data = _URDU_TRAIN if split == 'train' else _URDU_TEST

    def collate(batch):
        texts = [b['text'] for b in batch]
        enc = tokenizer(texts,
                        padding='max_length',
                        max_length=block_size,
                        truncation=True,
                        return_tensors='pt')
        enc['labels'] = enc['input_ids'].clone()
        enc['labels'][:, :-1] = enc['input_ids'][:, 1:]
        enc['labels'][:, -1] = tokenizer.eos_token_id
        return enc

    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        collate_fn=collate,
        drop_last=True,
        num_workers=min(4, os.cpu_count() or 2),
        pin_memory=True,
        persistent_workers=True
    )
