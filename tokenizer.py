import os
from transformers import AutoTokenizer, LlamaTokenizer
import sentencepiece as spm
from datasets import load_dataset
import json

class Tokenizer:
    def __init__(self, hf_token=None, dataset=None) -> None:
        if hf_token is None:
            hf_token = os.environ.get('HF_TOKEN')
        if hf_token and hf_token != '...':
            print(f"[INFO] Using HF token for model access")
        else:
            print("[INFO] No HF token provided - using public models only")
            hf_token = None
        if dataset == 'urdu':
            model_dir = 'urdu_tokenizer'
            model_file = os.path.join(model_dir, 'urdu_sentencepiece_tokenizer_v1.model')
            if not os.path.exists(model_file):
                self.train_urdu_tokenizer(model_dir)
            self.tokenizer = LlamaTokenizer(
                vocab_file=model_file,
                unk_token='[UNK]',
                pad_token='[PAD]',
                bos_token='[BOS]',
                eos_token='[EOS]'
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def train_urdu_tokenizer(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        dataset = load_dataset('El-chapoo/Urdu-1M-news-text', split='train')
        from data import clean_urdu_dataset
        dataset = clean_urdu_dataset(dataset, field='News Text')
        text_file = os.path.join(model_dir, 'urdu_corpus.txt')
        if os.path.exists(text_file):
            os.remove(text_file)
        with open(text_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                text = item['News Text'].strip()
                if text:
                    f.write(text + '\n')
        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=os.path.join(model_dir, 'urdu_sentencepiece_tokenizer_v1'),
            vocab_size=32000,
            model_type='bpe',
            character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            input_sentence_size=25000000,
            shuffle_input_sentence=True,
            user_defined_symbols=['[PAD]', '[UNK]', '[BOS]', '[EOS]']
        )
        config = {
            "model_type": "mistral",
            "unk_token": "[UNK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "pad_token": "[PAD]"
        }
        special_tokens_map = {
            "unk_token": "[UNK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "pad_token": "[PAD]"
        }
        with open(os.path.join(model_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        with open(os.path.join(model_dir, 'special_tokens_map.json'), 'w', encoding='utf-8') as f:
            json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)

    def ready_tokenizer(self):
        return self.tokenizer
    def get_vocab_size(self):
        return len(self.tokenizer.get_vocab())
    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)
    def decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)
    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)
