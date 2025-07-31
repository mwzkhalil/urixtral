import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional 
import math
import argparse
from dataclasses import dataclass
import os

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

DEFAULT_TOKEN = '...'

def get_args():
    parser = argparse.ArgumentParser(description='SmolUrixtral - Mixtral Inspired Model Training')

    parser.add_argument('--block_size', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--embeddings_dims', type=int, default=384, help='Model embedding dimensions')
    parser.add_argument('--no_of_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--no_of_decoder_layers', type=int, default=4, help='Number of decoder layers')
    
    parser.add_argument('--experts', type=int, default=8, help='Number of MoE experts')
    parser.add_argument('--top_experts', type=int, default=2, help='Number of experts to route to (top-k)')
    parser.add_argument('--noisy_topk', action='store_true', default=False, help='Use noisy top-k routing')
    
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--val_epochs', type=int, default=2, help='Validation frequency (in epochs)')
    parser.add_argument('--max_lr', type=float, default=6e-4, help='Maximum learning rate')
    parser.add_argument('--weight_decay_optim', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--beta_1', type=float, default=0.9, help='Beta1 for optimizer')
    parser.add_argument('--beta_2', type=float, default=0.95, help='Beta2 for optimizer')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon for optimizer')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping value')
    
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--attn_dropout', type=float, default=0.1, help='Attention dropout rate')
    
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--use_checkpointing', action='store_true', default=False, help='Use gradient checkpointing')
    parser.add_argument('--use_liger', action='store_true', default=True, help='Use Liger kernels for optimization')
    parser.add_argument('--use_flash_attention', action='store_true', default=True, help='Use Flash Attention')
    parser.add_argument('--use_compile', action='store_true', default=True, help='Use torch.compile')
    
    parser.add_argument('--vocab_size', type=int, default=16000, help='Vocabulary size')
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face token for accessing gated models like Llama-2')
    
    parser.add_argument('--dataset', type=str, default='tinystories', choices=['tinystories', 'fineweb', 'urdu'], help='Dataset to use for training')
    
    parser.add_argument('--generation_max_length', type=int, default=50, help='Maximum length for text generation')
    parser.add_argument('--generation_top_k', type=int, default=50, help='Top-k value for sampling during generation')
    parser.add_argument('--generation_temperature', type=float, default=1.0, help='Temperature for sampling during generation')
    
    parser.add_argument('--log_interval', type=int, default=100, help='Steps between logging')
    parser.add_argument('--save_interval', type=int, default=2000, help='Steps between saving checkpoints')
    parser.add_argument('--eval_interval', type=int, default=200, help='Steps between evaluation')
    parser.add_argument('--eval_iters', type=int, default=200, help='Number of iterations for evaluation (should be at 1000)')
    parser.add_argument('--eval_check', type=int, default=200, help='Evaluation check steps')
    parser.add_argument('--warmup_iters', type=int, default=1000, help='Number of warmup iterations')
    parser.add_argument('--total_iters', type=int, default=20000, help='Total training iterations')
    parser.add_argument('--lr_decay_iters', type=int, default=20000, help='Learning rate decay iterations')
    parser.add_argument('--wandb_project', type=str, default='Mixtral-DDP-Pretrain-10-billion-tokens', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    
    parser.add_argument('--total_batch_size', type=int, default=524288, help='Total batch size for gradient accumulation')
    parser.add_argument('--micro_batch_size', type=int, default=None, help='Micro batch size (defaults to batch_size)')
    
    parser.add_argument('--use_ddp', action='store_true', default=False, help='Use distributed data parallel')
    
    parser.add_argument('--model_name', type=str, default='smolUrixtral', help='Unique model name for saving and logging')
    
    return parser.parse_args()

def create_tokenizer(hf_token=None, dataset=None):
    import sys
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from tokenizer import Tokenizer
    
    if hf_token is None:
        hf_token = os.environ.get('HF_TOKEN', DEFAULT_TOKEN)
    
    return Tokenizer(hf_token=hf_token, dataset=dataset)

@dataclass
class ModelArgs:
    resume_ckpt: Optional[str] = None
    def __init__(self, args=None, tokenizer_instance=None):
        if args is None:
            args = get_args()
        
        if tokenizer_instance is None:
            hf_token = args.hf_token
            if hf_token is None:
                hf_token = os.environ.get('HF_TOKEN', DEFAULT_TOKEN)
            
            self.tokenizer_instance = create_tokenizer(hf_token, args.dataset)
        else:
            self.tokenizer_instance = tokenizer_instance
        
        self.tokenizer = self.tokenizer_instance.ready_tokenizer()
        self.hf_token = args.hf_token or os.environ.get('HF_TOKEN', DEFAULT_TOKEN)
        
        self.model_name = args.model_name
        self.use_amp = getattr(args, "use_amp", True)
        
        self.block_size = args.block_size
        self.batch_size = args.batch_size
        self.embeddings_dims = args.embeddings_dims
        self.no_of_heads = args.no_of_heads
        self.no_of_decoder_layers = args.no_of_decoder_layers
        # MOE
        self.experts = args.experts
        self.top_experts = args.top_experts
        self.noisy_topk = args.noisy_topk
        
        self.epochs = args.epochs
        self.val_epochs = args.val_epochs
        self.max_lr = args.max_lr
        self.weight_decay_optim = args.weight_decay_optim
        self.beta_1 = args.beta_1
        self.beta_2 = args.beta_2
        self.eps = args.eps
        self.clip = args.clip
        
        self.dropout = args.dropout
        self.attn_dropout = args.attn_dropout
        
        self.device = args.device
        self.use_checkpointing = args.use_checkpointing
        self.use_liger = args.use_liger
        self.use_flash_attention = args.use_flash_attention
        self.use_compile = args.use_compile
        
        self.vocab_size = len(self.tokenizer.get_vocab())  
        self.dataset = args.dataset
        
        self.generation_max_length = args.generation_max_length
        self.generation_top_k = args.generation_top_k
        self.generation_temperature = args.generation_temperature
        
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.eval_interval = args.eval_interval
        self.eval_iters = args.eval_iters
        self.eval_check = args.eval_check
        self.warmup_iters = args.warmup_iters
        self.total_iters = args.total_iters
        self.lr_decay_iters = args.lr_decay_iters
        self.wandb_project = args.wandb_project
        self.wandb_run_name = args.wandb_run_name
        
        self.total_batch_size = args.total_batch_size
        self.micro_batch_size = args.micro_batch_size if args.micro_batch_size else args.batch_size
        self.gradient_accumulation_steps = self.total_batch_size // (self.micro_batch_size * (self.block_size * 1))
        
        self.min_lr = 0.1 * self.max_lr
        self.save_checkpoint_iter = self.save_interval
        
        self.use_ddp = args.use_ddp



def create_model_args(args=None, hf_token=None):
    if args is None:
        args = get_args()
    if hf_token is None:
        hf_token = args.hf_token or os.environ.get('HF_TOKEN', DEFAULT_TOKEN)
    tokenizer_instance = create_tokenizer(hf_token, args.dataset)
    return ModelArgs(args, tokenizer_instance)
try:
    _default_args = get_args()
    _default_model_args = ModelArgs(_default_args)
    
    # Export 
    tokenizer = _default_model_args.tokenizer
    save_checkpoint_iter = _default_model_args.save_checkpoint_iter
    total_iters = _default_model_args.total_iters
    eval_iters = _default_model_args.eval_iters
    eval_check = _default_model_args.eval_check
    warmup_iters = _default_model_args.warmup_iters
    lr_decay_iters = _default_model_args.lr_decay_iters
    total_batch_size = _default_model_args.total_batch_size
    micro_batch_size = _default_model_args.micro_batch_size
    gradient_accumulation_steps = _default_model_args.gradient_accumulation_steps
    min_lr = _default_model_args.min_lr
    
except Exception as e:
    print(f"[WARNING] Could not initialize legacy config variables: {e}")
    print("[INFO] Use create_model_args() function for proper initialization")
    
    # Fallbackk 
    save_checkpoint_iter = 2000
    total_iters = 20000
    eval_iters = 200
    eval_check = 200
    warmup_iters = 1000
    lr_decay_iters = 20000
    total_batch_size = 524288
    micro_batch_size = 16
    gradient_accumulation_steps = 4
    min_lr = 6e-5
