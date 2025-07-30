# trainer.py
import os
import math
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from tokenizer import Tokenizer
from config import create_model_args
from model import Mixtral
from data import prepare_dataset
from inference import topk_sampling, save_to_file


# -----------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------
def _save_snapshot(model, optimizer, scheduler, epoch, step, model_name="urixtral"):
    snapshot = {
        "MODEL_STATE": model.state_dict(),
        "OPTIMIZER_STATE": optimizer.state_dict(),
        "SCHEDULER_STATE": scheduler.state_dict() if scheduler else None,
        "EPOCHS_RUN": epoch,
        "STEP_RUN": step
    }
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(snapshot, f"checkpoints/snapshot_{step}.pt")
    print(f"Snapshot saved at step {step}")


def _load_snapshot(snapshot_path, model, optimizer, scheduler):
    snapshot = torch.load(snapshot_path)
    model.load_state_dict(snapshot["MODEL_STATE"])
    optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
    if scheduler and snapshot.get("SCHEDULER_STATE"):
        scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
    epoch = snapshot.get("EPOCHS_RUN", 0)
    step = snapshot.get("STEP_RUN", 0)
    print(f"Resumed from snapshot at epoch {epoch}, step {step}")
    return epoch, step


# -----------------------------------------------------------
# Custom LR scheduler (cosine with warmup)
# -----------------------------------------------------------
class CustomLRScheduler:
    def __init__(self, optimizer, warmup_iters, lr_decay_iters, min_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.it = 0
        self._last_lr = [max_lr]

    def step(self):
        lr = self._get_lr()
        self._last_lr = [lr]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.it += 1

    def get_last_lr(self):
        return self._last_lr

    def _get_lr(self):
        if self.it < self.warmup_iters:
            return self.max_lr * (self.it + 1) / (self.warmup_iters + 1)
        if self.it > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (self.it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def state_dict(self):
        return {
            "warmup_iters": self.warmup_iters,
            "lr_decay_iters": self.lr_decay_iters,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "it": self.it
        }

    def load_state_dict(self, state_dict):
        self.warmup_iters = state_dict["warmup_iters"]
        self.lr_decay_iters = state_dict["lr_decay_iters"]
        self.min_lr = state_dict["min_lr"]
        self.max_lr = state_dict["max_lr"]
        self.it = state_dict["it"]


# -----------------------------------------------------------
# Main training routine
# -----------------------------------------------------------
# def train(model_args=None):
#     if model_args is None:
#         model_args = create_model_args()

#     tokenizer = model_args.tokenizer
#     wandb.init(project=model_args.wandb_project)
#     print("WandB initialized")

#     model = Mixtral(
#         attn_dropout=model_args.attn_dropout,
#         embeddings_dims=model_args.embeddings_dims,
#         no_of_heads=model_args.no_of_heads,
#         block_size=model_args.block_size,
#         dropout=model_args.dropout,
#         no_of_decoder_layers=model_args.no_of_decoder_layers,
#         vocab_size=model_args.vocab_size,
#         device=model_args.device,
#         tokenizer=tokenizer
#     ).to(model_args.device)

#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=model_args.max_lr,
#         betas=(model_args.beta_1, model_args.beta_2),
#         weight_decay=model_args.weight_decay_optim,
#         eps=model_args.eps,
#     )

#     scheduler = CustomLRScheduler(
#         optimizer,
#         model_args.warmup_iters,
#         model_args.lr_decay_iters,
#         model_args.min_lr,
#         model_args.max_lr
#     )

#     train_loader = prepare_dataset('train', model_args.device, model_args.batch_size,
#                                    tokenizer, model_args.block_size, model_args.hf_token,
#                                    dataset=model_args.dataset)
#     val_loader = prepare_dataset('val', model_args.device, model_args.batch_size,
#                                  tokenizer, model_args.block_size, model_args.hf_token,
#                                  dataset=model_args.dataset)

#     train_iter = iter(train_loader)
#     val_iter   = iter(val_loader)
#     token_count = 0
#     start_epoch, start_step = 0, 0

#     # Optional resume
#     if model_args.resume_ckpt and os.path.isfile(model_args.resume_ckpt):
#         start_epoch, start_step = _load_snapshot(model_args.resume_ckpt, model, optimizer, scheduler)
#         scheduler.it = start_step   # sync scheduler iteration

#     pbar = tqdm(range(start_step, model_args.total_iters), initial=start_step, total=model_args.total_iters)
#     for step in pbar:
#         model.train()
#         optimizer.zero_grad()
#         accumulated_loss = 0.0

#         for _ in range(model_args.gradient_accumulation_steps):
#             try:
#                 batch = next(train_iter)
#             except StopIteration:
#                 train_iter = iter(train_loader)
#                 batch = next(train_iter)

#             input_ids = batch['input_ids'].to(model_args.device, non_blocking=True)
#             labels    = batch['labels'].to(model_args.device, non_blocking=True)

#             # Forward
#             output = model(input_ids, labels)  # may return loss or logits

#             # Handle both cases
#             if isinstance(output, tuple):
#                 loss, logits = output
#             elif torch.is_tensor(output) and output.dim() == 0:
#                 loss = output
#                 logits = None  # not needed for training
#             else:
#                 logits = output
#                 loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
#                                        labels.view(-1),
#                                        ignore_index=tokenizer.pad_token_id)

#             loss = loss / model_args.gradient_accumulation_steps
#             loss.backward()
#             accumulated_loss += loss.item()
#             token_count += input_ids.numel()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), model_args.clip)
#         optimizer.step()
#         scheduler.step()

#         # Logging
#         pbar.set_postfix({"loss": accumulated_loss, "tok": token_count})
#         wandb.log({
#             "Train_Loss": accumulated_loss,
#             "Learning Rate": scheduler.get_last_lr()[0],
#             "Train_Perplexity": math.exp(accumulated_loss),
#             "Tokens": token_count,
#             "Step": step,
#         })

#         # Validation
#         if step % model_args.eval_iters == 0 or step == model_args.total_iters - 1:
#             model.eval()
#             with torch.no_grad():
#                 val_losses = []
#                 for val_batch in val_loader:
#                     input_ids = val_batch['input_ids'].to(model_args.device, non_blocking=True)
#                     labels    = val_batch['labels'].to(model_args.device, non_blocking=True)

#                     output = model(input_ids, labels)
#                     if isinstance(output, tuple):
#                         val_loss, _ = output
#                     elif torch.is_tensor(output) and output.dim() == 0:
#                         val_loss = output
#                     else:
#                         logits = output
#                         val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
#                                                    labels.view(-1),
#                                                    ignore_index=tokenizer.pad_token_id)
#                     val_losses.append(val_loss.item())

#                 val_loss = torch.tensor(val_losses).mean()
#                 wandb.log({
#                     "Val_Loss": val_loss.item(),
#                     "Val_Perplexity": math.exp(val_loss.item()),
#                     "Step": step
#                 })

#         # Snapshot
#         if step % 1000 == 0 and step != 0:
#             _save_snapshot(model, optimizer, scheduler, 0, step, model_args.model_name)

#         # Sample generation
#         if step % 200 == 0:
#             prompt = "ایک دن ایک بادشاہ تھا"
#             generated = topk_sampling(
#     model=model,
#     prompt=prompt,
#     tokenizer=tokenizer,
#     max_length=50,
#     temperature=1.0,
#     top_k=50,
#     device=args.device
# )


#             print(f"[Step {step}] Generated: {generated}")
#             save_to_file(step, generated)

#     # HuggingFace export
#     output_dir = os.path.join("hf_model", model_args.model_name)
#     os.makedirs(output_dir, exist_ok=True)
#     model.save_pretrained(output_dir)
#     tokenizer.tokenizer.save_pretrained(output_dir)
#     print(f"Model saved in HuggingFace format at {output_dir}")
#     wandb.finish()

def train(model_args=None):
    if model_args is None:
        model_args = create_model_args()

    tokenizer = model_args.tokenizer
    wandb.init(project=model_args.wandb_project)
    print("WandB initialized")

    model = Mixtral(
        attn_dropout=model_args.attn_dropout,
        embeddings_dims=model_args.embeddings_dims,
        no_of_heads=model_args.no_of_heads,
        block_size=model_args.block_size,
        dropout=model_args.dropout,
        no_of_decoder_layers=model_args.no_of_decoder_layers,
        vocab_size=model_args.vocab_size,
        device=model_args.device,
        tokenizer=tokenizer
    ).to(model_args.device)

    # ✅ Use torch.compile if requested
    if model_args.use_compile:
        model = torch.compile(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=model_args.max_lr,
        betas=(model_args.beta_1, model_args.beta_2),
        weight_decay=model_args.weight_decay_optim,
        eps=model_args.eps,
    )

    scheduler = CustomLRScheduler(
        optimizer,
        model_args.warmup_iters,
        model_args.lr_decay_iters,
        model_args.min_lr,
        model_args.max_lr
    )

    scaler = torch.cuda.amp.GradScaler(enabled=model_args.use_amp)

    train_loader = prepare_dataset('train', model_args.device, model_args.batch_size,
                                   tokenizer, model_args.block_size, model_args.hf_token,
                                   dataset=model_args.dataset)
    val_loader = prepare_dataset('val', model_args.device, model_args.batch_size,
                                 tokenizer, model_args.block_size, model_args.hf_token,
                                 dataset=model_args.dataset)

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    token_count = 0
    start_epoch, start_step = 0, 0

    if model_args.resume_ckpt and os.path.isfile(model_args.resume_ckpt):
        start_epoch, start_step = _load_snapshot(model_args.resume_ckpt, model, optimizer, scheduler)
        scheduler.it = start_step

    pbar = tqdm(range(start_step, model_args.total_iters), initial=start_step, total=model_args.total_iters)
    for step in pbar:
        model.train()
        optimizer.zero_grad()
        accumulated_loss = 0.0

        for _ in range(model_args.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch['input_ids'].to(model_args.device, non_blocking=True)
            labels = batch['labels'].to(model_args.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=model_args.use_amp):
                output = model(input_ids, labels)

                if isinstance(output, tuple):
                    loss, logits = output
                elif torch.is_tensor(output) and output.dim() == 0:
                    loss = output
                    logits = None
                else:
                    logits = output
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                           labels.view(-1),
                                           ignore_index=tokenizer.pad_token_id)

                loss = loss / model_args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            token_count += input_ids.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), model_args.clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        pbar.set_postfix({"loss": accumulated_loss, "tok": token_count})
        wandb.log({
            "Train_Loss": accumulated_loss,
            "Learning Rate": scheduler.get_last_lr()[0],
            "Train_Perplexity": math.exp(accumulated_loss),
            "Tokens": token_count,
            "Step": step,
        })

        if step % model_args.eval_iters == 0 or step == model_args.total_iters - 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for val_batch in val_loader:
                    input_ids = val_batch['input_ids'].to(model_args.device, non_blocking=True)
                    labels = val_batch['labels'].to(model_args.device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=model_args.use_amp):
                        output = model(input_ids, labels)
                        if isinstance(output, tuple):
                            val_loss, _ = output
                        elif torch.is_tensor(output) and output.dim() == 0:
                            val_loss = output
                        else:
                            logits = output
                            val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                                       labels.view(-1),
                                                       ignore_index=tokenizer.pad_token_id)
                    val_losses.append(val_loss.item())

                val_loss = torch.tensor(val_losses).mean()
                wandb.log({
                    "Val_Loss": val_loss.item(),
                    "Val_Perplexity": math.exp(val_loss.item()),
                    "Step": step
                })

        if step % 1000 == 0 and step != 0:
            _save_snapshot(model, optimizer, scheduler, 0, step, getattr(model_args, "model_name", "default"))

        if step % 200 == 0:
            prompt = "ایک دن ایک بادشاہ تھا"
            generated = topk_sampling(
                model=model,
                prompt=prompt,
                tokenizer=tokenizer,
                max_length=50,
                temperature=1.0,
                top_k=50,
                device=model_args.device
            )
            print(f"[Step {step}] Generated: {generated}")
            save_to_file(step, generated)

    output_dir = os.path.join("hf_model", getattr(model_args, "model_name", "default"))
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.tokenizer.save_pretrained(output_dir)
    print(f"Model saved in HuggingFace format at {output_dir}")
    wandb.finish()


# -----------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.set_float32_matmul_precision('high')

    args = create_model_args()
    train(args)