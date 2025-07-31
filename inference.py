import torch
import torch.nn.functional as F

def topk_sampling(model, prompt, device, tokenizer, max_length=50, top_k=50, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    input_ids_len = len(input_ids[0])
    
    generated_tokens = []

    for _ in range(max_length - input_ids_len):
        with torch.no_grad():
            outputs = model(input_ids, inference=True)
            logits = outputs[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            
            probs = probs / temperature
            
            next_token = torch.multinomial(top_k_probs, num_samples=1)
            
            xcol = torch.gather(top_k_indices, -1, next_token)
            input_ids = torch.cat([input_ids, xcol], dim=1)
            
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def save_to_file(step, text):
    
    with open(f'generated_data/generations_{step}.txt', 'w') as f:
        f.write(f"------------------------------------------------Step: {step}--------------------------------------------\n\n")
        f.write(text + "\n\n")
