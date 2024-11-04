import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler

# Added Imports for BLEU Score
import nltk
nltk.download('punkt_tab')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download NLTK data (only the first time)
nltk.download('punkt')

# 1. Disable tokenizer parallelism to prevent deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Enable anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)

# 4. Paths where the datasets are stored on the cluster
train_dataset_path = "data/jfleg_train"
validation_dataset_path = "data/jfleg_validation"

# 5. Load the datasets from disk (can load from internet- but very slow on clusters: https://huggingface.co/docs/datasets/en/index)
try:
    ds_train = load_from_disk(train_dataset_path)
    ds_validation = load_from_disk(validation_dataset_path)
    print("Datasets loaded successfully.")
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

# 6. Initialize tokenizer with special tokens (sos -> start of sentence, etc.)
tokenizer = AutoTokenizer.from_pretrained("gpt2") # should we create our own? BPE?
special_tokens = {
    "bos_token": "<SOS>",
    "eos_token": "<EOS>",
    "unk_token": "<UNK>",
    "pad_token": "<PAD>"
}
num_added_toks = tokenizer.add_special_tokens(special_tokens)
print(f"Added {num_added_toks} special tokens.")

# 7. Update tokenizer pad, bos, eos, unk tokens
tokenizer.pad_token = "<PAD>"
tokenizer.eos_token = "<EOS>"
tokenizer.bos_token = "<SOS>"
tokenizer.unk_token = "<UNK>"

# 8. Update vocab size after adding special tokens
vocab_size = len(tokenizer)
print(f"Tokenizer vocab size: {vocab_size}")

# 9. Define the GrammarDataset class
class GrammarDataset(Dataset):
    def __init__(self, dataset):
        self.src_data = []
        self.tgt_data = []
        for item in dataset:
            incorrect_sentence = item['sentence']
            for correct_sentence in item['corrections']:
                # Tokenize and encode the sentences
                src_tokens = tokenizer.encode(incorrect_sentence, add_special_tokens=True)
                tgt_tokens = tokenizer.encode(correct_sentence, add_special_tokens=True)

                # Ensure sequences are at least length 2 (<SOS> and <EOS>)
                if len(src_tokens) >= 2 and len(tgt_tokens) >= 2:
                    self.src_data.append(torch.tensor(src_tokens, dtype=torch.long))
                    self.tgt_data.append(torch.tensor(tgt_tokens, dtype=torch.long))

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

# 10. Define the collate function with padding
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    return src_padded, tgt_padded

# 11. Create DataLoaders with optimized settings
batch_size = 128
dataloader_train = DataLoader(
    GrammarDataset(ds_train),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4, 
    pin_memory=True
)
dataloader_validation = DataLoader(
    GrammarDataset(ds_validation),
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

# 12. Define the PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# 13. Define the MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model).to(device)
        self.k_linear = nn.Linear(d_model, d_model).to(device)
        self.v_linear = nn.Linear(d_model, d_model).to(device)
        self.out = nn.Linear(d_model, d_model).to(device)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)

        # Perform linear operation and split into num_heads
        q = self.q_linear(q).view(bs, seq_len_q, self.num_heads, self.d_k)
        k = self.k_linear(k).view(bs, seq_len_k, self.num_heads, self.d_k)
        v = self.v_linear(v).view(bs, seq_len_k, self.num_heads, self.d_k)

        # Transpose to get dimensions bs * num_heads * seq_len * d_k
        q = q.transpose(1, 2)  # [bs, num_heads, seq_len_q, d_k]
        k = k.transpose(1, 2)  # [bs, num_heads, seq_len_k, d_k]
        v = v.transpose(1, 2)  # [bs, num_heads, seq_len_k, d_k]

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [bs, num_heads, seq_len_q, seq_len_k]

        if mask is not None:
            # mask shape: [bs, 1, 1, seq_len_k] or [bs, 1, seq_len_q, seq_len_k]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        scores = torch.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)  # [bs, num_heads, seq_len_q, d_k]
        output = output.transpose(1, 2).contiguous().view(bs, seq_len_q, self.d_model)  # [bs, seq_len_q, d_model]
        output = self.out(output)
        return output

# 14. Define the FeedForward class
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff).to(device)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model).to(device)

    def forward(self, x):
        x = self.dropout(torch.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

# 15. Define the EncoderLayer class
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model)
        self.norm_1 = nn.LayerNorm(d_model).to(device)
        self.norm_2 = nn.LayerNorm(d_model).to(device)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        attn_output = self.attention(x2, x2, x2, mask)
        x = x + self.dropout(attn_output)
        x2 = self.norm_2(x)
        ff_output = self.feed_forward(x2)
        x = x + self.dropout(ff_output)
        return x

# 16. Define the DecoderLayer class
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.attention_1 = MultiHeadAttention(d_model, num_heads)
        self.attention_2 = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model)
        self.norm_1 = nn.LayerNorm(d_model).to(device)
        self.norm_2 = nn.LayerNorm(d_model).to(device)
        self.norm_3 = nn.LayerNorm(d_model).to(device)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, e_outputs, src_mask, tgt_mask):
        x2 = self.norm_1(x)
        attn_output = self.attention_1(x2, x2, x2, tgt_mask)
        x = x + self.dropout(attn_output)
        x2 = self.norm_2(x)
        attn_output = self.attention_2(x2, e_outputs, e_outputs, src_mask)
        x = x + self.dropout(attn_output)
        x2 = self.norm_3(x)
        ff_output = self.feed_forward(x2)
        x = x + self.dropout(ff_output)
        return x

# 17. Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model).to(device)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
# 18. Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model).to(device)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)

    def forward(self, tgt, e_outputs, src_mask, tgt_mask):
        x = self.embed(tgt)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, tgt_mask)
        return x

# 19. Define the Transformer Model class
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, num_heads=8):
        super(Transformer, self).__init__()
        self.device = device
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads)
        self.out = nn.Linear(d_model, tgt_vocab_size).to(device)

    def make_masks(self, src, tgt):
        if src is not None:
            src_mask = (src != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2).to(self.device)
        else:
            src_mask = None

        if tgt is not None:
            tgt_pad_mask = (tgt != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2).to(self.device)
            tgt_len = tgt.size(1)
            subsequent_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=self.device), diagonal=1).bool()
            subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1)
            tgt_mask = tgt_pad_mask & (~subsequent_mask)
        else:
            tgt_mask = None

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.make_masks(src, tgt)
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(tgt, e_outputs, src_mask, tgt_mask)
        output = self.out(d_output)
        return output
    
# 20. Initialize the Transformer model
d_model = 512
num_layers = 6
num_heads = 8
model = Transformer(vocab_size, vocab_size, d_model, num_layers, num_heads).to(device)
print("Transformer model initialized.")

# 21. Initialize model weights
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)
        # nn.Embedding does not have a bias
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

model.apply(initialize_weights)
print("Model weights initialized.")

# 22. Define training parameters
start_lr = 1e-3       # Starting learning rate  # makeshift lr scheduler 😂
end_lr = 1e-5         # Ending learning rate    # makeshift lr scheduler 😂
num_lrs = 25          # Number of learning rates
epochs_per_lr = 50    # Number of epochs per learning rate

# 23. Define loss function
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

epochs_per_lr = 50 # 50    # Number of epochs per learning rate

# 23. Define loss function
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# 24. Define the inference function
def correct_sentence(model, sentence, max_len=50):
    model.eval()
    src_tokens = tokenizer.encode(sentence, add_special_tokens=True)
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
    src_mask, _ = model.make_masks(src_tensor, None)  # Only compute src_mask
    e_outputs = model.encoder(src_tensor, src_mask)
    outputs = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
    for _ in range(max_len):
        _, tgt_mask = model.make_masks(None, outputs)  # Only compute tgt_mask
        d_output = model.decoder(outputs, e_outputs, src_mask, tgt_mask)
        out = model.out(d_output)
        token = out[:, -1, :].argmax(dim=-1)
        outputs = torch.cat([outputs, token.unsqueeze(1)], dim=1)
        if token.item() == tokenizer.eos_token_id:
            break
    predicted_tokens = outputs.squeeze(0).tolist()
    corrected_sentence = tokenizer.decode(predicted_tokens, skip_special_tokens=True)
    return corrected_sentence

# 25. Training Loop with Learning Rate Scheduling
for lr_idx, lr in enumerate(np.linspace(start_lr, end_lr, num_lrs), 1):
    print(f"\n===== Learning Rate {lr_idx}/{num_lrs}: {lr:.6f} =====")

    # Define optimizer with the current learning rate
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    print(f"Optimizer initialized with learning rate: {lr:.6f}")

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    for epoch in range(1, epochs_per_lr + 1):
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        for batch_idx, (src, tgt) in enumerate(dataloader_train, 1):
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast():
                output = model(src, tgt[:, :-1])  # Shifted right
                output = output.reshape(-1, output.size(-1))
                tgt_out = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt_out)

            # Check for NaNs or Infs in loss
            if torch.isnan(loss) or torch.isinf(loss): # if this occurs, lr is probably too high
                print("Loss is NaN or Inf. Skipping this batch.")
                continue  # Skip this batch # lr is probably too high

            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            # Calculate accuracy
            predicted_tokens = output.argmax(dim=1)
            mask = tgt_out != tokenizer.pad_token_id

            # Check if mask sum is zero
            if mask.sum().item() == 0:
                print(f"Batch {batch_idx}: All tokens are padding tokens in tgt_out. Skipping accuracy calculation.")
                continue  # Skip this batch

            correct = (predicted_tokens == tgt_out) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

        # Calculate average loss and accuracy for the epoch
        avg_loss = total_loss / len(dataloader_train)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        print(f"Epoch [{epoch}/{epochs_per_lr}], Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save the model after completing epochs for the current learning rate
    model_save_path = f"grammar_correction_model_lr_{lr:.6f}.pth"
    try:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    # Validation Phase after each learning rate
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total_tokens = 0
    total_bleu_score = 0
    num_sentences = 0
    smoothie = SmoothingFunction().method4  # For BLEU score smoothing # this is used for grammar correction? Source: ChatGPT
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(dataloader_validation, 1):
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            with autocast():
                output = model(src, tgt[:, :-1])
                output = output.reshape(-1, output.size(-1))
                tgt_out = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt_out)
            val_loss += loss.item()

            # Calculate accuracy
            predicted_tokens = output.argmax(dim=1)
            mask = tgt_out != tokenizer.pad_token_id

            if mask.sum().item() == 0:
                print(f"Validation Batch {batch_idx}: All tokens are padding tokens in tgt_out. Skipping accuracy calculation.")
                continue  # Skip this batch

            correct = (predicted_tokens == tgt_out) & mask
            val_correct += correct.sum().item()
            val_total_tokens += mask.sum().item()

            # Compute BLEU score for each sentence in the batch
            batch_size_val = src.size(0)
            seq_len = tgt.size(1) - 1  # Exclude the start token
            predicted_tokens = predicted_tokens.view(batch_size_val, seq_len)
            tgt_out = tgt_out.view(batch_size_val, seq_len)

            for i in range(batch_size_val):
                # Extract non-padded tokens
                mask_i = mask.view(batch_size_val, seq_len)[i]
                if mask_i.sum().item() == 0:
                    continue  # Skip if all tokens are padding

                pred_tokens = predicted_tokens[i][mask_i].tolist()
                tgt_tokens = tgt_out[i][mask_i].tolist()

                # Convert token IDs to sentences
                pred_sentence = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                tgt_sentence = tokenizer.decode(tgt_tokens, skip_special_tokens=True)

                # Tokenize sentences into words
                reference = [nltk.word_tokenize(tgt_sentence)]
                hypothesis = nltk.word_tokenize(pred_sentence)

                # Compute BLEU score with smoothing
                bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
                total_bleu_score += bleu_score
                num_sentences += 1

    # Calculate average validation loss, accuracy, and BLEU score
    avg_val_loss = val_loss / len(dataloader_validation)
    val_accuracy = val_correct / val_total_tokens if val_total_tokens > 0 else 0.0
    avg_bleu_score = total_bleu_score / num_sentences if num_sentences > 0 else 0.0
    print(f"After Learning Rate {lr:.6f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, BLEU Score: {avg_bleu_score:.4f}")

    # Example inference after each learning rate
    test_sentence = "That 's why he is a legend in these days and people repect him ."
    corrected_sentence = correct_sentence(model, test_sentence)
    print(f"Original: {test_sentence}")
    print(f"Corrected: {corrected_sentence}")

print("\nTraining completed.")
