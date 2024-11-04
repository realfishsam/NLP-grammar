# model.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device='cpu'):
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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device='cpu'):
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

        # Linear projections
        q = self.q_linear(q).view(bs, seq_len_q, self.num_heads, self.d_k)
        k = self.k_linear(k).view(bs, seq_len_k, self.num_heads, self.d_k)
        v = self.v_linear(v).view(bs, seq_len_k, self.num_heads, self.d_k)

        # Transpose for attention calculation
        q = q.transpose(1, 2)  # [bs, num_heads, seq_len_q, d_k]
        k = k.transpose(1, 2)  # [bs, num_heads, seq_len_k, d_k]
        v = v.transpose(1, 2)  # [bs, num_heads, seq_len_k, d_k]

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [bs, num_heads, seq_len_q, seq_len_k]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = torch.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)  # [bs, num_heads, seq_len_q, d_k]

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(bs, seq_len_q, self.d_model)  # [bs, seq_len_q, d_model]
        output = self.out(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1, device='cpu'):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff).to(device)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model).to(device)

    def forward(self, x):
        x = self.dropout(torch.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, device='cpu'):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, device)
        self.feed_forward = FeedForward(d_model, device=device)
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

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, device='cpu'):
        super(DecoderLayer, self).__init__()
        self.attention_1 = MultiHeadAttention(d_model, num_heads, device)
        self.attention_2 = MultiHeadAttention(d_model, num_heads, device)
        self.feed_forward = FeedForward(d_model, device=device)
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

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, device='cpu'):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model).to(device)
        self.pe = PositionalEncoding(d_model, device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, device) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, device='cpu'):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model).to(device)
        self.pe = PositionalEncoding(d_model, device=device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, device) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)

    def forward(self, tgt, e_outputs, src_mask, tgt_mask):
        x = self.embed(tgt)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, tgt_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, num_heads=8, device='cpu', pad_token_id=0):
        super(Transformer, self).__init__()
        self.device = device  # Store device as an attribute
        self.pad_token_id = pad_token_id  # Store pad_token_id
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, device)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, device)
        self.out = nn.Linear(d_model, tgt_vocab_size).to(device)

    def make_masks(self, src, tgt):
        if src is not None:
            src_mask = (src != self.pad_token_id).unsqueeze(1).unsqueeze(2).to(self.device)
        else:
            src_mask = None

        if tgt is not None:
            tgt_pad_mask = (tgt != self.pad_token_id).unsqueeze(1).unsqueeze(2).to(self.device)
            tgt_len = tgt.size(1)
            subsequent_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=self.device), diagonal=1).bool()
            subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1)  # Shape: [1, 1, tgt_len, tgt_len]
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
    ```

**Key Changes:**

- **Initialization Parameter:** Added `pad_token_id` as a parameter to the `Transformer` class and stored it as `self.pad_token_id`.
- **`make_masks` Method:** Replaced `tokenizer.pad_token_id` with `self.pad_token_id`, making the method independent of the `tokenizer` instance.

---

## **2. Modify `inference.py` to Pass `pad_token_id` During Model Initialization**

Now, update your `inference.py` (or `infer.py`) script to pass the `pad_token_id` when initializing the `Transformer` model.

**Updated `inference.py`:**

```python
# inference.py

import torch
from transformers import AutoTokenizer
from model import Transformer  # Import the Transformer model from model.py

# 1. Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Initialize Tokenizer with Special Tokens
tokenizer = AutoTokenizer.from_pretrained("gpt2")
special_tokens = {
    "bos_token": "<SOS>",
    "eos_token": "<EOS>",
    "unk_token": "<UNK>",
    "pad_token": "<PAD>"
}
num_added_toks = tokenizer.add_special_tokens(special_tokens)
print(f"Added {num_added_toks} special tokens.")

# Update tokenizer tokens
tokenizer.pad_token = "<PAD>"
tokenizer.eos_token = "<EOS>"
tokenizer.bos_token = "<SOS>"
tokenizer.unk_token = "<UNK>"

# 3. Update Vocab Size
vocab_size = len(tokenizer)
print(f"Tokenizer vocab size: {vocab_size}")

# 4. Initialize the Transformer Model
d_model = 512
num_layers = 6
num_heads = 8
model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    device=device,
    pad_token_id=tokenizer.pad_token_id  # Pass pad_token_id here
).to(device)
print("Transformer model initialized.")

# 5. Load the Trained Model Weights
model_path = "grammar_correction_model_lr_0.001000.pth"  # Replace with your actual model path
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model weights from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 6. Define the Inference Function
def correct_sentence(model, tokenizer, sentence, max_len=50):
    model.eval()
    with torch.no_grad():
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

# 7. User Interface for Inference
if __name__ == "__main__":
    print("\nWelcome to the Grammar Correction Inference Script.")
    print("Type a sentence and press Enter to get the corrected version.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            input_sentence = input("Enter a sentence to correct:\n> ")
            if input_sentence.strip().lower() == 'exit':
                print("Exiting inference script.")
                break
            corrected = correct_sentence(model, tokenizer, input_sentence)
            print(f"Corrected: {corrected}\n")
        except KeyboardInterrupt:
            print("\nExiting inference script.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
