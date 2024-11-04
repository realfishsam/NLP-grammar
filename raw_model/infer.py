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
    vocab_size, vocab_size, d_model, num_layers, num_heads, device=device
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
