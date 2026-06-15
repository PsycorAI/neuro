import os
import numpy as np
import multiprocessing
import subprocess
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm  # <--- THE MISSING PIECE

# --- CONFIG ---
TOKENIZER_ID = "meta-llama/Meta-Llama-3-8B"
TOKEN_ID = os.environ.get("HF_TOKEN")  # set HF_TOKEN env var (do NOT hardcode)
DATA_DIR = os.path.expanduser("~/projects/bdh/data/raw")
OUTPUT_DIR = os.path.expanduser("~/projects/bdh/data/tokenized")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CORES = max(1, multiprocessing.cpu_count() - 2) 

def get_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID, token=TOKEN_ID)

def verify_bin_file(bin_path, tokenizer):
    if not os.path.exists(bin_path): return
    print(f"\n[VERIFICATION] Checking {os.path.basename(bin_path)}...")
    with open(bin_path, "rb") as f:
        chunk = f.read(1000 * 4) 
    if chunk:
        tokens = np.frombuffer(chunk, dtype=np.uint32)
        decoded = tokenizer.decode(tokens)
        print(f"--- SAMPLE DECODE ---\n{decoded[:400]}...\n-------------------\n")

def reclaim_space():
    print("\nReclaiming SSD space for Windows...")
    try:
        # This will only work if you have sudo setup or run it manually after
        subprocess.run(["sudo", "fstrim", "-v", "/"], check=True)
        print("Space reclaimed successfully!")
    except Exception:
        print("Note: Run 'sudo fstrim -v /' in terminal to reclaim Windows SSD space.")

def process_folder(folder_path):
    folder_name = os.path.basename(folder_path)
    bin_file = os.path.join(OUTPUT_DIR, f"{folder_name}.bin")
    
    if os.path.exists(bin_file) and os.path.getsize(bin_file) > 0:
        print(f"Skipping {folder_name} (Bin already exists and has data).")
        return

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
             if f.endswith(('.parquet', '.jsonl', '.arrow'))]
    if not files: return

    print(f"\n🚀 STARTING FOLDER: {folder_name}")
    
    extension = "parquet" if files[0].endswith("parquet") else "json"
    ds = load_dataset(extension, data_files=files, split="train")
    tokenizer = get_tokenizer()

    def tokenize_function(examples):
        text_col = next((c for c in ["text", "content", "code"] if c in examples), None)
        return tokenizer(examples[text_col], add_special_tokens=True, truncation=False)

    tokenized_ds = ds.map(
        tokenize_function,
        batched=True,
        num_proc=NUM_CORES, 
        remove_columns=ds.column_names,
        desc=f"Tokenizing {folder_name}"
    )

    print(f"Saving {folder_name} to binary (Chunked)...")
    batch_size = 10000
    total_tokens = 0
    
    with open(bin_file, "wb") as f:
        # This tqdm call was causing your error
        for batch in tqdm(tokenized_ds.iter(batch_size=batch_size), desc="Writing to disk"):
            flat_array = np.array([t for row in batch['input_ids'] for t in row], dtype=np.uint32)
            flat_array.tofile(f)
            total_tokens += len(flat_array)

    print(f"✅ Saved {total_tokens / 1e9:.2f}B tokens to {bin_file}")
    verify_bin_file(bin_file, tokenizer)

if __name__ == "__main__":
    subfolders = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) 
                  if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    for folder in subfolders:
        process_folder(folder)
    
    reclaim_space()