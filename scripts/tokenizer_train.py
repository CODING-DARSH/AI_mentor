import os
import sentencepiece as spm

RAW_DATA_DIR = "data/raw"
TOKENIZER_MODEL_PREFIX = "data/tokenizer/mentor_tokenizer"
TOKENIZED_OUTPUT_FILE = "data/tokenizer/tokenized_dataset.txt"

# 1. Combine all raw .txt files into one large corpus
def combine_raw_files():
    combined_path = "combined_corpus.txt"
    with open(combined_path, "w", encoding="utf-8") as outfile:
        for filename in os.listdir(RAW_DATA_DIR):
            if filename.endswith(".txt"):
                filepath = os.path.join(RAW_DATA_DIR, filename)
                with open(filepath, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read() + "\n")
    return combined_path

# 2. Train the SentencePiece tokenizer
def train_tokenizer(corpus_path, vocab_size=6000):
    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=TOKENIZER_MODEL_PREFIX,
        vocab_size=vocab_size,
        character_coverage=1.0,   # for English + general text
        model_type='bpe',         # byte-pair encoding (GPT-like)
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

# 3. Load tokenizer and tokenize dataset
def tokenize_dataset(corpus_path):
    sp = spm.SentencePieceProcessor()
    sp.load(f"{TOKENIZER_MODEL_PREFIX}.model")

    total_tokens = 0
    os.makedirs("tokenizer", exist_ok=True)

    with open(corpus_path, "r", encoding="utf-8") as infile, \
         open(TOKENIZED_OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue
            tokens = sp.encode_as_ids(line)
            total_tokens += len(tokens)
            outfile.write(" ".join(map(str, tokens)) + "\n")

    return total_tokens


if __name__ == "__main__":
    print("Combining raw files...")
    corpus_path = combine_raw_files()

    print(" Training tokenizer (this may take a few minutes)...")
    train_tokenizer(corpus_path)

    print(" Tokenizing dataset...")
    total_tokens = tokenize_dataset(corpus_path)

    print("Tokenization complete!")
    print(f"Total number of tokens in dataset: {total_tokens}")
