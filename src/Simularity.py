import torch
import pickle
import numpy as np
from model import get_model
from args import get_parser

MODEL_PATH = "data/modelbest.ckpt"
INGR_VOCAB_PATH = "data/ingr_vocab.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K = 5   


# ===============================
# loading vocab
# ===============================
def load_vocab(path):
    vocab = pickle.load(open(path, "rb"))

    if hasattr(vocab, "idx2word"):
        return vocab.idx2word

    if isinstance(vocab, dict) and "idx2word" in vocab:
        return vocab["idx2word"]

    if isinstance(vocab, list):
        return vocab

    raise TypeError("Unsupported vocab format.")


print("Loading ingredient vocabulary.")
ingr_vocab = load_vocab(INGR_VOCAB_PATH)
ingr_vocab_size = len(ingr_vocab)
print(f"Vocab: {ingr_vocab_size} ingredients.")


# ===============================
# Loading Model.
# ===============================
args = get_parser()
model = get_model(args,
                  ingr_vocab_size=ingr_vocab_size,
                  instrs_vocab_size=23231)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")

# ===============================
# 提取 ingredient embedding
# ===============================
emb = model.ingredient_encoder.linear.weight.data.cpu().numpy()
emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)


# ===============================
# simliarity score cauculator
# ===============================
def top_k_similar(word, k=TOP_K):
    word = word.lower().strip()

    if word not in ingr_vocab:
        print(f"cannot find '{word}'. Please check.")
        return None

    idx = ingr_vocab.index(word)
    q = emb_norm[idx]

    sims = np.dot(emb_norm, q)   # cosine similarity
    sorted_idx = np.argsort(-sims)

    results = []
    for i in sorted_idx:
        if ingr_vocab[i] != word:
            results.append((ingr_vocab[i], float(sims[i])))
        if len(results) == k:
            break

    return results


# ===============================
# ===============================
print("Ingredient Similarity Finder")
print("Enter any ingredient name, such as onion, garlic, egg, milk, etc.")
print("Enter exit or quit to exit.")
print("===============================\n")

while True:
    query = input("Enter an ingredient name: ").strip()

    if query.lower() in ["exit", "quit"]:
        print("Thank you for using! 🥰")
        break

    results = top_k_similar(query)

    if results:
        print(f"\nThe {TOP_K} most similar ingredients to '{query}':")
        for i, (w, score) in enumerate(results, 1):
            print(f"{i}. {w:20s} Similarity = {score:.4f}")
        print("\n-------------------------------\n")