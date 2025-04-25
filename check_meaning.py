from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# List of captions
captions = [
    "A man is riding a horse.",
    "Someone is on a horse.",
    "The man rides the horse across the field.",
    "A woman is cooking in the kitchen.",
    "a large stadium filled with lots of people",
    "A large stadium packed with a lively crowd watches fireworks light up the sky",
    "Someone is preparing food inside the house."
]
# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and effective

def get_unique_captions(captions: list):
    # Generate embeddings for each caption
    embeddings = model.encode(captions)

    # Define threshold for similarity (between 0 and 1)
    SIMILARITY_THRESHOLD = 0.7

    # Keep unique captions
    unique_captions = []
    removed_captions = []

    # Track which captions we've already considered
    used = set()

    for i, emb1 in enumerate(embeddings):
        if i in used:
            continue

        unique_captions.append(captions[i])
        
        for j in range(i + 1, len(embeddings)):
            if j not in used:
                sim = cosine_similarity([emb1], [embeddings[j]])[0][0]
                if sim >= SIMILARITY_THRESHOLD:
                    removed_captions.append((captions[i], captions[j], sim))
                    used.add(j)

    return unique_captions
    # Output results
    # print("✔ Unique Captions:")
    # for c in unique_captions:
    #     print("-", c)

    # print("\n�� Removed (Similar) Captions:")
    # for base, removed, score in removed_captions:
    #     print(f"- \"{removed}\" is similar to \"{base}\" (Similarity: {score:.2f})")

