# helpers.py

import re
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model globally so it's reused (avoids reloading every call)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text):
    text = re.sub(r'\S+@\S+', '', text)                # remove emails
    text = re.sub(r'https?://\S+', '', text)           # remove URLs
    return text.strip()

def generate_distractors_from_doc(correct_answer, context_text, top_n=3):
    """
    Generate distractor options from the document text using sentence similarity.
    """
    # Split document into sentences (remove empty & very short ones)
    sentences = list(set(
        [s.strip() for s in context_text.split(".") if 5 < len(s.strip()) < 100]
    ))

    # Compute embedding for the correct answer
    correct_embedding = embedding_model.encode(correct_answer, convert_to_tensor=True)

    # Compute embeddings for all candidate sentences
    sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)

    # Compute cosine similarity scores
    scores = cosine_similarity(
        [correct_embedding.cpu().numpy()],
        sentence_embeddings.cpu().numpy()
    )[0]

    distractors = []
    for idx in scores.argsort()[::-1]:  # highest similarity first
        sent = sentences[idx]

        # Avoid distractors containing the correct answer directly
        if correct_answer.lower() not in sent.lower():
            phrase = " ".join(sent.split()[:6]) + "..."  # short snippet as distractor
            if phrase not in distractors and phrase.lower() != correct_answer.lower():
                distractors.append(phrase)

        if len(distractors) >= top_n:
            break

    return distractors
