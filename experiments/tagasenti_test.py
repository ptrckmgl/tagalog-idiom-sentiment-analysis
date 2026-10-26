# %%
from transformers import pipeline

# XLM-RoBERTa-large (~355M params), so the first run downloads about 2 GB.
classifier = pipeline(
    "text-classification",
    model="jjjardev/tagasenti_model",
    truncation=True,
    max_length=128
)

# %%
# Example sentences written by Claude; they need native-speaker review.
# The last two use idioms (the model card says idioms are a weak spot).
sample_texts = [
"Agaw-buhay na dinala sa ospital ang lalaki matapos ang aksidente."
]

results = classifier(sample_texts)

for text, prediction in zip(sample_texts, results):

    print("=" * 80)
    print(f"Text: {text}")
    print("-" * 80)
    print(f"{prediction['label']:<10} Score: {prediction['score']:.4f}")
    print()

# %%
# Scores for all three labels instead of only the top one.
results = classifier(sample_texts, top_k=None)

for text, predictions in zip(sample_texts, results):

    print("=" * 80)
    print(f"Text: {text}")
    print("-" * 80)

    for prediction in predictions:
        print(f"{prediction['label']:<10} Score: {prediction['score']:.4f}")

    print()

# %%
