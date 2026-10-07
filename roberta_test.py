# %%
from transformers import pipeline

classifier = pipeline(
    "fill-mask",
    model="jcblaise/roberta-tagalog-base"
)

# %%
sample_texts = [
    "Isa kang <mask>",
    "Ayoko <mask>"
]

results = classifier(
    sample_texts,
    top_k=5
)

for text, predictions in zip(sample_texts, results):

    print("=" * 80)
    print(f"Text: {text}")
    print("-" * 80)

    for i, prediction in enumerate(predictions, start=1):
        print(
            f"{i}. "
            f"{prediction['token_str']!r:<20} "
            f"Score: {prediction['score']:.4f}"
        )

    print()

# %%