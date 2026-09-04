import os
from datasets import load_dataset

# 1. Define idioms and helper sets outside the main block
idioms_list = ["walang ulo"]
idioms_set = set(idiom.lower() for idiom in idioms_list)


# 2. Define the transformation function
def extract_and_label(batch):
  matched_text = []
  found_idioms = []

  for text in batch["text"]:
    text_lower = text.lower()
    matches = [idiom for idiom in idioms_set if idiom in text_lower]

    if matches:
      matched_text.append(text)
      found_idioms.append(matches[0])  # Stores the first matched idiom

  return {"text": matched_text, "extracted_idiom": found_idioms}


# 3. Main execution block (Required for Windows multiprocessing)
if __name__ == "__main__":
  # Load dataset
  newsph_dataset = load_dataset(r'C:\Users\patri\Downloads\newsph\newsph')

  # Process dataset using multiprocessing
  extracted_dataset = newsph_dataset["train"].map(
      extract_and_label,
      batched=True,
      remove_columns=newsph_dataset["train"].column_names,
      num_proc=os.cpu_count(),
  )

  # Inspect results
  print(f"Extracted {len(extracted_dataset)} matching rows.")
  if len(extracted_dataset) > 0:
    print("First match:", extracted_dataset[0])