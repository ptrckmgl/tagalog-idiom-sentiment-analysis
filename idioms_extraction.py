import os
import ahocorasick
from datasets import load_dataset


# ============================================================
# 1. Define your idioms
# ============================================================

idioms_list = [
"daanan",
"daanin sa tigas ng buto",
"dagok ng kapalaran",
"dagok ng kapalaran",
"di-kabagang",
"di-maabot-sabi",
"di-maabot-tanaw",
"di-mababayaran ang tiyan",
"di-mahapayang gatang",
"di-mahulugang karayom",
"di-makabasag pinggan",
"di-maliparang-uwak",
"di-mataga",
"di-nagtatanaw-tama",
"dibdibin",
"dilang-ginto",
"dilat ang mata",
"dinaan sa bibig",
"dinapuan ng karamdaman",
"dugo ng kanyang dugo",
"dugong mahal",
"dupong",
"dupong",
"durugin ang puso",
]


# ============================================================
# 2. Build the Aho-Corasick automaton
# ============================================================

automaton = ahocorasick.Automaton()

for idiom in idioms_list:
    idiom_lower = idiom.lower()

    # Store the original idiom as the value
    automaton.add_word(
        idiom_lower,
        idiom
    )

automaton.make_automaton()


# ============================================================
# 3. Check whether a match is a complete word/phrase
# ============================================================

def is_word_boundary(text, start, end):
    """
    Returns True if the matched idiom is not part of
    a larger alphanumeric word.

    Example:
        "apa"       -> True
        "apa."      -> True
        "mapayapa"  -> False
    """

    # Character immediately before the match
    if start > 0 and text[start - 1].isalnum():
        return False

    # Character immediately after the match
    if end < len(text) and text[end].isalnum():
        return False

    return True


# ============================================================
# 4. Extract and label idioms from each batch
# ============================================================

def extract_and_label(batch):

    matched_text = []
    found_idioms = []

    for text in batch["text"]:

        text_lower = text.lower()

        matches = []

        # Aho-Corasick searches for all idioms at once
        for end_index, idiom in automaton.iter(text_lower):

            # Convert ending index to starting index
            start_index = end_index - len(idiom) + 1

            # Check that the idiom is not part of a larger word
            if is_word_boundary(
                text_lower,
                start_index,
                end_index + 1
            ):

                # Avoid storing the same idiom multiple times
                # within the same article
                if idiom not in matches:
                    matches.append(idiom)

        # Keep only articles containing at least one idiom
        if matches:
            matched_text.append(text)
            found_idioms.append(matches)

    return {
        "text": matched_text,
        "extracted_idioms": found_idioms
    }


# ============================================================
# 5. Main execution
# ============================================================

if __name__ == "__main__":

    # Load NewsPH
    newsph_dataset = load_dataset(
        r"C:\Users\patri\Downloads\newsph\newsph"
    )

    train_dataset = newsph_dataset["train"]

    print(f"Total articles: {len(train_dataset)}")
    print("Searching for idioms...")

    # Search using multiple CPU processes
    extracted_dataset = train_dataset.map(
        extract_and_label,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=os.cpu_count(),
    )

    # ========================================================
    # 6. Results
    # ========================================================

    print()
    print(f"Extracted {len(extracted_dataset)} matching rows.")

    if len(extracted_dataset) > 0:

        print("\nFirst match:")
        print(extracted_dataset[0])

        print("\nFirst 3 matches:")

        for i in range(min(3, len(extracted_dataset))):
            print()
            print(f"Match {i + 1}:")
            print("Idiom(s):", extracted_dataset[i]["extracted_idioms"])
            print("Text:", extracted_dataset[i]["text"])