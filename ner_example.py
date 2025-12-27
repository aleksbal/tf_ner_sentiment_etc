from collections import defaultdict

from transformers import AutoTokenizer, TFAutoModelForTokenClassification, pipeline

MODEL_NAME = "dslim/bert-base-NER"

# Load tokenizer and model for NER
model = TFAutoModelForTokenClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create a NER pipeline (TensorFlow backend)
ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    framework="tf",
)

# Example documents
DOCUMENTS = {
    "doc_1": "Alice Johnson met Bob Smith in Paris and later emailed Charlie Brown.",
    "doc_2": "Dr. Emily Zhang spoke with Michael Jordan before the conference.",
    "doc_3": "The report was authored by Priya Kapoor and reviewed by Alex Li.",
}


def find_person_names(documents):
    results = defaultdict(list)
    for doc_id, text in documents.items():
        entities = ner(text)
        for entity in entities:
            if entity.get("entity_group") == "PER":
                results[doc_id].append(entity.get("word"))
    return results


if __name__ == "__main__":
    found_names = find_person_names(DOCUMENTS)

    for doc_id, names in found_names.items():
        names_display = ", ".join(names) if names else "No person names found"
        print(f"{doc_id}: {names_display}")
