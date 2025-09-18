import spacy

# Load spaCy pre-trained model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Elon Musk is the CEO of Tesla and SpaceX, and he lives in Texas."

doc = nlp(text)

# Extract Named Entities
for ent in doc.ents:
    print(ent.text, "→", ent.label_)

# Example structured table
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Extracted Entities:", entities)
