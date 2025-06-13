import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Tesla stock climbed 2% after Elon Musk indicated the company may launch its robotaxi service in Austin on June 22, also expressing regret about his earlier critical posts about President Trump."

# Process the text
doc = nlp(text)

# Tokenization
print("🔹 Tokens:")
for token in doc:
    print(token.text)

# Stopword Removal + Punctuation Filtering
filtered_tokens = [token for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
print("\n🔹 Filtered Tokens (No stopwords, no punctuation):")
for token in filtered_tokens:
    print(token.text)

# Lemmatization
print("\n🔹 Lemmatized Words:")
for token in filtered_tokens:
    print(f"{token.text} ➡ {token.lemma_}")

