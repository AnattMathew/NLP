from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample dataset
corpus = [
    "I love natural language processing.",
    "Language processing is fun with Python.",
    "Machine learning helps in text processing."
]

# 1. Bag-of-Words (BoW)
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(corpus)
print("BoW Feature Names:", vectorizer.get_feature_names_out())
print("BoW Representation:\n", X_bow.toarray())

# 2. TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)
print("TF-IDF Feature Names:", tfidf.get_feature_names_out())
print("TF-IDF Representation:\n", X_tfidf.toarray())
