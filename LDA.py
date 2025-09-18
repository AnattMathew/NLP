# Step 1: Install required libraries (uncomment if not installed)
# !pip install gensim nltk spacy pyLDAvis
# !python -m spacy download en_core_web_sm

# Step 2: Import libraries
import nltk
import spacy
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
import os
import webbrowser

# Step 3: Download NLTK stopwords
nltk.download('stopwords')

# Step 4: Sample text corpus
documents = [
    "The economy is showing signs of recovery after the pandemic.",
    "New technological advances in AI are transforming healthcare.",
    "Climate change effects are becoming more evident each year.",
    "The stock market is experiencing volatility due to global events.",
    "Artificial intelligence and machine learning are hot topics in research."
]

# Step 5: Initialize spaCy model and stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

# Step 6: Preprocessing function
def preprocess(text):
    doc = nlp(text.lower())  # Lowercase and tokenize
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]
    return tokens

# Step 7: Preprocess documents
processed_docs = [preprocess(doc) for doc in documents]
print("Processed Documents:\n", processed_docs)

# Step 8: Create dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

print("\nDictionary:\n", dictionary.token2id)
print("\nCorpus (Bag-of-Words representation):\n", corpus)

# Step 9: Train LDA model
num_topics = 2  # Number of topics to extract
lda_model = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=num_topics,
                     random_state=42,
                     update_every=1,
                     chunksize=10,
                     passes=10,
                     alpha='auto',
                     per_word_topics=True)

# Step 10: Print topics
print("\nLDA Topics:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx+1}: {topic}")

# Step 11: Visualize topics, save as HTML, and open automatically
lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

html_file = 'lda_visualization.html'
pyLDAvis.save_html(lda_vis, html_file)

# Automatically open in default browser
abs_path = os.path.abspath(html_file)
webbrowser.open(f"file:///{abs_path.replace(os.sep, '/')}")
print(f"\nLDA visualization saved and opened in your browser: {abs_path}")
