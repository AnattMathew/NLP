import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
import string

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Sample text
text = "Tesla stock climbed 2% after Elon Musk indicated the company may launch its robotaxi service in Austin on June 22, also expressing regret about his earlier critical posts about President Trump."

# Tokenize
tokens = word_tokenize(text)
print("🔹 Tokens:")
print(tokens)

# Remove stopwords and punctuation
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
print("\n🔹 Filtered Tokens (No stopwords, no punctuation):")
print(filtered_tokens)

# POS tagging for better lemmatization
pos_tags = pos_tag(filtered_tokens)

# Function to map POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

# Lemmatization
print("\n🔹 Lemmatized Words:")
for word, tag in pos_tags:
    lemma = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
    print(f"{word} ➡ {lemma}")

# Stemming
print("\n🔹 Stemmed Words:")
for word in filtered_tokens:
    stemmed_word = stemmer.stem(word)
    print(f"{word} ➡ {stemmed_word}")
