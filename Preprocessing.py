import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample text corpus
text = "Cats are running and dogs are barking loudly"

# 1. Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

# 2. Stop word removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
print("Stopword Removal:", filtered_tokens)

# 3. Stemming
stemmer = PorterStemmer()
stems = [stemmer.stem(word) for word in filtered_tokens]
print(" Stemming:", stems)

# 4. Lemmatization (using WordNet)
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("Lemmatization:", lemmas)

# 5. Lemmatization using spaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
print("SpaCy Lemmatization:", [token.lemma_ for token in doc if not token.is_stop])
