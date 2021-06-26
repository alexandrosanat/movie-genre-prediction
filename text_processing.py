import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
import joblib


# Check if the tokenizer exists and download if not
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def clean_text(text: str):
    """
    Applies basic pre-processing to clean text
    """
    # replace underscores with space
    text = text.replace("_", " ")
    # remove backslash-apostrophe
    text = re.sub("\'", "", text)
    # remove everything except alphabets
    text = re.sub("[^a-zA-Z]", " ", text)
    # remove whitespaces
    text = ' '.join(text.split())
    # convert text to lowercase
    text = text.lower()

    return text


def tokenise_text(text: str):
    """
    Converts a string to a list of words.
    """
    # Convert a sentence to words
    toke_list = word_tokenize(text)
    return toke_list


def lemmatise_text(token_list: list):
    """
    Converts a list of words to their lemmas.
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_list = []
    # Convert each word to each lemma and append to a new list
    for word in token_list:
        lemma_list.append(wordnet_lemmatizer.lemmatize(word))
    return lemma_list


def filter_stopword(lemma_list: list):
    """
    Removes common stopwords for a list of words.
    """
    filtered_sentence = []  # New list to store words
    nltk_stop_words = set(stopwords.words("english"))
    for w in lemma_list:
        # Loop through all words and filter out stopwords
        if w not in nltk_stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


def vectorise_text(text: str):
    # Get the path to the trained model file and load it
    vectoriser_path = "./movie_genre_vectoriser.pkl"
    with open(vectoriser_path, 'rb') as filename:
        vectoriser = joblib.load(filename)

    return vectoriser.transform([text])


def transform_text(title: str, description: str):
    """
    Applies a series of pre-processing steps to the inputs
    :param title:
    :param description:
    :return:
    """
    # Concatenate the title and the description fields
    combined_text = " ".join([title, description])
    # Remove unwanted chars, lower case
    text = clean_text(combined_text)
    # Convert the combined string to a list of words
    tokens = tokenise_text(text)
    # Converts the list of tokens to their lemmas.
    lemmas = lemmatise_text(tokens)
    # Filters out stopwords
    processed_text = filter_stopword(lemmas)
    # Joins all words back to a string
    concatenated_text = " ".join(processed_text)

    return concatenated_text
