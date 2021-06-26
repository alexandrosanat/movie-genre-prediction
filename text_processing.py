import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *


# Check if the tokenizer exists and download if not
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def clean_text(text: str):
    """
    Applies basic pre-processing to clean a sentence
    :param text: (str) A string of text that contains the movie title and description
    :return: text: (str) Input string with only characters of the alphabet
    """
    text = text.replace("_", " ")  # replace underscores with space
    text = re.sub("\'", "", text)  # remove backslash-apostrophe
    text = re.sub("[^a-zA-Z]", " ", text)  # remove everything except alphabets
    text = ' '.join(text.split())  # remove whitespaces
    text = text.lower()  # convert text to lowercase

    return text


def lemmatise_text(token_list: list):
    """
    Converts a list of words to their lemmas.
    :param token_list: (list) A list of tokenised words
    :return: lemma_list: (list) A list of the lemmas of the word tokens
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
    :param lemma_list: (list) A list of lemmatised words
    :return: filtered_sentence: (str) A string of joined list of words excluding stopwords
    """
    filtered_sentence = []  # New list to store words
    nltk_stop_words = set(stopwords.words("english"))
    for w in lemma_list:
        # Loop through all words and filter out stopwords
        if w not in nltk_stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


def transform_text(title: str, description: str):
    """
    Applies a series of pre-processing steps to the inputs
    :param title: (str) The movie title
    :param description: (str) The movie description
    :return: concatenated_text: (str) A transformed concatenated string of title and description
    """

    combined_text = " ".join([title, description])  # Concatenate the title and description fields
    text = clean_text(combined_text)  # Apply basic text cleaning
    tokens = word_tokenize(text)  # Convert the combined string to a list of words
    lemmas = lemmatise_text(tokens)  # Converts the list of tokens to their lemmas.
    processed_text = filter_stopword(lemmas)  # Filters out stopwords
    concatenated_text = " ".join(processed_text)  # Joins all words back to a string

    return concatenated_text
