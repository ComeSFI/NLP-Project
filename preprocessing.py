import numpy as np
import pandas as pd 
import re 
import nltk 
import spacy 
import string
from nltk.corpus import stopwords
from collections import Counter
from data.utils import emoticons
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocessed(data):
    pd.options.mode.chained_assignment = None
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('wordnet')
    cnt = Counter()
    
    data["text"] = data["text"].str.lower()
    data["noemoji"] = data["text"].apply(remove_emoji)
    data["noemote"] = data["noemoji"].apply(remove_emoji)
    data["wopunct"] = data["noemote"].apply(remove_punctuation)
    data["text_wo_stop"] = data["wopunct"].apply(remove_stopwords)
    totaltext = ' '.join(data["text_wo_stop"])
    splited_text = totaltext.split(' ')
    cnt = Counter(splited_text)
    most_common = cnt.most_common()
    FREQWORDS = [w for (w, word_count) in most_common[:10]] #if word_count>10000]
    data["text_wo_stopfreq"] = data["text_wo_stop"].apply(lambda text: remove_freqwords(text, FREQWORDS))
    
    RAREWORDS = [w for (w, word_count) in most_common[-len(FREQWORDS):]]
    
    data["text_wo_rare"] = data["text_wo_stopfreq"].apply(lambda text: remove_freqwords(text, RAREWORDS))
    data["text_lemmatized"] = data["text_wo_rare"].apply(lemmatize_words)
    data2 = pd.DataFrame(data["text_lemmatized"])
    
    return data2

def remove_emoticons(text):
    # You don't have to learn this regular expression, this is just an example of 
    # how tedious writting regex can be
    EMOTICONS = emoticons()
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)



def remove_emoji(text: str) -> str:
    """
    Removes emojis from the input text

    Args: 
        text (str): The input text to remove emojis from

    Returns:
        str: A next text without emojis
    """
    # define a regular expression pattern
    emoji_pattern = re.compile("[" 
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"  # Miscellaneous symbols
                           u"\U000024C2-\U0001F251"  # Enclosed characters
                           "]+", flags=re.UNICODE)   # '+' signifies that those characters can occur once or more consecutively

    text_wo_emoji = re.sub(emoji_pattern, '',text)
    return text_wo_emoji

def remove_punctuation(text: str) -> str:
    """
    Removes punctuation characters from the input text

    Args:
        text (str): The input text from which punctuation characters will be removed

    Returns:
        str: A new string with all punctuation characters removed
    """
    PUNCT_TO_REMOVE = string.punctuation
    translation_table = str.maketrans('','',PUNCT_TO_REMOVE) 
    return text.translate(translation_table)

def remove_stopwords(text: str) -> str:
    STOPWORDS = set(stopwords.words('english'))
    """
    Removes stopwords from the input text

    Args: 
        text (str): The input text from which stopwords will be removed

    Returns:
        str: A new string without stopwords
    """
    split = text.split()
    filtered_words =  [x for x in split if x not in STOPWORDS]# list comprehension, will go through split list and add the word to a new list only if the word is not in STOPWORDS
    filtered_string = " ".join(filtered_words)
    return filtered_string

def remove_freqwords(text: str, freq_words: list) -> str:
    """
    Removes a selection of frequent words from the input string

    Inputs:
        text (str): The input text from which frequent words will be removed
        freq_words (list): A list of frequent words to remove from the text

    Returns:
        str: A new string with all frequent words removed
    """
    split = text.split()
    filtered_words =  [x for x in split if x not in freq_words]# list comprehension, will go through split list and add the word to a new list only if the word is not in STOPWORDS
    filtered_string = " ".join(filtered_words)
    return filtered_string

def remove_rarewords(text: str, rare_words: list) -> str:
    """
    Removes a selection of rare words from the input string

    Inputs:
        text (str): The input text from which rare words will be removed
        rare_words (list): A list of rare words to remove from the text

    Returns:
        str: A new string with all rare words removed
    """
    # Your code here:
    # Filter out the most frequent words from a text string
    split = text.split()
    filtered_words =  [x for x in split if x not in rare_words]# list comprehension, will go through split list and add the word to a new list only if the word is not in STOPWORDS
    filtered_string = " ".join(filtered_words)
    return filtered_string


def lemmatize_words(text: str) -> str:
    
    """
    Apply lemmatization to the input string, considering words' POS tags.

    This function lemmatizes words in the input string based on their POS (Part-of-Speech) tags.
    
    Args:
        text (str): The input text to be lemmatized.

    Returns:
        str: A new string with lemmatized words.
    """
    wordnet_map = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }

    pos_tagged_text = nltk.pos_tag(nltk.tokenize.word_tokenize(text))
    lemmatized_words = [lemmatizer.lemmatize(words, pos = wordnet_map.get(pos[0], wordnet.NOUN)) for words, pos in pos_tagged_text ]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text