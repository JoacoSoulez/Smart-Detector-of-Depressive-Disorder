from MHFA.data import get_tweets, get_tweets_and_reddit

import pandas as pd
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from preprocessor import clean
import joblib
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def expand_contractions(text):
    """ Replace contractions in the english language by the complete phrase"""
    # Contraction dictionary
    contractions = {
      "ain't": "am not",
      "aren't": "are not",
      "can't": "cannot",
      "can't've": "cannot have",
      "'cause": "because",
      "could've": "could have",
      "couldn't": "could not",
      "couldn't've": "could not have",
      "didn't": "did not",
      "doesn't": "does not",
      "don't": "do not",
      "hadn't": "had not",
      "hadn't've": "had not have",
      "hasn't": "has not",
      "haven't": "have not",
      "he'd": "he would",
      "he'd've": "he would have",
      "he'll": "he will",
      "he'll've": "he will have",
      "he's": "he is",
      "how'd": "how did",
      "how'd'y": "how do you",
      "how'll": "how will",
      "how's": "how is",
      "I'd": "I would",
      "I'd've": "I would have",
      "I'll": "I will",
      "I'll've": "I will have",
      "I'm": "I am",
      "I've": "I have",
      "isn't": "is not",
      "it'd": "it had",
      "it'd've": "it would have",
      "it'll": "it will",
      "it'll've": "it will have",
      "it's": "it is",
      "let's": "let us",
      "ma'am": "madam",
      "mayn't": "may not",
      "might've": "might have",
      "mightn't": "might not",
      "mightn't've": "might not have",
      "must've": "must have",
      "mustn't": "must not",
      "mustn't've": "must not have",
      "needn't": "need not",
      "needn't've": "need not have",
      "o'clock": "of the clock",
      "oughtn't": "ought not",
      "oughtn't've": "ought not have",
      "shan't": "shall not",
      "sha'n't": "shall not",
      "shan't've": "shall not have",
      "she'd": "she would",
      "she'd've": "she would have",
      "she'll": "she will",
      "she'll've": "she will have",
      "she's": "she is",
      "should've": "should have",
      "shouldn't": "should not",
      "shouldn't've": "should not have",
      "so've": "so have",
      "so's": "so is",
      "that'd": "that would",
      "that'd've": "that would have",
      "that's": "that is",
      "there'd": "there had",
      "there'd've": "there would have",
      "there's": "there is",
      "they'd": "they would",
      "they'd've": "they would have",
      "they'll": "they will",
      "they'll've": "they will have",
      "they're": "they are",
      "they've": "they have",
      "to've": "to have",
      "wasn't": "was not",
      "we'd": "we had",
      "we'd've": "we would have",
      "we'll": "we will",
      "we'll've": "we will have",
      "we're": "we are",
      "we've": "we have",
      "weren't": "were not",
      "what'll": "what will",
      "what'll've": "what will have",
      "what're": "what are",
      "what's": "what is",
      "what've": "what have",
      "when's": "when is",
      "when've": "when have",
      "where'd": "where did",
      "where's": "where is",
      "where've": "where have",
      "who'll": "who will",
      "who'll've": "who will have",
      "who's": "who is",
      "who've": "who have",
      "why's": "why is",
      "why've": "why have",
      "will've": "will have",
      "won't": "will not",
      "won't've": "will not have",
      "would've": "would have",
      "wouldn't": "would not",
      "wouldn't've": "would not have",
      "y'all": "you all",
      "y'alls": "you alls",
      "y'all'd": "you all would",
      "y'all'd've": "you all would have",
      "y'all're": "you all are",
      "y'all've": "you all have",
      "you'd": "you had",
      "you'd've": "you would have",
      "you'll": "you will",
      "you'll've": "you will have",
      "you're": "you are",
      "you've": "you have"}

    contractions = dict((k.lower(), v.lower()) for k,v in contractions.items())

    c_re = re.compile('(%s)' % '|'.join(contractions.keys()))

    def replace(match):
        return contractions[match.group(0)]
    return c_re.sub(replace, text)

def remove_numbers(text):
    """ Remove numbers """
    words_only = ''.join([w for w in text if not w.isdigit()])
    return words_only

def remove_consecutive_duplicates(text):
    """Remove consecutive duplicates from text"""

    new_s = ""
    prev = ""
    for c in text:
        if len(new_s) == 0:
            new_s += c
            prev = c
        if c == prev:
            continue
        else:
            new_s += c
            prev = c
    return new_s

def replace_emojis_and_emoticons(text):
    """
    Find emoticons in the text and replace them by a word according to their sentiment
    Remove emojis from text
    """
    emoticons_happy = set([
        ':-\)', ':\)', ';\)', ':o\)', ':\]', ':\3', ':c\)', ':>', '=]', '8\)', '=\)', ':}',
        ':^\)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-\)\)', ':\*', ':^\*', '>:P', ':-P', ':P', 'X-P', ":'-\)", ":'\)",
        'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:\)', '>;\)', '>:-\)','<3'])

    emoticons_happy_compiled = re.compile('(%s)' % "|".join(emoticons_happy))

    emoticons_sad = set([':L', ':-/', '>:/', ':S', '>:\[',':@',':-\(', ':\[', '=L', ':<',':-\|\|',
                    ':-\[', ':-<', '=\\', '=/', '>:\(', ':\(', '>.<', ":'-\(", ":'\(", ':\\', ':-c',
                     ':c', ':\{',';\('])

    emoticons_sad_compiled = re.compile('(%s)' % "|".join(emoticons_sad))

    emoji_pattern = re.compile("["
             u"\U0001F600-\U0001F64F"  # emoticons
             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
             u"\U0001F680-\U0001F6FF"  # transport & map symbols
             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
             u"\U00002702-\U000027B0"
             u"\U000024C2-\U0001F251"
             "]+", flags=re.UNICODE)


    text = emoji_pattern.sub(r'', text)
    text = emoticons_happy_compiled.sub(r'happy', text)
    text = emoticons_sad_compiled.sub(r'sad', text)

    return text

def to_lower(text):
    """ Lower case all the letters of the string """
    return text.lower()

def remove_bad_symbols(text):
    """Remove unwanted symbols from text"""
    bad_symbols = re.compile('[^0-9a-z #+_]')
    return bad_symbols.sub(' ', text)

def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    return text

def remove_stop_words(text):
    """ Remove Stop words from text """


    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.append('https')
    stopwords.append('com')
    stopwords.append('http')
    stopwords.append('twitter')
    stopwords.append('m')
    stopwords.append('www')

    stop_words = set(stopwords)

    word_tokens = nltk.word_tokenize(text)

    filtered_text = [w for w in word_tokens if not w in stop_words]

    text = ' '.join(filtered_text)

    return text

def remove_stop_words_and_lemmatize(text):
    """ Remove Stop words from text """
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.append('https')
    stopwords.append('com')
    stopwords.append('http')
    stopwords.append('twitter')
    stopwords.append('m')
    stopwords.append('www')

    stop_words = set(stopwords)

    word_tokens = nltk.word_tokenize(text)

    without_stopwords = [w for w in word_tokens if not w in stop_words]

    lemmatizer = WordNetLemmatizer()

    lemmatized = [lemmatizer.lemmatize(word) for word in without_stopwords]

    return ' '.join(lemmatized)

def remove_context_symbol(text):
    import re
    return re.sub('<[^>]+>', '', text)

def clean_text(texts_sequence):
    """ Return a preprocessed sequence of texts """
    print('cleaning')


    return texts_sequence.apply(
        to_lower).apply(
        expand_contractions).apply(
        replace_emojis_and_emoticons).apply(
        clean).apply(
        remove_context_symbol).apply(
        remove_bad_symbols).apply(
        remove_punctuation).apply(
        remove_numbers).apply(
        remove_stop_words_and_lemmatize).apply(
        remove_consecutive_duplicates)


def vectorize(X):

    tfid3 = joblib.load('tfid3_vectorizer.sav')


    vector = tfid3.transform(X)


    return vector




if __name__ == "__main__":
    df = get_tweets_and_reddit()

    print(clean_text(df['text']))
