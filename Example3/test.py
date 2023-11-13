# import pandas as pd
# python -m spacy download en_core_web_sm


# df = pd.read_csv('Example3Schedule.csv')


import spacy
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk

string = "R01 - North PR, Piping Receive, Erect, Bolt-up/Weld-out, Trim/Punch UPR. Exp. Loop Mod 19 WP-89"
delimiters = [".", ","]

for delimiter in delimiters:
    string = " ".join(string.split(delimiter))

result = string.split("  ")

print(result)


# Download the NLTK data (only need to do this once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load the English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")


def is_verb_spacy(word):
    # Process the word with spaCy
    doc = nlp(word)

    # Check if any token in the processed document is a verb
    for token in doc:
        if token.pos_ == 'VERB':
            return True
    return False


def is_verb(word):
    # Tokenize the word
    words = word_tokenize(word)

    # Tag the words with part-of-speech
    tagged_words = pos_tag(words)

    # Check if any of the tags indicate a verb
    for _, tag in tagged_words:
        if tag.startswith('VB'):
            return True
    return False


# for word in result:
#     if is_verb(word):
#         print(f"{word} is a verb.")
#     else:
#         print(f"{word} is not a verb.")

# print("SPACY")

# for word in result:
#     if is_verb_spacy(word):
#         print(f"{word} is a verb.")
#     else:
#         print(f"{word} is not a verb.")
# # Example usage
# word_to_check = "run"
# if is_verb(word_to_check):
#     print(f"{word_to_check} is a verb.")
# else:
#     print(f"{word_to_check} is not a verb.")
