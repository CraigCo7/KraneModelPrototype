import pandas as pd
import openai
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import spacy

# Load your API key from an environment variable or secret management service
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "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"

scheduleDf = pd.read_csv("Example3Schedule.csv")
delimiters = [".", ","]
# Returns a boolean to indicate if word is a verb.
nlp = spacy.load("en_core_web_sm")


def is_verb_spacy(word):
    # Process the word with spaCy
    doc = nlp(word)

    # Check if any token in the processed document is a verb
    for token in doc:
        if token.pos_ == 'VERB':
            return True
    return False


def is_verb_nltk(word):
    # Tokenize the word
    words = word_tokenize(word)

    # Tag the words with part-of-speech
    tagged_words = pos_tag(words)

    # Check if any of the tags indicate a verb
    for _, tag in tagged_words:
        if tag.startswith('VB'):
            return True
    return False


def verbsOnly(value):
    newItems = []
    for delimiter in delimiters:
        string = " ".join(value.split(delimiter))
        result = string.split("  ")
        for i in result:
            if is_verb_spacy(i) or is_verb_nltk(i):
                if i not in newItems:
                    newItems.append(i)
    return ', '.join(newItems)


def findRack(value):
    return value.split('-')[0].strip()


def assignOutput(value, output):
    if value == output:
        return 1
    else:
        return 0
# Takes csv path as argument.
# Creates a new column based off the 'Activity Name' column, which retains only the verbs
# Drops the duplicates.
# Returns the new dataframe


def createScheduleDataframe(scheduleDf):
    df = pd.read_csv(scheduleDf)
    df.dropna(subset=['Activity Name'], inplace=True)
    df['VerbsOnly'] = df['Activity Name'].apply(verbsOnly)

    # Drop duplicates, 25 left
    df.drop_duplicates(subset=['VerbsOnly'], inplace=True)
    df['Rack'] = df['Activity Name'].apply(findRack)

    # Create Output Features

    uniqueOutputs = df.drop_duplicates(subset=['Rack'])
    # Iterate through all different output classes
    for i in uniqueOutputs['Rack'].values.tolist():
        label = "Output " + str(i.strip())
        df[label] = df['Rack'].apply(assignOutput, args=(i,))

    return df


createScheduleDataframe('Example3ScheduleAdjusted.csv')
