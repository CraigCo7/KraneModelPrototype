import numpy as np
import pandas as pd
from word2vec import sentence2VecConditional, word2vec, sentence2vec
import torch
import gensim.downloader as api
import torch.nn.functional as F

# Manually create label for prediction, based on Guard Shack Data


def createLabel(id1, id2):  # returns 1 if id1 precedes id2
    if (id1 == 1) & (id2 == 2):
        return 1
    elif (id1 == 2) & (id2 == 3):
        return 1
    elif (id1 == 3) & (id2 == 5):
        return 1
    elif (id1 == 3) & (id2 == 6):
        return 1
    elif (id1 == 6) & (id2 == 7):
        return 1
    elif (id1 == 9) & (id2 == 7):
        return 1
    elif (id1 == 3) & (id2 == 8):
        return 1
    elif (id1 == 4) & (id2 == 8):
        return 1
    elif (id1 == 5) & (id2 == 8):
        return 1
    elif (id1 == 4) & (id2 == 10):
        return 1
    elif (id1 == 7) & (id2 == 11):
        return 1
    elif (id1 == 2) & (id2 == 13):
        return 1
    elif (id1 == 4) & (id2 == 14):
        return 1
    elif (id1 == 13) & (id2 == 14):
        return 1
    elif (id1 == 14) & (id2 == 15):
        return 1
    elif (id1 == 15) & (id2 == 16):
        return 1
    elif (id1 == 8) & (id2 == 17):
        return 1
    elif (id1 == 10) & (id2 == 17):
        return 1
    elif (id1 == 16) & (id2 == 17):
        return 1
    else:
        return 0


# Importing the dataset
# df = pd.read_csv('Guard Shack - Schedule 4WLA.csv')
df = pd.read_csv('CE Input/GuardShackSchedule.csv')
model = api.load("word2vec-google-news-300")
# model = api.load("conceptnet-numberbatch-17-06-300")

hardcodedProcesses = ["Base", "Wall", "Roof", "Electrical"]

date_format = "%d-%b-%y"


# Feature Engineering


def cosineSim(value, word):
    process = word2vec(model, word)
    word_vector = torch.tensor(
        process).clone().detach().requires_grad_(True)
    sentence_vector = torch.tensor(
        sentence2vec(model, value)).clone().detach().requires_grad_(True)
    cosine_similarity_value = F.cosine_similarity(
        word_vector, sentence_vector, dim=0)
    return cosine_similarity_value.item()


def get_sentence_embedding2(value, word):
    process = word2vec(model, word)
    sentence = value.split()[0:2]

    # Initialize an empty vector to store the sentence vector
    sentence_vector = np.zeros(model.vector_size)

    # Counter to keep track of the number of words in the sentence
    word_count = 0

    # Calculate the sum of word vectors for the words in the sentence
    for word in sentence:
        if word in model:
            sentence_vector += model[word.lower()]
            word_count += 1
    # Check if there are words in the sentence that have embeddings
    if word_count > 0:
        # Calculate the average by dividing by the number of words
        sentence_vector /= word_count

    word_vector = torch.tensor(
        process).clone().detach().requires_grad_(True)
    sentence_vector = torch.tensor(
        sentence_vector).clone().detach().requires_grad_(True)

    cosine_similarity_value = F.cosine_similarity(
        word_vector, sentence_vector, dim=0)
    return cosine_similarity_value.item()


def cosineSimConditional(value, word, comparison):
    process = word2vec(model, word)
    word_vector = torch.tensor(
        process).clone().detach().requires_grad_(True)
    sentence_vector = torch.tensor(
        sentence2VecConditional(model, value, comparison)).clone().detach().requires_grad_(True)
    cosine_similarity_value = F.cosine_similarity(
        word_vector, sentence_vector, dim=0)
    return cosine_similarity_value.item()


def addFeaturesForProcessOutput(df, hardcodedProcesses):
    # Remove empty rows
    df.dropna(subset=['Activity Name'], inplace=True)

    # Fix Syntax for Duration (from format Xd to float)
    df.loc[df['Duration'].str.endswith("d"), 'Duration'] = df.loc[df['Duration'].str.endswith(
        "d"), 'Duration'].str.rstrip('d').astype(float)
    df['ID'] = df.reset_index().index + 1

    # New Features. Cosine Similarity Between WordVectors
    for process in hardcodedProcesses:
        label = "cosineSimilarity(" + str(process) + ")"
        df[label] = df['Activity Name'].apply(cosineSim, args=(process,))
    for process in hardcodedProcesses:
        label = "cosineSimilarityComparison(" + str(process) + ")"
        df[label] = df['Activity Name'].apply(
            cosineSimConditional, args=(process, "construction"))
    for process in hardcodedProcesses:
        label = "cosineSimilarity2Words(" + str(process) + ")"
        df[label] = df['Activity Name'].apply(
            get_sentence_embedding2, args=(process,))
    return df


def addFeaturesForProcessStepSuccessor(schedule_path, process_assignments_path):
    # Create a new csv input with pairs of tasks from the dataset.
    csv_input_path = schedule_path
    process_level_assignments = process_assignments_path

    processDf = pd.read_csv(process_level_assignments)
    mapping_dict_process = processDf.set_index('Activity Name')[
        'Process Output'].to_dict()

    df = pd.read_csv(csv_input_path)
    df.dropna(subset=['Activity Name'], inplace=True)
    task_pairs = [(task1, task2) for task1 in df['Activity Name']
                  for task2 in df['Activity Name'] if (task1 != task2)]

    # Create a new DataFrame from the task_pairs list
    new_df = pd.DataFrame(task_pairs, columns=['Task1', 'Task2'])

    mapping_dict_start = df.set_index('Activity Name')[
        'Baseline Start'].to_dict()
    mapping_dict_finish = df.set_index('Activity Name')[
        'Baseline Finish'].to_dict()

    new_df['Task1Start'] = pd.to_datetime(
        new_df['Task1'].map(mapping_dict_start), format=date_format)
    new_df['Task1Finish'] = pd.to_datetime(
        new_df['Task1'].map(mapping_dict_finish), format=date_format)
    new_df['Task2Start'] = pd.to_datetime(
        new_df['Task2'].map(mapping_dict_start), format=date_format)
    new_df['Task2Finish'] = pd.to_datetime(
        new_df['Task2'].map(mapping_dict_finish), format=date_format)

    new_df['Task1ProcessOutput'] = new_df['Task1'].map(mapping_dict_process)
    new_df['Task2ProcessOutput'] = new_df['Task2'].map(mapping_dict_process)

    new_df['Task1StartDay'] = new_df['Task1Start'].dt.day
    new_df['Task1StartMonth'] = new_df['Task1Start'].dt.month
    new_df['Task1StartYear'] = new_df['Task1Start'].dt.year

    new_df['Task1FinishDay'] = new_df['Task1Finish'].dt.day
    new_df['Task1FinishMonth'] = new_df['Task1Finish'].dt.month
    new_df['Task1FinishYear'] = new_df['Task1Finish'].dt.year

    new_df['Task2StartDay'] = new_df['Task2Start'].dt.day
    new_df['Task2StartMonth'] = new_df['Task2Start'].dt.month
    new_df['Task2StartYear'] = new_df['Task2Start'].dt.year

    new_df['Task2FinishDay'] = new_df['Task2Finish'].dt.day
    new_df['Task2FinishMonth'] = new_df['Task2Finish'].dt.month
    new_df['Task2FinishYear'] = new_df['Task2Finish'].dt.year

    return new_df


def general(schedule_input):
    df = pd.read_csv(schedule_input)
    df.dropna(subset=['Activity Name'], inplace=True)
    df.drop_duplicates(subset=['Activity Name'])
    return df
# addFeaturesForProcessStepSuccessor('CE Input/Input.csv', 'CE Input/ProcessOutput.csv')
