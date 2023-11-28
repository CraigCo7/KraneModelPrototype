import csv
import copy
import ast
import os
import openai
from datetime import datetime
from itertools import product
import numpy as np
import torch
from torch import cosine_similarity
import pandas as pd
import gensim.downloader as api

# Import the datasets

# schedDf = pd.read_csv('CE Input/Prototype Schedule Input.csv')
# bomDf = pd.read_csv('CE Input/Prototype Materials Input (Updated).csv')
# csv_out_path = "CE Output/Output-CEModelProcess (GlorifiedDataManipulation).csv"

schedDf = pd.read_csv('CE Input/Prototype 2 Schedule Input (Updated).csv')
bomDf = pd.read_csv('CE Input/Prototype 2 Materials Input (Updated).csv')
csv_out_path = "CE Output/Output-CEModelProcess Prototype 2(GlorifiedDataManipulation).csv"

# Import the word model
model = api.load("word2vec-google-news-300")


def word2vec(model, word):
    return model[word.lower()]


def sentence2vec(value, model):
    sentence = value.split()

    # Initialize an empty vector to store the sentence vector
    sentence_vector = np.zeros(model.vector_size)

    # Counter to keep track of the number of words in the sentence
    word_count = 0

    # Calculate the sum of word vectors for the words in the sentence
    for word in sentence:
        if word.lower() in model:
            sentence_vector += model[word.lower()]
            word_count += 1

    # Check if there are words in the sentence that have embeddings
    if word_count > 0:
        # Calculate the average by dividing by the number of words
        sentence_vector /= word_count
    return sentence_vector


def cosineSimilarity(vector1, vector2):

    word_vector = torch.tensor(vector1).clone().detach().requires_grad_(True)
    sentence_vector = torch.tensor(
        vector2).clone().detach().requires_grad_(True)
    # Calculate cosine similarity between the sentence embeddings
    similarity = cosine_similarity(
        word_vector.unsqueeze(0), sentence_vector.unsqueeze(0))
    val = similarity[0].item()
    return val


# Feature Modelling

date_format = "%m/%d/%Y"

bomDf.dropna(subset=['Material BOM'], inplace=True)
schedDf.dropna(subset=['Activity Description'], inplace=True)

# Remove all whitespace
bomDf = bomDf.map(lambda x: x.strip() if isinstance(x, str) else x)
schedDf = schedDf.map(lambda x: x.strip() if isinstance(x, str) else x)


# startDate = datetime.strptime(row[2].strip(), date_format)
# endDate = datetime.strptime(row[3].strip(), date_format)
# ProcStepDuration = str(endDate-startDate).split(',')[0]


# Create SemanticWordVec Feature
schedDf['SemanticWordVec'] = schedDf['Activity Description'].apply(
    sentence2vec, args=(model,))

# Create Duration Feature
schedDf['Duration'] = schedDf.apply(lambda row: str(datetime.strptime(
    row['End Date'], date_format) - datetime.strptime(row['Start Date'], date_format)).split(',')[0], axis=1)


# PROCSTEP

threshold = 0.95

# Array of 2D data [activity description, wordvec]
selected_columns = schedDf[['Activity Description',
                            'SemanticWordVec']].values.tolist()
# print(selected_columns[0])

for i, row1 in enumerate(selected_columns):
    for j, row2 in enumerate(selected_columns[i+1:]):
        cosine_sim = cosineSimilarity(row1[1], row2[1])
        if cosine_sim > threshold:
            # print(row1[0], row2[0], cosine_sim)
            row1.append(1)
            selected_columns.remove(row2)

# for i in selected_columns:
#     print(i[0])


# PROCESSOUTPUT
# Try to use ML model?

# GPT3.5
# openai.api_key = "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"
# GPT4.0
openai.api_key = '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'


activity_list = [i[0] for i in selected_columns]


def findProcessOutputOpenAi():
    prompt = "You will be given a list of sub tasks. Answer this question for each sub task: _ is a sub task you need to fulfil in order to complete which part of a building? Example: 'Install Tiles'. Output: 'Roof'. Answer in one word only, and make sure the keys are identical to the values in the input list. Return a python dictionary. Here is the list of sub tasks: " + \
        ', '.join(map(str, activity_list))
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        # model="gpt-4-1106-preview",
        prompt=prompt,
        max_tokens=1000,
        temperature=0,
        best_of=1
    )
    # Convert ChatGPT Output to Dictionary.
    my_dict = ast.literal_eval(response['choices'][0]['text'].strip())
    return my_dict

# print(findProcessOutputOpenAi())

# PROCESS SUCCESSOR
# Try to use ML model?


def findProcessStepSuccessorOpenAi():
    prompt = "You will be given a list of sub tasks. These sub tasks are done sequentially, and your job is to determine which task precedes/succeeds the other. Answer in one word only, and make sure the keys are identical to the input list. Return a python dictionary. Here is the list of sub tasks: " + \
        ', '.join(map(str, activity_list))
    response = openai.Completion.create(
        # model="gpt-3.5-turbo-instruct",
        model="gpt-4-1106-preview",
        prompt=prompt,
        max_tokens=1000,
        temperature=0,
        best_of=1
    )
#     return response
    my_dict = ast.literal_eval(response['choices'][0]['text'].strip())
    return my_dict

# print(findProcessStepSuccessorOpenAi())


# MATERIAL ASSIGNMENT


# Thinking: compare cosine_sim of each row to categories in scheduleDF. Assign each material to the category that is most similar.
categories_wordVec = [[inner_list[0], inner_list[1]]
                      for inner_list in selected_columns]
categories_only = [inner_list[0] for inner_list in selected_columns]

# Key: activity, Value: Empty list
my_dict = {key: [] for key in categories_only}

materialAssignment = copy.deepcopy(my_dict)

material_list = bomDf['Material BOM'].values.tolist()

# FIRST = "You will be given a list of materials and a list of activities. Your job is to determine which material is required to perform which activity. Some activities don't require any materials and some activities require multiple materials. If you are unsure, assign a material to the activity with the highest probability. Return a python dictionary, and make sure the keys are identical to the input list. Here is the list of activities: " + \
#         ', '.join(map(str, activity_list)) + ". " + "Here is the list of materials: " + \
#         ', '.join(map(str, material_list))


# def findMaterialAssignmentOpenAi():
#     prompt = "Assign the following list of materials to the most likely activity it is associated with. Some activities don't require any materials and some activities require multiple materials. Return a python dictionary, and make sure the keys are identical to the input list. Here is the list of activities: " + \
#         ', '.join(map(str, activity_list)) + ". " + "Here is the list of materials: " + \
#         ', '.join(map(str, material_list))
#     response = openai.Completion.create(
#         model="gpt-3.5-turbo-instruct",
#         prompt=prompt,
#         max_tokens=2000,
#         temperature=0,
#         best_of=1
#     )
#     my_dict = ast.literal_eval(response['choices'][0]['text'].strip())
#     return my_dict

for index, row in bomDf.iterrows():
    highest = -1
    assignment = None
    activity_desc = row[0]
    wordVec1 = sentence2vec(row[0], model)
    for category in categories_wordVec:
        wordVec2 = category[1]
        cosine_sim = cosineSimilarity(wordVec1, wordVec2)
        if cosine_sim > highest:
            highest = cosine_sim
            assignment = category[0]
    materialAssignment[assignment].append(row[1])

# print(materialAssignment)
# materialAssignment Stores the Assignments from BOM to Schedule...


# BatchSize

non_empty_activities = {key: value for key,
                        value in materialAssignment.items() if len(value) > 0}

batchSize = copy.deepcopy(my_dict)

# for i in non_empty_activities.values():
#     print(i)

# for i in non_empty_keys.keys():
#     print(i)
# print(non_empty_keys)

num_duplicates = {sublist[0]: len(sublist)-1 for sublist in selected_columns}

for key, value in non_empty_activities.items():
    for material in value:
        amount = int(bomDf.loc[bomDf['Material BOM']
                     == material]['Amount'].values)
        batchSize[key].append(amount/num_duplicates[key])

# print(batchSize)

# Duration


date_format = "%m/%d/%Y"

duration = copy.deepcopy(my_dict)

for key in duration.keys():
    start = datetime.strptime(
        schedDf[schedDf['Activity Description'] == key]['Start Date'].values.tolist()[0], date_format)
    end = datetime.strptime(
        schedDf[schedDf['Activity Description'] == key]['End Date'].values.tolist()[0], date_format)
    duration[key] = str(end-start).split(',')[0]

# print(duration)


# Resource Assignment

resourceAssignment = copy.deepcopy(my_dict)

for key in duration.keys():
    resource = schedDf[schedDf['Activity Description']
                       == key]['Subcontractor'].values.tolist()[0]
    resourceAssignment[key] = resource

# print(resourceAssignment)


# Resource Quantity

resourceQuantity = {}


for subcontractor in schedDf['Subcontractor'].unique():
    filteredDf = schedDf[schedDf['Subcontractor'] == subcontractor]
    max_value = filteredDf['Manpower'].max()
    resourceQuantity[subcontractor] = max_value

# print(resourceQuantity)


# Output Configuration!!


output_data = [["ID", "ProcessOutput", "ProcStep", "ProcStepSuccessor", "ProcStepDuration",
                "ProcMaterialAssignment", "ProcMaterialBatchSize", "ProcResourceAssignment", "ProcResourceQuantity"]]

processOutput = findProcessOutputOpenAi()
processSuccessor = findProcessStepSuccessorOpenAi()


def findProcessOutput(activity):
    return processOutput[activity.split(' - ')[0]]


def findProcStepSuccessor(activity):
    try:
        out = processSuccessor[activity]
        return out.split(' - ')[0]
    except KeyError:
        return []


def findProcStepDuration(activity):
    return duration[activity]


def findProcMaterialAssignment(activity):
    return materialAssignment[activity]


def findProcMaterialBatchSize(activity):
    return batchSize[activity]


def findProcResourceAssignment(activity):
    return resourceAssignment[activity]


def findProcResourceQuantity(activity):
    resAssignment = resourceAssignment[activity]
    return resourceQuantity[resAssignment]


data = [[i+1, findProcessOutput(item[0]), item[0].split(' - ')[0], findProcStepSuccessor(item[0]), findProcStepDuration(item[0]), findProcMaterialAssignment(
    item[0]), findProcMaterialBatchSize(item[0]), findProcResourceAssignment(item[0]), findProcResourceQuantity(item[0])] for i, item in enumerate(selected_columns)]

for row in data:
    output_data.append(row)

with open(csv_out_path, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)

    # Write the processed data to the output file
    csv_writer.writerows(output_data)
