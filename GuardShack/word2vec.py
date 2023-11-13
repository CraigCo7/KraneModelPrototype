# pip install numpy
# pip install scipy
# pip install --upgrade gensim

import gensim.downloader as api
import numpy as np
from torch import cosine_similarity
import torch

# Load the pre-trained Word2Vec model (you can choose a different model)
# "word2vec-google-news-300" is just an example; replace it with the model of your choice.

info = api.info()  # show info about available models/datasets

model = api.load("word2vec-google-news-300")


def word2vec(model, word):
    return model[word.lower()]


def sentence2vec(model, sentence):
    sentence = sentence.split()

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
    return sentence_vector


def checkLowestCosine():
    majorProcesses = ["Base", "Walls", "Roof", "Electrical"]
    baseList = ["Base framing erection", "Subfloor installation"]
    wallList = ["Wall framing erection", "Sliding barrier and installation", "Window fitting",
                "Interior hardware installation", "Wall insulation", "Interior painting", "Exterior painting"]
    roofList = ["Roof framing erection",
                "Roof vapor barrier and tile installation", "Roofing insulation"]
    electricalList = ["Building power installation", "Lighting installation",
                      "Low voltage electrical installation", "Solar panel system install", "Backup power system test"]

    lists = [baseList,
             wallList, roofList, electricalList
             ]

    for process in majorProcesses:
        word_vector = model[process.lower()]
        for list in lists:
            lowestValue = 1
            for activity in list:
                words = activity.split()

                # Initialize an empty vector to store the sentence vector
                sentence_vector = np.zeros(model.vector_size)

                # Counter to keep track of the number of words in the sentence
                word_count = 0

                # Calculate the sum of word vectors for the words in the sentence
                for word in words:
                    if word in model:
                        sentence_vector += model[word.lower()]
                        word_count += 1

                # Check if there are words in the sentence that have embeddings
                if word_count > 0:
                    # Calculate the average by dividing by the number of words
                    sentence_vector /= word_count

                # Convert the numpy arrays to PyTorch tensors
                word_vector = torch.tensor(
                    word_vector).clone().detach().requires_grad_(True)
                sentence_vector = torch.tensor(
                    sentence_vector).clone().detach().requires_grad_(True)
                # Calculate cosine similarity between the sentence embeddings
                similarity = cosine_similarity(
                    word_vector.unsqueeze(0), sentence_vector.unsqueeze(0))
                val = similarity[0].item()
                if val < lowestValue:
                    lowestValue = val
            print(process, list, " had the lowest value of: ", lowestValue)
            print()
            print()


def sentence2VecConditional(model, sentence, word):
    comparison = word2vec(model, word)
    sentence = sentence.split()

    # Initialize an empty vector to store the sentence vector
    sentence_vector = np.zeros(model.vector_size)

    # Counter to keep track of the number of words in the sentence
    word_count = 0

    # Calculate the sum of word vectors for the words in the sentence
    for word in sentence:
        if word in model:
            word_vector = model[word.lower()]
            word_vector_test = torch.tensor(
                word_vector).clone().detach().requires_grad_(True)
            comparison_vector_test = torch.tensor(
                comparison).clone().detach().requires_grad_(True)
            # Calculate cosine similarity between the sentence embeddings
            similarity = cosine_similarity(
                word_vector_test.unsqueeze(0), comparison_vector_test.unsqueeze(0))
            val = similarity[0].item()
            if val > 0:
                sentence_vector += model[word.lower()]
                word_count += 1

    # Check if there are words in the sentence that have embeddings
    if word_count > 0:
        # Calculate the average by dividing by the number of words
        sentence_vector /= word_count
    return sentence_vector


def checkNoise(comparison):
    baseList = ["Base framing erection", "Subfloor installation"]
    wallList = ["Wall framing erection", "Sliding barrier and installation", "Window fitting",
                "Interior hardware installation", "Wall insulation", "Interior painting", "Exterior painting"]
    roofList = ["Roof framing erection",
                "Roof vapor barrier and tile installation", "Roofing insulation"]
    electricalList = ["Building power installation", "Lighting installation",
                      "Low voltage electrical installation", "Solar panel system install", "Backup power system test"]

    lists = [baseList,
             wallList, roofList, electricalList
             ]
    try:
        comparison_vector = model[comparison.lower()]
    except KeyError:
        print("The key ", comparison, " has no associated word vector.")
        return
    for list in lists:
        for sentence in list:
            wordList = sentence.split()
            for word in wordList:
                try:
                    word_vector = model[word.lower()]
                except KeyError:
                    print("The key ", word, " has no associated word vector.")
                    continue
                word_vector = torch.tensor(
                    word_vector).clone().detach().requires_grad_(True)
                comparison_vector = torch.tensor(
                    comparison_vector).clone().detach().requires_grad_(True)

                similarity = cosine_similarity(
                    word_vector.unsqueeze(0), comparison_vector.unsqueeze(0))
                val = similarity[0].item()

                print(comparison, " & ", word, " similarity: ", val)


# checkNoise('construction')
# checkLowestCosine()
