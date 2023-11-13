import re
from collections import Counter
import math


def cosine_similarity(str1, str2):
    # Tokenize the input strings into words
    words1 = re.findall(r'\w+', str1)
    words2 = re.findall(r'\w+', str2)

    # Create a set of unique words from both strings
    unique_words = set(words1).union(set(words2))

    # Create word vectors for both strings
    vector1 = Counter(words1)
    vector2 = Counter(words2)

    # Calculate the dot product of the two vectors
    dot_product = sum(vector1[word] * vector2[word] for word in unique_words)

    # Calculate the magnitude of each vector
    magnitude1 = math.sqrt(sum(vector1[word] ** 2 for word in unique_words))
    magnitude2 = math.sqrt(sum(vector2[word] ** 2 for word in unique_words))

    # Calculate the cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


# Example usage
str1 = "apple banana orange"
str2 = "banana orange grape"
similarity = cosine_similarity(str1, str2)
print("Cosine Similarity:", similarity)
