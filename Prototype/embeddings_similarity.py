import pandas as pd
import numpy as np
import string
import itertools
import spacy

# function to remove punctuation from text (input is a string)


def clean_text(sentence):

    clean_sentence = "".join(
        l for l in sentence if l not in string.punctuation)

    return clean_sentence

# function to calculate the cosine


def cosine_similarity_calc(vec_1, vec_2):

    sim = np.dot(vec_1, vec_2)/(np.linalg.norm(vec_1)*np.linalg.norm(vec_2))

    return sim

# function to calculate cosine similarity using word vectors (input is a series)


def embeddings_similarity(sentences):
    sentence_pairs = list(itertools.combinations(sentences, 2))
    sentence_a = [pair[0] for pair in sentence_pairs]
    sentence_b = [pair[1] for pair in sentence_pairs]
    sentence_pairs_df = pd.DataFrame(
        {'sentence_a': sentence_a, 'sentence_b': sentence_b})

    sentence_pairs_df = sentence_pairs_df.loc[
        pd.DataFrame(
            np.sort(sentence_pairs_df[['sentence_a', 'sentence_b']], 1),
            index=sentence_pairs_df.index
        ).drop_duplicates(keep='first').index
    ]

    sentence_pairs_df = sentence_pairs_df[sentence_pairs_df['sentence_a']
                                          != sentence_pairs_df['sentence_b']]

    embeddings = spacy.load('en_core_web_lg')

    def calculate_similarity(row):
        doc_a = embeddings(clean_text(row['sentence_a']))
        doc_b = embeddings(clean_text(row['sentence_b']))

        # Check if SpaCy provided vectors for all words in both sentences
        if doc_a.has_vector and doc_b.has_vector:
            return cosine_similarity_calc(doc_a.vector, doc_b.vector)
        else:
            # Handle cases where vectors are missing (e.g., for some words)
            return 0.0  # You can choose a default value for missing vectors

    sentence_pairs_df['similarity'] = sentence_pairs_df.apply(
        calculate_similarity,
        axis=1
    )

    return sentence_pairs_df


# calculate similarity for sample sentences

def similarity(str1, str2):
    return embeddings_similarity([str1, str2])
