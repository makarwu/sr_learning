import argparse
from cmath import cos
import json
import re
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a sample of a JSON dataset

example_data = [
    {"instruction": "What is the capital of Italy?",
     "input": "", "output": "The capital of Italy is Rome."
     },
    {"instruction": "What's the capital city of Italy?",
     "input": "", "output": "The capital city is Rome."
     },
    {"instruction": "Identify the main verb in the sentence: 'The cat sleeps on the couch.'",
     "input": "", "output": "The verb is 'sleeps'."
     },
    {"instruction": "Identify the verb in the following sentence: The cat sleeps on the couch.",
     "input": "", "output": "The verb in the sentence is \"sleeps.\""
     },
    # ...
]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def find_near_duplicates(json_data, threshold=0.75, key="instruction"):
    # The higher the threshold, the more similar the texts have to be match

    #1. Extract the instructions
    text = [preprocess_text(item[key]) for item in json_data if item[key]]
    near_duplicates = []
    indices_to_remove = set()

    if not text:
        return {}, near_duplicates
    
    # Vectorize the text data
    vectorizer = TfidfVectorizer(stop_words=None, analyzer="char", ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(text)

    # Compute cosine similarity between each pair of entries
    cos_sim_matrix = cosine_similarity(tfidf_matrix)

    # Find pairs of near-duplicate instructions based on the threshold

    for i in range(len(cos_sim_matrix)):
        for j in range(i+1, len(cos_sim_matrix)):
            if cos_sim_matrix[i, j] > threshold:
                if len(json_data[i][key]) <= 1 or len(json_data[j][key]) <= 1:
                    continue
                near_duplicates.append((json_data[i], json_data[j], cos_sim_matrix[i, j]))
                if key in ("input", "output"): # Don't remove duplicates based on the instruction
                    indices_to_remove.add(j) # Mark the second entry for removal
    
    # Remove the near-duplicate entries
    filtered_json_data = [item for index, item in enumerate(json_data) if index not in indices_to_remove]

    return filtered_json_data, near_duplicates

def find_print_and_remove_near_duplicates(json_data, remove_duplicates=False, threshold=0.75):
    """
    Searches each key in the first JSON object for duplicates across a list of JSON objects.
    Prints the duplicates if found.
    """
    for key in json_data[0].keys():
        if remove_duplicates:
            json_data, near_duplicates = find_near_duplicates(json_data, key=key, threshold=threshold)
        else:
            _, near_duplicates = find_near_duplicates(json_data, key=key, threshold=threshold)
        separator = 50*'='
        print(f"\n\n{separator}\nSearching '{key}' for duplicates...\n{separator}")
        if not near_duplicates:
            print("No duplicates found")
        else:
            for dup in near_duplicates:
                print(
                    f"Duplicate pair found with similarity {dup[2]:.2f}:\n"
                    f"1. {dup[0][key]}\n2. {dup[1][key]}\n"
                )     
    return json_data

if __name__ == "__main__":
    pass