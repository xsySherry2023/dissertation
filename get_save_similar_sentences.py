import json


with open('data/dev.json', 'r') as f:
    json_data = json.load(f)



# restructure original json

new_datas_intra = []
for item in json_data['data']['intrasentence']:   ###!
    sentences = []
    for sentence in item['sentences']:
        sentences.append({
            'sentence': sentence['sentence'],
            'label': sentence['gold_label']
        })
    new_item = {
        'id': item['id'],
        'context': item['context'],
        'sentences': sentences
    }
    new_datas_intra.append(new_item)

print(new_datas_intra)





with open('data/wikisent2.txt', 'r', encoding='utf8') as file:
    lines = file.readlines()
# print(data)

#sample the 2 millon from the 7.8 millon examples
import random
# set seed to fix the sample
random.seed(100)
# print(len(lines))
lines = random.sample(lines, 2000000) # set the size of sample as 2 millon







from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# process the most similar 5 sentences，and save them as the later other shots prompt datasets

five_prompt_examples  = []

# #omit "BLANK " part
contexts_cleaned = [item['context'].replace('BLANK', '').strip() for item in new_datas_intra]
five_prompt_examples = []
for i, new_data in enumerate(new_datas_intra, 1):
    ### process the prompt information
    # Select the most similar five sentences from wikisent2 as propmt information.
    vectorizer = TfidfVectorizer()
    # Calculate the matrix and save it for future use.
    new_lines = lines[:]  # Duplicate the lines list using slices 
    new_lines.append(contexts_cleaned[i-1])
    tfidf_matrix = vectorizer.fit_transform(new_lines)
    wiki_sentence_tfidf_matrix = tfidf_matrix[:-1]
    test_sentence_tfidf_matrix = tfidf_matrix[-1]

    # Calculate the similarity between test_sentence and each sentence in test_lists
    similarity_scores = cosine_similarity(test_sentence_tfidf_matrix, wiki_sentence_tfidf_matrix)

    # only put ID inside
    highest_similarity_index = similarity_scores.argsort()[0][-6:-1]
    
    five_prompt_examples.append(highest_similarity_index)






# save all similar sentences to be used later
import csv
output_file_path = 'data/similar_sentences_data.csv'


# transpose the datab
headers = ["one-sentence", "two-sentence","three-sentence","four-sentence","five-sentence"]

# Write list to CSV file
with open(output_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)  # write column name
    for row in five_prompt_examples:
        writer.writerow(row)
