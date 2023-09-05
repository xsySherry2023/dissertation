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

print(new_datas_intra) #len(new_datas_intra) 2123


stereotype_sentences = []
anti_stereotype_sentences = []
for item in new_datas_intra:
    for sentence_info in item['sentences']:
        if sentence_info['label'] == 'stereotype':
            stereotype_sentences.append(sentence_info['sentence'])
        elif sentence_info['label'] == 'anti-stereotype':
            anti_stereotype_sentences.append(sentence_info['sentence'])

print(stereotype_sentences)
print("Anti-Stereotype Sentences:", anti_stereotype_sentences)

with open('data/neutral_sentences.txt', 'r', encoding='utf8') as file:
     neutral_sentences = file.readlines()
neutral_sentences = [item.replace('\n','',) for item in neutral_sentences]
print(neutral_sentences)
# len(neutral_sentences) 250


dataset = [(i,0)for i in stereotype_sentences] + [(i,1)for i in anti_stereotype_sentences] + [(i,2)for i in neutral_sentences]

# dataset = [("This is a positive text.", 0), 
#            ("This is a negative text.", 1), 
#            ("This is a neutral text.", 2)]
# dataset

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Specify the local path
model_path = "/scratch2/2787367x/new_env/bert-base-uncased" # or take replace the path position into 'bert-base-uncased'

# download tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)



# process the data using tokenizer
def encode(examples):
    return tokenizer([text for text, _ in examples], truncation=True, padding='max_length', max_length=256, return_tensors="pt")
def encode_labels(examples):
    return torch.tensor([label for _, label in examples])

# Split the data set into training set and validation set 8:2
train_texts, val_texts = train_test_split(dataset, test_size=0.2, random_state=42)

train_encodings = encode(train_texts)
val_encodings = encode(val_texts)
train_labels = encode_labels(train_texts)
val_labels = encode_labels(val_texts)



#Create a PyTorch dataset

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)




training_args = TrainingArguments(
    output_dir='./model_labels',  # Output and save path
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Set save strategy to "epoch"
    logging_dir='./logs',
    # logging_steps=10,
    # do_train=True,
    # do_eval=True,
    # no_cuda=False,
    # load_best_model_at_end=True,
    # metric_for_best_model="accuracy",
    # weight_decay=0.01,
    # save_total_limit=1,
    # push_to_hub=False
)


trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset=val_dataset
)

trainer.train()


with open('data/wikisent2.txt', 'r', encoding='utf8') as file:
    lines = file.readlines()
# print(data)

#sample the 2 millon from the 7.8 millon examples
import random
random.seed(100)  # set seed to fix the sample
lines = random.sample(lines, 2000000) # set the size of sample as 2 millon
lines = [item.replace('\n','',) for item in lines]



# evaluation

results = trainer.evaluate()
print(results)


from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
# 1. Load the model and tokenizer
# model_path = './model_labels'
# model = BertForSequenceClassification.from_pretrained(model_path)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



# ensure models and data in the same device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = model.to(device)



# lines = lines.to(device)
# classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
# results = classifier(lines)
# print(results)

# Convert lines to tensors and move to device before passing to pipeline
inputs = tokenizer(lines, return_tensors="pt", padding=True, truncation=True).to(device)

# Using model and tokenizer directly instead of pipeline for more control
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    results = torch.argmax(logits, dim=-1)

print(results)  # This will give the predicted class indices



# similarsentences = "People are very thin. This sentence is stereotype. People are unathetic. This sentence is anti-stereotype..."
# text = similarsentences+ + ' {"placeholder":"text_a"} {"mask"} {"placeholder":"text_b"} '



import csv

output_file_path = 'data/wiki_sentences_label.csv'

# Write list to CSV file
with open(output_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    for item in results:
        writer.writerow([item])
