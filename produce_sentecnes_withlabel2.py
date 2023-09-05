with open('data/wikisent2.txt', 'r', encoding='utf8') as file:
    lines = file.readlines()
# print(data)

#sample the 2 millon from the 7.8 millon examples
import random
random.seed(100)  # set seed to fix the sample
lines = random.sample(lines, 2000000) # set the size of sample as 2 millon
lines = [item.replace('\n','',) for item in lines]



import torch
from transformers import BertTokenizer, BertForSequenceClassification
import csv

# 加载模型和分词器
model_path = "model_labels/checkpoint-1341"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # 替换为你的分词器

# 如果使用GPU
model = model.to('cuda')

predictions = []

# 对lines中的每一行进行分类
for text in lines:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # 如果使用GPU
    inputs = {key: val.to('cuda') for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        predictions.append(predicted_class)

# 保存到CSV
with open('data/labels.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Text", "Predicted Label"])
    for text, pred in zip(lines, predictions):
        writer.writerow([text, pred])
