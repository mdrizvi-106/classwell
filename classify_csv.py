import csv
from transformers import pipeline

sentences = []
with open('/Users/rizvi/Desktop/Basics/headlines.csv','r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        sentences.append(row[0])
        
print(sentences)
    
classifier = pipeline("sentiment-analysis",model = "ProsusAI/finbert")
with open("results.csv","w") as f:
    writer = csv.writer(f)
    writer.writerow(['Sentences','Label','Score'])
    for sentence in sentences:
        x = classifier(sentence)
        print(sentence,x[0]['label'],x[0]['score'])
        writer.writerow([sentence,x[0]['label'],x[0]['score']])