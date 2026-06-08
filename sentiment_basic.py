import csv #we need to save the results in a csv file so that we can interpret the sentiment score for each headlines
from transformers import pipeline #importing pipeline from transformers library so that we can use pipeline to call/download the model we need for the task

sentences = ["Strong earnings growth and expanding margins signal robust profitability ahead for the company.",
            "Investor confidence surged after the Fed signaled a pause in rate hikes, driving markets to new highs.",
            "The firm's record-breaking revenue this quarter exceeded analyst expectations by a wide margin.",
            "The merger is expected to create synergies, though integration risks remain a concern for analysts.",
            "Credit rating agencies downgraded the company's bonds, citing deteriorating fundamentals and elevated leverage."] #assigned variable for the lists of sentences we got from news.

classifier = pipeline("sentiment-analysis", model = "ProsusAI/finbert") #downloading the specific model from HF with the help of pipeline and naming the model as classifier

with open("sent_file.csv","w") as file: 
    writer = csv.writer(file) #creating/writing a csv file named "sent_file.csv"
    writer.writerow(["sentence", "label", "score"]) #naming the columns in the file
    for sentence in sentences: #taking out each sentence from the sentences
        x = classifier(sentence) #x is the result of each sentence running into classifier
        print(sentence,x[0]['label'],x[0]['score']) #printing the results with sentences
        writer.writerow([sentence,x[0]['label'],x[0]['score']]) #writing all the results to the csv file


    
