import pandas as pd
from transformers import pipeline

def classify_headlines(input_path, output_path):

    classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    try:
        df = pd.read_csv(input_path)
        
    except FileNotFoundError:
        print(f"Erro:file not found - {input_path}")
        # return
    sentences = df['headline'].tolist()


    rows = []
    for sentence in sentences:
        x = classifier(sentence)        
        rows.append({
        'Headline':sentence,
        'label':x[0]['label'],
        'score':x[0]['score']
        })
        
    a = pd.DataFrame(rows)
    a.to_csv(output_path,index=False)
 
if __name__ == "__main__":
    
    classify_headlines("/Users/rizvi/Desktop/Basics/headlines.csv","/Users/rizvi/Desktop/Basics/output.csv")
        
