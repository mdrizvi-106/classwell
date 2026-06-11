from fastapi import FastAPI
from transformers import pipeline
app = FastAPI()

@app.get("/classify")
def classify(headlines: str):
    classifier = pipeline("sentiment-analysis", model = "ProsusAI/finbert")
    x = classifier(headlines)
    return {"Headline":headlines,"Label": x[0]['label'],"Score": x[0]['score']}
        
    
    