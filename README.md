# Classwell
A Sentiment analyser using finbert model.

## What it does?
- Classifies the given new headlines into 3 classes such as Positive, Negative and Neutral with confident score.

## Models used -
- Used ProsusAI/finbert from huggingface, which was downloaded with help of pipeline from transformers.

## How to run?
- get a list of news healines of a particular company from the desired news source.
- open the file "classify_pandas.py"
- create a csv file with those healines (or) download csv file with headlines in it.
- place them all inside the same folder.
- make sure you are entering the same name of the file inside the input_path argument while calling the function, if not no worries about the errors -- since we have programmed in a way that it doesn't crash.
- name the output_path file as per your demand, make sure you enter the name with the paths if needed.
- Finally run the code and your desired results will be saved your expected path -- Time to interpret the Analysis NOW🔥.

## Files - 
- classify_csv.py - basic script, first version
- classify_pandas.py - reads from CSV using Python's built-in csv module
- sentiment_basic.py - final version, reads and saves using pandas with error handling
