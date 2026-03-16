#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Phase 1 - Imports
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
REVIEWS_DATA = os.getenv("REVIEWS_DATA")

import re
import random
import numpy as np
import pandas as pd
import pickle
from statistics import mode
import nltk
from nltk import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import stopwords

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup


# In[ ]:


# Phase 2 - Loading and Preprocessing the Data (This cleans the data and prepares it for training the model.)
# Load the dataset
data = pd.read_csv(REVIEWS_DATA, encoding='utf-8')
# Drop duplicate rows with 'Text' column
data.drop_duplicates(subset=['Text'], inplace=True)
# Drop any rows with missing values
data.dropna(inplace=True)
# Extract the 'Text' columns as input data
texts = data['Text'].tolist() # Converts the 'Text' column to a list of strings
# Extract the 'Summary' column
summary = data['Summary'].tolist() # Converts the 'Summary' column to a list of strings
# Replace any empty string summaries with NaN, then drop those rows too
data['Summary'].replace('', np.nan, inplace=True)
data.dropna(subset=['Summary'], inplace=True)


# In[ ]:


# Phase 3 - Preprocessing the Text Data (This includes tokenization, stemming, and removing stop words.)

# Step 1: Setting up tools:
file_path = REVIEWS_DATA
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        reviews_data = pd.read_csv(file)
        print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
# intialize lancaster stemmer and stop words
stemmer = LancasterStemmer()
stop_words = set(stopwords.words('english'))
# Intialize NLKT's WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# In[ ]:


# Phase 2: Step 2
"""
Step 2: A clean function that labeles inputs or outputs as either text or summary and then cleans them accordingly. This includes removing HTML tags, special characters, and stop words, as well as performing stemming and lemmatization.
@param text: The input text to be cleaned.
@return: A list of cleaned tokens.
"""
def clean_text(text):
    #Remove HTML tags using B4S
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    # Remove special characters and numbers using regex
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_text)
    # tokenize the cleaned text
    tokens = [word.lower() for word in word_tokenize(cleaned_text)]
    # Filter out words that are shorter than 3 chars
    tokens = [word for word in tokens if len(word) > 2]
    # Expand contractiosn using dictionary
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'t": " not",
        "'ve": " have",
        "'m": " am"
    }
    # check if sours is inputs: if it is remove stopwords
    if text in texts:
        tokens = [contractions.get(word, word) for word in tokens if word not in stop_words]
    # Check if source is a target: if it si only remove stop words
    elif text in summary:
        tokens = [contractions.get(word, word) for word in tokens if word not in stop_words]
    # finally return the list of cleaned tokens
    return tokens


# In[ ]:


# Phase 2: Step 3

# Step 3: Loop through the tokens and apply stemming and lemmatization to each word. This will help reduce the vocabulary size and improve the model's ability to generalize.
for text in texts:
    cleaned_tokens = clean_text(text)
    # Apply stemming and lemmatization
    stemmed_tokens = [stemmer.stem(word) for word in cleaned_tokens]
    # Apply lemmatization to the cleaned tokens
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in cleaned_tokens]


# In[ ]:


# Phase 2: Step 4: Deduplication and Final Cleaning
# Step 4: After stemming and lemmatization, we will have a list of tokens for each text. We can then deduplicate these tokens to further reduce the vocabulary size. Finally, we can join the cleaned tokens back into a single string for each text, which will be used as input to the model.
cleaned_texts = []
for text in texts:
    cleaned_tokens = clean_text(text)
    # Count unque tokens
    unique_tokens = set(cleaned_tokens)
    # Find the most common token
    most_common_token = mode(cleaned_tokens)
    # Find input and target length of tokens
    input_length = len(cleaned_tokens)
    target_length = len(clean_text(summary[texts.index(text)]))
    # Printing all 4 vals
    print(f"Unique tokens: {len(unique_tokens)}, Most common token: '{most_common_token}', Input length: {input_length}, Target length: {target_length}")


