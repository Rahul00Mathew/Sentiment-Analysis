from google_play_scraper import search, reviews, Sort
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re

# Get all the details about Kuku fm
results = search("kuku fm", lang="en", country="in")

# Get package id of the app
app_package = results[0]['appId']
num_samples = 60000

# Get 'num_samples'(60000) of the latest reviews sorted in the newest order
result, continuation_token = reviews(
    app_package,
    lang='en',  # Language
    country='in',  # Country
    sort=Sort.NEWEST,  # or Sort.MOST_RELEVANT
    count=num_samples  # Number of reviews to fetch
)

reviews = list([r['content'] for r in result])
ratings = list([r['score'] for r in result])

# Checks if the text has only numbers or special characters like '?', '!!', '499' , etc
def has_letters(text):
    if pd.isna(text):
        return False
    
    # Check if there's at least one letter (English alphabet)
    return bool(re.search(r'[a-zA-Z]', text))


def clean(text):

    # Removes all non-english letters from the review
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Removes extra spaces between words
    text = re.sub(r'\s+', ' ', text)

    # Removes the leading and trailing white spaces
    text = text.strip()

    # If the text has no english alphabets after the above mentioned steps, return an empty string
    if not has_letters(text):
        text = re.sub(r'[^a-zA-Z]', '', text)
    
    # If the length of the resulting string is <3 like 'op', 'ok' replace it with nan
    if len(text) < 3:
        text = np.nan

    return text

# Get a clean dataset passing each review through the clean function and label each review using its rating
data = [(clean(reviews[r]), ratings[r]) for r in range(num_samples)]

# Convert into pandas dataframe
df = pd.DataFrame(data, columns=['Review', 'Rating'])

# Drop all the rows with nan values 
df_cleaned = df.dropna()
df_cleaned.to_csv(r'C:\Users\Rahul Mathew\Desktop\Python\ML pipeline\gplay_reviews.csv' , index=False)

# Convert the reviews into sentiments based on the rating
# Rating > 3 -> Positive
# Rating = 3 -> Neutral
# Rating < 3 -> Negative
def rating_to_sentiment(rating):
    if rating > 3:
        return 1
    elif rating == 3:
        return 0
    else:
        return -1

sentiments = [rating_to_sentiment(rating) for rating in df_cleaned['Rating']]

df_cleaned = pd.DataFrame(list(zip(df_cleaned['Review'], sentiments)), columns=['Review', 'Rating'])

# Split the cleaned dataframe into train/test split
train_df, test_df = train_test_split(df_cleaned, test_size=0.2, random_state=42)

# Save the train and test files
train_df.to_csv(r'C:\Users\Rahul Mathew\Desktop\Python\ML pipeline\train_gplay_reviews.csv' , index=False)
test_df.to_csv(r'C:\Users\Rahul Mathew\Desktop\Python\ML pipeline\test_gplay_reviews.csv' , index=False)