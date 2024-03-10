import pandas as pd
import os
import re

# Load the original dataset CSV file
csv_path = os.path.join(os.path.dirname(__file__), "..", "/Users/mohanadsafi/Desktop/spam_emailsProject/spam_assassin.csv")
df = pd.read_csv(csv_path)

# df['target'] = df['target'].apply(lambda x: 1 if x == 0 else 0)

# Swap the positions of the columns
# counts = df['target'].value_counts()
# print(counts)

# Define a function to extract email addresses using the "From" pattern
# def extract_email(text):
#     match = re.search(r'From\s+([^\s]+)', text)
#     if match:
#         return match.group(1)
#     else:
#         return None

# Apply the function to create a new 'email' column
# df['emailAddress'] = df['text'].apply(extract_email)

