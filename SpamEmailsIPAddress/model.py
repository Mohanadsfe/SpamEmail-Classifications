# import pandas as pd
# import re

# # Load the dataset
# df = pd.read_csv('spam_assassin.csv')

# # Define regular expressions to extract headers
# to_pattern = re.compile(r'^To: (.+)$')

# # Initialize lists to store extracted headers
# from_list = []
# to_list = []
# faddress_list = []
# message_list = []

# # Extract headers from each email
# for index, row in df.iterrows():
#     # Use slicing to extract the email from index 5 to 24
#     from_email = row['text'][5:25].strip()
#     to_match = to_pattern.match(row['text'])
#     faddress = row['text'][169:178].strip()
#     message = row['text'][2400:2451].strip()

#     if from_email:
#         from_list.append(from_email)
#     else:
#         from_list.append(None)
        
#     if to_match:
#         to_list.append(to_match.group(1))
#     else:
#         to_list.append(None)
#     faddress_list.append(faddress)
#     message_list.append(faddress)

# # Add extracted headers to DataFrame
# df['From'] = from_list
# df['Faddress'] = faddress_list
# df['Message'] = message_list

# # Create a new DataFrame with extracted headers
# new_df = df[['text', 'From', 'Faddress','Message', 'target']]
# # Create a new DataFrame with extracted headers

# # Save the new dataset to a CSV file
# new_df.to_csv('spam_assassin_new.csv', index=False)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the preprocessed dataset
df = pd.read_csv('spam_assassin_new.csv')

# Prepare features and target variable
X_message = df['Message']
X_faddress = df['Faddress']
X_combined = X_message + ' ' + X_faddress  # Combine message and faddress
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vect, y_train)
nb_pred = nb_model.predict(X_test_vect)
nb_accuracy = accuracy_score(y_test, nb_pred)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_vect, y_train)
rf_pred = rf_model.predict(X_test_vect)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vect, y_train)
lr_pred = lr_model.predict(X_test_vect)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("Naive Bayes Accuracy:", nb_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)
