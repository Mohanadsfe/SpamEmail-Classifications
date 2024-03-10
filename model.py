import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
import os
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import string 
nltk.download('punkt')
import contractions
import re
from Algorithm import EmailClassifierFactory


# Define a custom preprocessor that joins tokenized words with spaces
def custom_preprocessor(text):
    return ' '.join(text)

# loading the data from a CSV file to a pandas DataFrame
csv_path = os.path.join(os.path.dirname(__file__), "..", "/Users/mohanadsafi/Desktop/spam_emailsProject/mail_dataSpam.csv")
df = pd.read_csv(csv_path)

# Replace the null values with a null string
mail_data = df.where((pd.notnull(df)), '')


# Define a list of specific keywords
spam_keywords = ["money", "win", "discount", "free", "offer", "guaranteed","free","click","website","bank","credit","visa","account"]

def fix_contractions(email):
  email = contractions.fix(email)
  return email

def custom_tokenize(email, keep_punct=False, keep_alnum=False, keep_stop=False):
    token_list = word_tokenize(email)

    if not keep_punct:
        token_list = [token for token in token_list if token not in string.punctuation]

    if not keep_alnum:
        token_list = [token for token in token_list if token.isalpha()]

    if not keep_stop:
        stop_words = set(stopwords.words('english'))
        stop_words.discard('not')  # Remove 'not' from stopwords, as it's important 
        token_list = [token for token in token_list if not token in stop_words]

    return token_list


# Convert the messages of emails to lowercase
def to_lowercase(word):
    result = word.lower()
    return result

def remove_special_characters(word):
    result = word.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return result

def stem_tokens(tokens, stemmer):
  token_list = []
  for token in tokens:
    token_list.append(stemmer.stem(token))
  return token_list

def fix_contractions(email):
  email = contractions.fix(email)
  return email

def word_repetition(email):
  email = re.sub(r'(.)\1+', r'\1\1', email)
  return email

def punct_repetition(email, default_replace=""):
  email = re.sub(r'[\?\.\!]+(?=[\?\.\!])', default_replace, email)
  return email

def replace_url(tweet, default_replace=""):
  tweet = re.sub('(http|https):\/\/\S+', default_replace, tweet)
  return tweet

def tokenizer(text):
    text=re.sub("(\\W)"," \\1 ",text)
    return re.split("\\s+",text)

# Define tweet processing function
# def process_email(email, verbose=False):
#     email = to_lowercase(email)  # Convert to lowercase
#     email = fix_contractions(email)  # Replace contractions
#     email = punct_repetition(email)  # Replace punctuation repetition
#     email = word_repetition(email)  # Replace word repetition
#     email = remove_special_characters(email)  # Remove special characters
#     email = replace_url(email)  # Replace URLs

#     # Tokenization & Stemming
#     tokens = custom_tokenize(email, keep_alnum=False, keep_stop=False)  # Tokenize
#     stemmer = SnowballStemmer("english")  # Define stemmer
#     stem = stem_tokens(tokens, stemmer)  # Stem tokens

#     return stem

porter_stemmer=PorterStemmer()
def preprocessor(text):
    
    text=text.lower() 
    text=re.sub("\\W"," ",text) 
    text=re.sub("\\s+(in|the|all|for|and|on)\\s+"," _connector_ ",text) 
    words=re.split("\\s+",text)
    stemmed_words=[porter_stemmer.stem(word=word) for word in words]
    return ' '.join(stemmed_words)


# Create a function to check for the presence of keywords
def check_keywords(tokens):
    return [int(keyword in tokens) for keyword in spam_keywords]

# Add keyword features to your dataset
mail_data['money_keyword'] = mail_data['Message'].apply(lambda x: 0 if any(check_keywords(x)) else 1)
currency_symbols = ["$", "€", "£", "¥", "₹", "₽", "₺", "₡", "₴", "₩", "₮", "﷼", "৳", "د.إ", "฿", "₿", "₭", "₾", "₦", "₣", "ₛ", "₪"]


def contains_money_keywords(email_text):
    for symbol in currency_symbols:
        if symbol in email_text:
            return True
    return False

# Define a custom preprocessor that joins tokenized words with spaces and applies lowercase
def custom_preprocessor(tokens):
    text = ' '.join(map(str, tokens))  # Convert integers to strings before joining
    return text.lower()


mail_data['contains_money_keywords'] = mail_data['Message'].apply(lambda x: 0 if contains_money_keywords(x) else 1)

# Label spam mail as 0; ham mail as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

# Separating the data as texts and label
# mail_data['Message'] = mail_data['Message'].apply(process_email)
mail_data['Message'] = mail_data['Message']

# X = mail_data[['Message', 'contains_money_keywords', 'money_keyword']].values.tolist()
X = mail_data['Message']

Y = mail_data['Category']

# Splitting the data into training data & test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=3, stratify=Y)

# Transform the text data to feature vectors using the custom preprocessor
feature_extraction = TfidfVectorizer(tokenizer=tokenizer,ngram_range=(1,2),min_df=0.006,preprocessor=preprocessor)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


# Convert Y_train and Y_test values to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#################################################################
# Create a text-based menu
print("Select an algorithm:")
print("1. Logistic Regression")
print("2. Support Vector Machine (SVM)")
print("3. Random Forest")
print("4. Decision Tree")
factory = EmailClassifierFactory()
id = input()
model = factory.create_classifier(int(id))

# Training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)

# Prediction on training data
prediction_on_training_dataTF = model.predict(X_train_features)
accuracy_on_training_dataTF = accuracy_score(Y_train, prediction_on_training_dataTF)
print('TF-IDF Accuracy on training data: ', accuracy_on_training_dataTF)

# Prediction on test data
prediction_on_test_dataTF = model.predict(X_test_features)
accuracy_on_test_dataTF = accuracy_score(Y_test, prediction_on_test_dataTF)
print('TF-IDF Accuracy on test data:', accuracy_on_test_dataTF)
print("\n")
################################


feature_extractionCV = CountVectorizer(tokenizer=tokenizer,ngram_range=(1,2),min_df=0.006,preprocessor=preprocessor)
X_train_features_CV = feature_extractionCV.fit_transform(X_train)
X_test_features_CV = feature_extractionCV.transform(X_test)

# Prediction on training data
prediction_on_training_dataCV = model.predict(X_train_features_CV)
accuracy_on_training_dataCV = accuracy_score(Y_train, prediction_on_training_dataCV)
print('CV Accuracy on training data: ', accuracy_on_training_dataCV)

# Prediction on test data
prediction_on_test_dataCV = model.predict(X_test_features_CV)
accuracy_on_test_dataCV = accuracy_score(Y_test, prediction_on_test_dataCV)
print('CV Accuracy on test data: ', accuracy_on_test_dataCV)
print("\n")

################################





confusion = confusion_matrix(Y_test, prediction_on_test_dataTF)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()




x=1
while(int(x)):
    
    # input_mail = ["I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted and will fulfill my promise. You have been wonderful and a blessing at all times"]
    input_mail = input("Please enter your email text:\n ")

    # Convert text to feature vectors
    # processed_email = process_email(input_mail)
    input_data_features = feature_extraction.transform([input_mail])

    # Making prediction
    prediction = model.predict(input_data_features)

    if prediction == 1:
        print('Ham mail')
    else:
        print('Spam mail')

    x = input('Please enter 1 if you want to continue ,else 0 \n')
    x = int(x)
    
