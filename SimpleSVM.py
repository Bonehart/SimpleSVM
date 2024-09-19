import nltk
import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Constants
# redacted actual names
PROF_YEAR = ''
QUESTION = ''
FILE_PATH = './recode.csv'

# Read the CSV file and preprocess it
data = pd.read_csv(FILE_PATH, encoding='utf-8').astype(str)
data['RECODE'].replace('nan', '99.0', inplace=True)

# Define the text cleaning function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean, lower, remove stopwords, and lemmatize text."""
    return ' '.join(
        lemmatizer.lemmatize(word) 
        for word in re.sub('[^a-zA-Z]', ' ', text).lower().split() 
        if word not in stop_words
    )

# Apply text preprocessing
data['VALUE'] = data['VALUE'].apply(clean_text)

# Filter relevant data
X = data.loc[data['QUESTION_ID'] == QUESTION, 'VALUE']
y = data.loc[data['QUESTION_ID'] == QUESTION, 'RECODE']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train SVM model
model = svm.SVC(C=55, kernel='linear', probability=True, degree=100, gamma='auto')
model.fit(X_train_vec, y_train)

# Evaluate the model
train_preds = model.predict(X_train_vec)
test_preds = model.predict(X_test_vec)

print(f"SVM Train Accuracy: {accuracy_score(train_preds, y_train) * 100:.2f}%")
print(f"SVM Test Accuracy: {accuracy_score(test_preds, y_test) * 100:.2f}%")

# Load JSON input data
with open(f'./{PROF_YEAR}.json', encoding='utf-8-sig') as f:
    data_input = json.load(f)

# Extract relevant inputs
inputs = [x for x in data_input['recodeother'] if x['question'] == QUESTION]
input_values = [x['value'] for x in inputs]
input_options = [x['options'] for x in inputs]

# Clean and vectorize input values
input_values_clean = [clean_text(text) for text in input_values]
X_input_vec = vectorizer.transform(input_values_clean)

# Make predictions for input values
input_predictions = model.predict(X_input_vec)

# Store predictions in a DataFrame
results_df = pd.DataFrame({
    'Value': input_values,
    'Prediction': input_predictions,
    'Options': input_options
})

# Update JSON with predictions
for row in results_df.itertuples():
    for entry in data_input['recodeother']:
        if entry['question'] == QUESTION and entry['value'] == row.Value:
            entry['recode'] = row.Prediction

# Write the updated JSON back to the file
with open(f'./{PROF_YEAR}.json', 'w', encoding='utf-8-sig') as outfile:
    json.dump(data_input, outfile, indent=3)

# Display summary and distribution of predictions
print(results_df.groupby('Prediction').count())

# Calculate and print class distribution of y
y_counts = Counter(y)
for value, count in y_counts.items():
    print(f"{value}: {count / len(y):.2%}")
