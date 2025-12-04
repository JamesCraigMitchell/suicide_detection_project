import re
import nltk
import pandas as pd
import matplotlib

# Set up matplotlib to use Agg backend
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import joblib

# Section 1 - Data Collection and Preprocessing

# Loading the dataset. "r" at the start ensures that single backslashes can be used as it is a raw string
df = pd.read_csv("Suicide_Detection.csv")

# Alternatively, the dataset URL = https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch

df = df.drop('Unnamed: 0', axis=1)  # Removing the first column that contains redundant indexing

df = df.head(2000)  # Reducing the number of rows in the dataset

print("Dataset preview:\n", df.describe())  # Displaying a preview of the dataset

print(df.info(), "\n")

print(df.head, "\n")

print("Missing values:\n", df.isnull().sum())  # Checks the df for any missing values and returns the count

column_names = df.columns


# Processing the text


def clean_text(text):
    cleaned_text = []
    lemmatizer = WordNetLemmatizer()

    stop_words = set(stopwords.words('english'))

    for sent in text:
        sent = sent.lower()
        sent = re.sub(r'[^a-zA-Z\s]', '', sent)  # removing special characters and numbers
        sent = re.sub(r'(\b\w*(\w)\2{2,}\w*\b)', '', sent)  # removing words with plus 3 consecutive repeating letters
        sent = ' '.join([word for word in sent.split() if word not in stop_words])  # removing stopwords
        sent = ' '.join([lemmatizer.lemmatize(word) for word in sent.split()])  # Applying lemmatization to each word

        cleaned_text.append(sent)
    return cleaned_text


# Cleaning the text column using the above method
df['cleaned_text'] = clean_text(df['text'])

# Tokenization
df['tokens'] = df['cleaned_text'].apply(lambda x: nltk.word_tokenize(x))

# Joining the tokens back into a single word before vectorisation
df['joined_tokens'] = df['tokens'].apply(lambda x: ' '.join(x))

# Vectorizing the tokens
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['joined_tokens']).toarray()
y = df['class']
features = (vectorizer.get_feature_names_out())

# Check 50 tokens
print("Sample Vocabulary:", features[:50], "\n")

# Checking the samples and features
print("Shape of X (TF-IDF matrix):", X.shape, "\n")

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Section 2 + 3 - Model Selection Training + Prediction and Evaluation


# Base model training and prediction for later comparison

# Train Base Logistic Regression
base_lr = LogisticRegression(max_iter=1000)  # Default Logistic Regression model
base_lr.fit(X_train, y_train)
y_pred_base_lr = base_lr.predict(X_test)
print("Base Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_base_lr))

# Train Base Random Forest
base_rf = RandomForestClassifier()  # Default Random Forest
base_rf.fit(X_train, y_train)
y_pred_base_rf = base_rf.predict(X_test)
print("Base Random Forest Classification Report:\n", classification_report(y_test, y_pred_base_rf))

# Calculating the accuracy of the base models
base_lr_accuracy = accuracy_score(y_test, y_pred_base_lr)
base_rf_accuracy = accuracy_score(y_test, y_pred_base_rf)

# Define model parameters for optimization
model_params = {
    'logistic_regression': {
        'model': LogisticRegression(max_iter=1000),  # Logistic Regression model
        'params': {
            'C': [0.1, 1, 10]  # Regularization parameter
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),  # Random Forest Classifier
        'params': {
            'n_estimators': [10, 50],  # Number of trees
            'max_depth': [None, 10],  # Tree depth
            'min_samples_split': [2, 5]  # Minimum samples for split
        }
    }
}

# Train and tune the models using GridSearchCV
trained_models = {}

for model_name, model_info in model_params.items():
    print(f"Training and tuning {model_name}...")

    # Create GridSearchCV for each model
    grid_search = GridSearchCV(estimator=model_info['model'],
                               param_grid=model_info['params'],
                               cv=5,  # 5-fold cross-validation
                               n_jobs=-1,  # Use all processors
                               scoring='accuracy',  # Accuracy as evaluation metric
                               verbose=1)

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters and best score
    print(f"Best Hyperparameters for {model_name}: {grid_search.best_params_}")
    print(f"Best Accuracy for {model_name}: {grid_search.best_score_}")

    # Store the best model in trained_models
    best_model = grid_search.best_estimator_
    trained_models[model_name] = best_model

    # Save the best model to a file
    joblib.dump(best_model, f'{model_name}_best_model.pkl')

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))

# Defining the base models for the stacking model
base_models = [
    ('lr', trained_models['logistic_regression']),
    ('rf', trained_models['random_forest'])
]

# Defining the metamodel
meta_model = SVC(kernel='linear', probability=True, random_state=42)

# Creating the stacking model
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

print("Training stacking_model...")

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Save the stacking model to a file
joblib.dump(stacking_model, 'stacking_model.pkl')

# Evaluate the model
y_pred = stacking_model.predict(X_test)
print("Stacking Model Classification Report:\n", classification_report(y_test, y_pred))

# Section 4 - Visualisation

# Load the best models (Logistic Regression and Random Forest) after GridSearchCV
best_lr = joblib.load('logistic_regression_best_model.pkl')  # Load the best Logistic Regression model
best_rf = joblib.load('random_forest_best_model.pkl')  # Load the best Random Forest model

# Load the stacking model
final_stacking = joblib.load('stacking_model.pkl')

# Predict on the test set using the best and stacking model(s)
y_pred_lr = best_lr.predict(X_test)
y_pred_rf = best_rf.predict(X_test)
y_pred_fs = final_stacking.predict(X_test)


def plot_confusion_matrices():  # Function to plot the confusion matrices

    final_model_names = ['Tuned Logistic Regression', 'Tuned Random Forest', 'Stacking Model']
    final_predictions = [y_pred_lr, y_pred_rf, y_pred_fs]
    labels = ['Non-Suicide', 'Suicide']

    for names, predictions in zip(final_model_names, final_predictions):
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'{names} Confusion Matrix')
        plt.savefig(f'/output/{names}_confusion_matrix.png') # Save to unique file
        plt.close() # Close the figure to free memory


plot_confusion_matrices()

# Outputting accuracy scores of each model

# Consolidate accuracy calculations
base_lr_accuracy = accuracy_score(y_test, y_pred_base_lr)  # Base Logistic Regression
tuned_lr_accuracy = trained_models['logistic_regression'].score(X_test, y_test)  # Tuned Logistic Regression
base_rf_accuracy = accuracy_score(y_test, y_pred_base_rf)  # Base Random Forest
tuned_rf_accuracy = trained_models['random_forest'].score(X_test, y_test)  # Tuned Random Forest
stacking_model_accuracy = stacking_model.score(X_test, y_test)  # Stacking Model

# List of accuracies and model names
model_names = ['Base Logistic Regression', 'Tuned Logistic Regression',
               'Base Random Forest', 'Tuned Random Forest', 'Stacking Model']
accuracies = [base_lr_accuracy, tuned_lr_accuracy, base_rf_accuracy, tuned_rf_accuracy, stacking_model_accuracy]

# Output all accuracies
print("Model Accuracies:")
for model_name, accuracy in zip(model_names, accuracies):  # Zip to pair model names with accuracies
    print(f"{model_name}: {accuracy:.4f}")


# Convert X_train to a DataFrame for better visualization with feature names as columns
X_train_df = pd.DataFrame(X_train, columns=vectorizer.get_feature_names_out())


# Feature importance plot for Random Forest
def plot_feature_importance_rf(model, X_train_df, top_n=20):
    importance_df = pd.DataFrame({
        'Feature': X_train_df.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title(f'Random Forest - Top {top_n} Features')
    plt.savefig(f'/output/random_forest_feature_importance.png') # Save to unique file
    plt.close() # Close the figure


# Plot the top 20 features
plot_feature_importance_rf(best_rf, X_train_df, top_n=20)
