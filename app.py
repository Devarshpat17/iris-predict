from flask import Flask, request, render_template
import pandas as pd
import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
from waitress import serve
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_data(df):
    return df.drop('Id', axis=1)


class GaussianNaiveBayesModel:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.accuracy = None
        self.conf_matrix = None

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df = preprocess_data(df)
        X = df.drop('Species', axis=1)
        y = df['Species']
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    def train(self):
        from sklearn.naive_bayes import GaussianNB
        self.model = GaussianNB()
        self.model.fit(self.X_train, self.y_train)
        self.evaluate()

    def evaluate(self):
        from sklearn.metrics import accuracy_score, confusion_matrix
        y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.conf_matrix = confusion_matrix(self.y_test, y_pred)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, sepal_length, sepal_width, petal_length, petal_width):
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns=self.X_train.columns)
        return self.model.predict(input_data)[0]

    
        


app = Flask(__name__)

model = GaussianNaiveBayesModel()
model.load_data('Iris.csv')

# Check if model pickle exists
if os.path.exists('model.pkl'):
    model.load('model.pkl')
    model.train()  # To get accuracy/confusion matrix
else:
    model.train()
    model.save('model.pkl')

@app.route('/')
def home():
    accuracy = model.accuracy
    return render_template('index.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepalLength'])
    sepal_width = float(request.form['sepalWidth'])
    petal_length = float(request.form['petalLength'])
    petal_width = float(request.form['petalWidth'])
    
    prediction = model.predict(sepal_length, sepal_width, petal_length, petal_width)
    accuracy = model.accuracy
    return render_template('index.html', prediction=prediction, accuracy=accuracy)

if __name__ == '__main__':
    logger.info("Starting Iris Classifier application...")
    logger.info("Loading model and starting server...")
    print("Server is running at http://localhost:5000")
    print("Press Ctrl+C to quit")
    serve(app, host='0.0.0.0', port=5000, _quiet=False)