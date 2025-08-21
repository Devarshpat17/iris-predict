from flask import Flask, request, render_template
import pandas as pd
import pickle
import os
import logging
from waitress import serve
from werkzeug.middleware.proxy_fix import ProxyFix
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(df):
    return df.drop('Id', axis=1)

class IrisClassifier:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.accuracy = None
        self.feature_importance = None
        
    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df = preprocess_data(df)
        X = df.drop('Species', axis=1)
        y = df['Species']
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
            
    def train(self):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.evaluate()
        self.feature_importance = dict(zip(
            self.X_train.columns,
            self.model.feature_importances_
        ))
        
    def evaluate(self):
        from sklearn.metrics import accuracy_score
        y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, filename):
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
            
    def predict(self, sepal_length, sepal_width, petal_length, petal_width):
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                columns=self.X_train.columns)
        prediction = self.model.predict(input_data)[0]
        probabilities = self.model.predict_proba(input_data)[0]
        confidence = max(probabilities)
        return prediction, confidence

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Initialize and train model
model = IrisClassifier()
model.load_data('Iris.csv')

if os.path.exists('model1.pkl'):
    model.load('model1.pkl')
else:
    model.train()
    model.save('model1.pkl')

@app.route('/')
def home():
    host = request.headers.get('Host')
    accuracy = model.accuracy
    
    if host.startswith('api.'):
        return {
            'service': 'Iris Classifier Pro API',
            'version': '2.0',
            'status': 'running',
            'model': {
                'type': 'Random Forest Classifier',
                'accuracy': accuracy,
                'feature_importance': model.feature_importance,
                'features': list(model.feature_importance.keys()),
                'classes': ['setosa', 'versicolor', 'virginica']
            },
            'endpoints': {
                'root': '/ - API information (GET)',
                'predict': '/predict - Make predictions (POST)'
            },
            'example_payload': {
                'sepalLength': 5.1,
                'sepalWidth': 3.5,
                'petalLength': 1.4,
                'petalWidth': 0.2
            }
        }
    return render_template('index1.html', accuracy=accuracy)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    host = request.headers.get('Host')
    
    if request.method == 'GET':
        if host.startswith('api.'):
            return {
                'message': 'Send a POST request with iris measurements'
            }
        return render_template('index1.html')
    
    try:
        if host.startswith('api.'):
            data = request.get_json()
            sepal_length = float(data['sepalLength'])
            sepal_width = float(data['sepalWidth'])
            petal_length = float(data['petalLength'])
            petal_width = float(data['petalWidth'])
        else:
            sepal_length = float(request.form['sepalLength'])
            sepal_width = float(request.form['sepalWidth'])
            petal_length = float(request.form['petalLength'])
            petal_width = float(request.form['petalWidth'])
        
        prediction, confidence = model.predict(
            sepal_length, sepal_width, petal_length, petal_width)
        
        if host.startswith('api.'):
            return {
                'success': True,
                'prediction': {
                    'species': prediction,
                    'confidence': float(confidence)
                },
                'input': {
                    'sepalLength': sepal_length,
                    'sepalWidth': sepal_width,
                    'petalLength': petal_length,
                    'petalWidth': petal_width
                },
                'timestamp': datetime.datetime.now().isoformat()
            }
        return render_template('index1.html', 
                             prediction=prediction,
                             accuracy=model.accuracy)
                             
    except (KeyError, ValueError) as e:
        error_msg = f"Invalid input: {str(e)}"
        if host.startswith('api.'):
            return {'error': error_msg}, 400
        return render_template('index1.html', error=error_msg)

if __name__ == '__main__':
    logger.info("Starting Enhanced Iris Classifier application...")
    logger.info("Using Random Forest model with feature importance...")
    print("\nServer is running behind Nginx reverse proxy")
    print("Access the application at:")
    print("- http://localhost:5001")
    print("- http://app2.abcdp17.com")
    print("- http://api.app2.abcdp17.com")
    print("\nPress Ctrl+C to quit")
    
    # Start Waitress server on a different port
    serve(app, host='127.0.0.1', port=5001, _quiet=False)
