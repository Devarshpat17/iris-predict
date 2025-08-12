# Iris Flower Classification with Flask

This project implements a web application for classifying iris flower species using the Gaussian Naive Bayes algorithm. The application is built with Flask and utilizes the Iris dataset for training the model.

## Project Structure

```
iris-flask-app
├── src
│   ├── app.py            # Main entry point of the Flask application
│   ├── model.py          # Contains the Gaussian Naive Bayes model class
│   ├── utils.py          # Utility functions for data preprocessing
│   └── templates
│       └── index.html    # HTML template for user interface
├── data
│   └── Iris.csv          # Dataset used for training the model
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd iris-flask-app
   ```

2. **Install dependencies**:
   It is recommended to create a virtual environment before installing the dependencies.
   ```
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```
   python src/app.py
   ```

4. **Access the application**:
   Open your web browser and go to `http://127.0.0.1:5000/` to access the iris flower classification application.

## Usage

- Input the petal measurements (length and width) in the provided fields.
- Click on the "Predict" button to classify the iris flower species based on the input measurements.
- The predicted species will be displayed on the same page.

## License

This project is licensed under the MIT License.