# app.py
import os
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from io import StringIO
import sys

# sklearn common imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

app = Flask(__name__, static_folder='.')
CORS(app)

UPLOAD_FOLDER = 'datasets'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

AVAILABLE_DATASETS = ['Iris.csv', 'Titanic-Dataset.csv']

@app.route('/')
def index():
    # Serve the index.html file directly
    return send_from_directory('.', 'index.html')

@app.route('/run', methods=['POST'])
def run_code():
    user_code = request.form.get('code')
    dataset = request.files.get('dataset')
    dataset_name = request.form.get('dataset_name')

    try:
        dataset_path = ''
        
        if dataset:
            filename = dataset.filename
            dataset_path = os.path.join(UPLOAD_FOLDER, filename)
            dataset.save(dataset_path)
        elif dataset_name in AVAILABLE_DATASETS:
            dataset_path = os.path.join(UPLOAD_FOLDER, dataset_name)
            if not os.path.exists(dataset_path):
                return jsonify({'output': f"❌ Dataset {dataset_name} not found on server!"}), 400
        else:
            return jsonify({'output': "❌ Please upload a dataset or select a valid pre-existing dataset."}), 400

        # Read dataset
        df = pd.read_csv(dataset_path)

        # Available libraries + df + dataset_path for user code
        exec_globals = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'df': df,
            'dataset_path': dataset_path,  # 🔥 Now dataset_path is available inside user code
            # sklearn modules
            'train_test_split': train_test_split,
            'LabelEncoder': LabelEncoder,
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            'LogisticRegression': LogisticRegression,
            'LinearRegression': LinearRegression,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'RandomForestClassifier': RandomForestClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'SVC': SVC,
            'KNeighborsClassifier': KNeighborsClassifier,
            'GaussianNB': GaussianNB,
            'accuracy_score': accuracy_score,
            'confusion_matrix': confusion_matrix,
            'classification_report': classification_report
        }
        exec_locals = {}

        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        try:
            exec(user_code, exec_globals, exec_locals)
        except Exception as e:
            sys.stdout = old_stdout
            return jsonify({'output': f"❌ Error while executing your code:\n{traceback.format_exc()}"}), 400
        finally:
            sys.stdout = old_stdout

        output_text = mystdout.getvalue()

        # Handle plot
        plot_generated = False
        if os.path.exists('output.png'):
            plot_generated = True

        return jsonify({
            'output': output_text.strip(),
            'plot_generated': plot_generated
        })

    except Exception as e:
        return jsonify({'output': f"❌ Unexpected Server Error:\n{traceback.format_exc()}"}), 500

@app.route('/output.png')
def get_plot():
    return send_from_directory('.', 'output.png')

if __name__ == "__main__":
    # Use the PORT environment variable provided by Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
