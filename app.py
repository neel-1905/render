import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'  # Temporary file storage on the server

# Load the training dataset to initialize the model
file_path = 'Instagram_data_by_Bhanu.csv'  # Replace with your dataset path
data = pd.read_csv(file_path, encoding='latin1')

# Feature Engineering for training
numeric_features = data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']].values
tfidf = TfidfVectorizer(max_features=50)
hashtag_features = tfidf.fit_transform(data['Hashtags'].astype(str)).toarray()

X = np.hstack([numeric_features, hashtag_features])
y = data['Impressions'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numeric features
scaler = StandardScaler()
X_train[:, :numeric_features.shape[1]] = scaler.fit_transform(X_train[:, :numeric_features.shape[1]])
X_test[:, :numeric_features.shape[1]] = scaler.transform(X_test[:, :numeric_features.shape[1]])

# Train the model
model = PassiveAggressiveRegressor(max_iter=1000, random_state=42, tol=1e-3)
model.fit(X_train, y_train)

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload, process it, and return predictions."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file temporarily
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load and process the uploaded file
    try:
        uploaded_data = pd.read_excel(file_path)

        # Process numeric and hashtag features
        numeric_inputs = uploaded_data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']].values
        hashtag_inputs = tfidf.transform(uploaded_data['Hashtags'].astype(str)).toarray()

        # Combine features
        features = np.hstack([scaler.transform(numeric_inputs), hashtag_inputs])

        # Predict impressions
        predictions = model.predict(features)
        uploaded_data['Predicted Reach'] = predictions

        # Save the result to a CSV in /tmp
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.csv')
        uploaded_data.to_csv(result_path, index=False)

        return jsonify({'message': 'Predictions complete', 'download_url': '/uploads/result.csv'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve the result file."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
