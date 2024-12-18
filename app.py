import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'  # Use /tmp for temporary file storage on Render

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
        # Load the file dynamically (Excel or CSV)
        if file.filename.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        elif file.filename.endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            return jsonify({'error': 'Unsupported file format. Upload an Excel or CSV file.'}), 400

        # Validate required columns
        required_columns = ['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows', 'Hashtags']
        if not all(col in data.columns for col in required_columns):
            return jsonify({'error': f'Missing required columns. Ensure the file contains {required_columns}'}), 400

        # Feature Engineering
        numeric_features = data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']].values
        tfidf = TfidfVectorizer(max_features=50)
        hashtag_features = tfidf.fit_transform(data['Hashtags'].astype(str)).toarray()

        # Combine features
        X = np.hstack([numeric_features, hashtag_features])

        # Normalize numeric features
        scaler = StandardScaler()
        X = np.hstack([scaler.fit_transform(numeric_features), hashtag_features])

        # Mock Impressions for Training (Temporary for Prediction)
        y = np.random.randint(1000, 10000, size=X.shape[0])  # Simulated impressions for demo purposes

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = PassiveAggressiveRegressor(max_iter=1000, random_state=42, tol=1e-3)
        model.fit(X_train, y_train)

        # Predict on the uploaded dataset
        predictions = model.predict(X)

        # Add predictions to the DataFrame
        data['Predicted Reach'] = predictions

        # Save the result to a CSV in /tmp
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.csv')
        data.to_csv(result_path, index=False)

        return jsonify({'message': 'Predictions complete', 'download_url': '/uploads/result.csv'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve the result file."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
