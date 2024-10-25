import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering without GUI
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Load the dataset
file_path = '/Users/aditi/Desktop/Finance_Project/TSLA_stock_data_classification.csv'
data = pd.read_csv(file_path)

# Clean the data
data_cleaned = data.drop(columns=['Unnamed: 0', 'Date'])
data_cleaned['Price_Change'] = data_cleaned['Price_Change'].map({'Yes': 1, 'No': 0})

# Convert 'Date' to datetime format for better plotting
data['Date'] = pd.to_datetime(data['Date'])
data_cleaned['Date'] = data['Date']
data_cleaned.set_index('Date', inplace=True)

# Prepare the prediction model
X = data_cleaned.drop(columns=['Price_Change'])
y = data_cleaned['Price_Change']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        open_price = float(request.form['open_price'])
        high_price = float(request.form['high_price'])
        low_price = float(request.form['low_price'])
        close_price = float(request.form['close_price'])
        adj_close_price = float(request.form['adj_close_price'])
        volume = float(request.form['volume'])

        # Prepare input for prediction
        user_input = np.array([[open_price, high_price, low_price, close_price, adj_close_price, volume]])
        prediction = rf_classifier.predict(user_input)

        result = "Significant change (Yes)" if prediction[0] == 1 else "No significant change (No)"
        flash(f'Prediction Result: {result}', 'success')
    except ValueError:
        flash('Input Error: Please enter valid numerical values.', 'danger')
    
    return redirect(url_for('prediction'))

# Prediction Route (HTML Form)
@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

# Visualization Route
@app.route('/visualization', methods=['GET', 'POST'])
def visualization():
    if request.method == 'POST':
        x_axis = request.form['x_axis']
        y_axis = request.form['y_axis']
        graph_type = request.form['graph_type']
        start_year = int(request.form['start_year'])
        end_year = int(request.form['end_year'])
        
        img_base64 = generate_visualization(x_axis, y_axis, graph_type, start_year, end_year)
        
        if img_base64:
            return render_template('visualization.html', img_data=img_base64)
        else:
            flash('Error: Could not generate the image.', 'danger')
            return redirect(url_for('visualization'))

    return render_template('visualization_form.html')

# Function to generate the graph and return the base64 string
def generate_visualization(x_axis, y_axis, graph_type, start_year, end_year):
    img_base64 = None

    try:
        plt.clf()
        data_filtered = data_cleaned[(data_cleaned.index.year >= start_year) & (data_cleaned.index.year <= end_year)]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        if graph_type == 'scatter':
            sns.scatterplot(data=data_filtered, x=x_axis, y=y_axis, hue='Price_Change', palette='Set1', ax=ax)
            ax.set_title(f'{y_axis} vs {x_axis} (Scatter Plot)')
        elif graph_type == 'line':
            ax.plot(data_filtered.index, data_filtered[x_axis], label=x_axis, color='blue')
            ax.plot(data_filtered.index, data_filtered[y_axis], label=y_axis, color='orange')
            ax.set_title(f'{y_axis} vs {x_axis} (Line Plot)')
            ax.legend()
        elif graph_type == 'histogram':
            ax.hist(data_filtered[x_axis], bins=30, color='blue', alpha=0.7)
            ax.set_title(f'Distribution of {x_axis} (Histogram)')
            ax.set_xlabel(x_axis)
            ax.set_ylabel('Frequency')
        elif graph_type == 'boxplot':
            sns.boxplot(data=data_filtered[x_axis], ax=ax, color='blue')
            ax.set_title(f'Boxplot of {x_axis}')
            ax.set_ylabel(x_axis)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"Error generating the graph: {e}")

    return img_base64

if __name__ == '__main__':
    app.run(debug=True)
