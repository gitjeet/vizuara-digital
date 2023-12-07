from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
app = Flask(__name__)

df = pd.read_csv('./train.csv')
dataset = pd.read_csv('./train.csv')
features_X = list(df.columns)
embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
print(features_X)
X = df[features_X]
numeric_cols = X.select_dtypes(include=['number']).columns
missing_cols= numeric_cols
dataset["Gender"] = dataset["Gender"].map({'male': 0, 'female': 1})
X_final = dataset[['Gender', 'Pclass', 'Fare', 'Embarked', 'Age']]

y_final = dataset['Survived']


X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(X_final, y_final, test_size=0.1, random_state=0)
print(X_final.head())
criterion_ = 'gini'
max_ = 'sqrt'
decision_tree_model = DecisionTreeClassifier(criterion=criterion_, random_state=0, max_features=max_)

decision_tree_model.fit(X_final_train, y_final_train)

# Calculate accuracy on the test set
y_pred = decision_tree_model.predict(X_final_test)
acc = accuracy_score(y_final_test, y_pred)
print(f"Accuracy on test set: {acc}")

decision_tree_model.fit(X_final_train, y_final_train)
if len(missing_cols) and len(X):
    missing_strategy = 'mean'

    execute_title_extraction = False


    if missing_strategy == 'mean':
        si = SimpleImputer(strategy='mean')
        X[missing_cols] = si.fit_transform(X[missing_cols])
        df=X
@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        request_data = request.json  # Get JSON data from POST request
        
        # Read the CSV file locally or from the provided file path in the request
        
        # Get feature columns from the POST request object
        feature_columns = request_data.get('features')
        
        
        X_data = df[feature_columns]
        y_data = df['Survived']
        categorical_cols = X_data.select_dtypes(include=['object']).columns
        d_size = request_data.get('test_size')
        # Apply one-hot encoding to these categorical columns
        X_data = pd.get_dummies(X_data, columns=categorical_cols, drop_first=True)
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=float(d_size), random_state=0)
        
        # Train the Decision Tree model
        decision_tree_model = DecisionTreeClassifier(criterion='gini', random_state=0, max_features='log2')
        decision_tree_model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = decision_tree_model.predict(X_test)
        
        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        
        # Prepare the response
        if acc > 0.7:
            accuracy_response = 'High'
        elif acc > 0.6:
            accuracy_response = 'Medium'
        elif acc > 0.5:
            accuracy_response = 'Low'
        else:
            accuracy_response = 'Very Low'
        
        response = {
            'accuracy': acc,
            'accuracy_level': accuracy_response
        }
        
        return jsonify(response), 200  # Return accuracy in JSON format with status code 200
    except Exception as e:
        error_response = {'error': str(e)}
        return jsonify(error_response), 500  # Return error message with status code 500 if an exception occurs

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Mapping for categorical features
        embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
        gender_mapping = {'male': 0, 'female': 1}  # Mapping for 'Gender' column
        
        # Prepare data for prediction
        custom_data = pd.DataFrame({
            'Gender': [data['Gender']],
            'Pclass': [data['Pclass']],
            'Age': [data['Age']],
            'Embarked': [data['Embarked']],
            'Fare': [data['Fare']]
        })

        # Transform categorical columns
        custom_data['Embarked'] = custom_data['Embarked'].map(embarked_mapping)
        custom_data['Gender'] = custom_data['Gender'].map(gender_mapping)  # Apply gender mapping

        # Apply one-hot encoding to categorical features
        categorical_cols = custom_data.select_dtypes(include=['object']).columns
        custom_data_encoded = pd.get_dummies(custom_data, columns=categorical_cols, drop_first=True)

        # Reorder columns to match training data
        custom_data_encoded = custom_data_encoded[X_final_train.columns]

        # Make predictions using the trained model
        custom_predictions = decision_tree_model.predict(custom_data_encoded)
        
        # Return prediction result
        return jsonify({'prediction': int(custom_predictions[0])}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500





if __name__ == '__main__':
    app.run(debug=True)
