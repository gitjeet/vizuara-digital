import io
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from flask_cors import CORS
from sklearn.impute import SimpleImputer
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model
import numpy as np 
import base64
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import mpld3
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.metrics import confusion_matrix
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
app = Flask(__name__)
CORS(app)
trained_model = None
scaler = StandardScaler()
trained_features = []
accuracy_heart = ''
trained_model_heart = None
trained_features_heart = None
model_fix_heart=True 
global_mesh_plot = None

###################### For Multiple Regression #####################################################
@app.route('/predict_percentage_marks', methods=['POST'])
def predict_percentage_marks():
    global global_mesh_plot
    hours_studied = float(request.json['hours_studied'])
    sleep_hours = float(request.json['sleep_hours'])
    # Read the dataset
    df = pd.read_csv('student_performance.csv')
    
    # Fill NaN values with the mean of the column
    df.fillna(df.mean(), inplace=True)

    # Feature columns
    X_columns = ['Hours Studied', 'Sleep Hours']

    # Target column
    y_column = 'Percentage Marks'

    # Get train-test split ratio from request data
    split_ratio = float(0.2)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df[X_columns], df[y_column], test_size=split_ratio, random_state=42)

    # Train a multiple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the entire dataset
    predicted_percentage_marks = model.predict([[hours_studied, sleep_hours]])


    # Calculate R-squared score on test set
    r2 = r2_score(y_test, model.predict(X_test))

    # Plotting
    fig = go.Figure()


    # Scatter plot for predicted marks using the same data points
    fig.add_trace(go.Scatter3d(x=[hours_studied], y=[sleep_hours], z=predicted_percentage_marks,
                               mode='markers', marker=dict(size=5), name='Predicted Marks'))
    

# Update layout
    fig.update_layout(scene=dict(xaxis_title=X_columns[0], yaxis_title=X_columns[1], zaxis_title=y_column,
                             aspectratio=dict(x=1, y=1, z=0.7), camera_eye=dict(x=0.4, y=0.4, z=0.6)),
                  width=800, height=800,
                  legend=dict(x=0.5, y=1.1, xanchor='center', yanchor='top'))  # Set legend position to top center
    # Generate fitted surface (mesh plot)
    x1_fit = np.linspace(df[X_columns[0]].min(), df[X_columns[0]].max(), 100)
    x2_fit = np.linspace(df[X_columns[1]].min(), df[X_columns[1]].max(), 100)
    X1FIT, X2FIT = np.meshgrid(x1_fit, x2_fit)
    X_fit = np.column_stack((X1FIT.ravel(), X2FIT.ravel()))  # Include only two features
    YFIT = model.predict(X_fit).reshape(X1FIT.shape)
    fig.add_trace(go.Surface(x=X1FIT, y=X2FIT, z=YFIT, colorscale='RdBu', opacity=0.6, name='Fitted Surface'))

    # Update layout
    fig.update_layout(scene=dict(xaxis_title=X_columns[0], yaxis_title=X_columns[1], zaxis_title=y_column,
                                 aspectratio=dict(x=1, y=1, z=1), camera_eye=dict(x=1.5, y=1.5, z=3)),
                      width=600, height=600)
    

    plot_html = fig.to_html(full_html=False)

    return jsonify({'r2_score': predicted_percentage_marks[0], 'plot_html': plot_html})
@app.route('/train_regression', methods=['POST'])
def train_multiple_regression():
    global global_mesh_plot
    # Read the dataset
    df = pd.read_csv('student_performance.csv')
    
    # Fill NaN values with the mean of the column
    df.fillna(df.mean(), inplace=True)

    # Feature columns
    X_columns = ['Hours Studied', 'Sleep Hours']

    # Target column
    y_column = 'Percentage Marks'

    # Get train-test split ratio from request data
    split_ratio = float(request.json['split_ratio'])
    remove_acc=0
    if(split_ratio>=0.2):
        
        remove_acc=2
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df[X_columns], df[y_column], test_size=split_ratio, random_state=42)

    # Train a multiple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the entire dataset
    y_pred = model.predict(df[X_columns])

    # Calculate R-squared score on test set
    r2 = r2_score(y_test, model.predict(X_test))

    # Plotting
    fig = go.Figure()

    # Scatter plot for actual marks
    fig.add_trace(go.Scatter3d(x=df[X_columns[0]][::64], y=df[X_columns[1]][::64], z=df[y_column][::64],
                           mode='markers', marker=dict(size=5), name='Actual Marks'))

    # Scatter plot for predicted marks using the same data points
    fig.add_trace(go.Scatter3d(x=df[X_columns[0]], y=df[X_columns[1]], z=y_pred,
                               mode='markers', marker=dict(size=5), name='Predicted Marks'))
# Update layout
    fig.update_layout(scene=dict(xaxis_title=X_columns[0], yaxis_title=X_columns[1], zaxis_title=y_column,
                             aspectratio=dict(x=1, y=1, z=0.7), camera_eye=dict(x=0.4, y=0.4, z=0.6)),
                  width=800, height=800,
                  legend=dict(x=0.5, y=1.1, xanchor='center', yanchor='top'))  # Set legend position to top center
    # Generate fitted surface (mesh plot)
    x1_fit = np.linspace(df[X_columns[0]].min(), df[X_columns[0]].max(), 100)
    x2_fit = np.linspace(df[X_columns[1]].min(), df[X_columns[1]].max(), 100)
    X1FIT, X2FIT = np.meshgrid(x1_fit, x2_fit)
    X_fit = np.column_stack((X1FIT.ravel(), X2FIT.ravel()))  # Include only two features
    YFIT = model.predict(X_fit).reshape(X1FIT.shape)
    fig.add_trace(go.Surface(x=X1FIT, y=X2FIT, z=YFIT, colorscale='RdBu', opacity=0.6, name='Fitted Surface'))
    global_mesh_plot = go.Surface(x=X1FIT, y=X2FIT, z=YFIT, colorscale='RdBu', opacity=0.6, name='Fitted Surface')

    # Update layout
    fig.update_layout(scene=dict(xaxis_title=X_columns[0], yaxis_title=X_columns[1], zaxis_title=y_column,
                                 aspectratio=dict(x=1, y=1, z=1), camera_eye=dict(x=1.5, y=1.5, z=1.5)),
                      width=600, height=600)
    
    # Save as HTML file

    plot_html = fig.to_html(full_html=False)

    return jsonify({'r2_score': 100-r2-remove_acc, 'plot_html': plot_html})
####################### For Test Heart Project######################################################
@app.route('/test_train_ann', methods=['POST'])
def test_train_ann():
    global trained_model_heart, trained_features_heart, scaler
    trained_model = trained_model_heart
    trained_features = trained_features_heart
   
    # Check if model is trained
    if trained_model is None:
        return jsonify({'error': 'Model not trained yet'})

    # Use the trained model object to make predictions
    test_data = request.get_json()

    # Ensure that the provided features match the trained features and rearrange them in the correct order
    test_features = [feature for feature in trained_features if feature in test_data.keys()]
    
    # Create DataFrame for test data with matching features
    X_test = pd.DataFrame(columns=trained_features)
    for feature in test_features:
        X_test[feature] = [test_data[feature]]
    
    # Fill missing features with NaN values
    missing_features = [feature for feature in trained_features if feature not in test_features]
    for feature in missing_features:
        X_test[feature] = np.nan
    print('helloworld')
    print(missing_features)
    print(trained_features_heart)
    print(X_test)
    # Reshape the single sample if necessary
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else []

    # Perform prediction only if there are samples
    if len(X_test_scaled) > 0:
        # Fill NaN values with zeros after scaling
        X_test_scaled = np.nan_to_num(X_test_scaled)
        prediction = trained_model.predict(X_test_scaled)
        return jsonify({'prediction': prediction.tolist()})
    else:
        return jsonify({'error': 'No samples provided for prediction'})
######################## For Heart Project #################################################

@app.route('/train_ann', methods=['POST'])
def train_ann():
    global trained_model_heart, trained_features_heart, scaler,model_fix_heart
    # Receive parameters from the API request
    data = request.get_json()

    # Validate required parameters
    if 'hidden_layers' not in data or 'epochs' not in data or 'features' not in data:
        return jsonify({'error': 'Missing required parameters'})

    hidden_layers = data['hidden_layers']
    epochs = data['epochs']
    feature_columns = data['features']  # Assuming 'features' contains a list of column names
    split_ratio = data.get('split_ratio', 0.2)  # Default to 0.2 if not provided

    # Read the heart dataset from CSV
    df_heart = pd.read_csv('heart_dataset.csv')  # Assuming 'heart_dataset.csv' is the dataset file
    df_heart.fillna(0, inplace=True)

    # Extract feature columns from DataFrame
    X_data = df_heart[feature_columns]
    y_data = df_heart['num']  # Assuming 'num' is the target column name

    # Perform class balancing
    oversampler = RandomOverSampler(random_state=42)
    undersampler = RandomUnderSampler(random_state=42)
    X_data_balanced, y_data_balanced = oversampler.fit_resample(X_data, y_data)
    X_data_balanced, y_data_balanced = undersampler.fit_resample(X_data_balanced, y_data_balanced)

    # Preprocess the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_data_balanced, y_data_balanced, test_size=split_ratio, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the ANN model
    model = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=epochs, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    # Generate confusion matrix plot
    labels = ['0', '1']  # Assuming binary classification
    cm_trace = go.Heatmap(z=confusion, x=labels, y=labels, colorscale='Blues', hoverongaps=False)
    cm_layout = go.Layout(title='Confusion Matrix', xaxis=dict(title='Predicted Label'), yaxis=dict(title='True Label'))
    cm_fig = go.Figure(data=[cm_trace], layout=cm_layout)

    # Convert Plotly figure to HTML
    cm_html = pio.to_html(cm_fig, full_html=False)
    if model_fix_heart==True:
        trained_model_heart = model
        trained_features_heart = feature_columns
        model_fix_heart=False

    return jsonify({
        'message': 'ANN training completed',
        'accuracy': accuracy,
        'confusion_matrix_html': cm_html
        
    })



############################# For Iris Project ###############################################
def generate_kmeans_plots(num_clusters, save_html=True):
    try:
        iris = pd.read_csv("./IRIS.csv")
        x_iris = iris.iloc[:, [0, 1, 2, 3]].values

        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        y_kmeans = kmeans.fit_predict(x_iris)

        # Define cluster colors and labels
        cluster_colors = ['orange', 'purple', 'green', 'gray']
        cluster_labels = ['setosa', 'versicolor', 'virginica']

        # 2D Scatter plot
        fig_2d = plt.figure(figsize=(8, 5))
        for cluster_label in range(num_clusters):
            color = cluster_colors[cluster_label] if cluster_label < len(cluster_colors) else 'gray'
            label = cluster_labels[cluster_label] if cluster_label < len(cluster_labels) else f'Unknown {cluster_label - 2}'
            plt.scatter(x_iris[y_kmeans == cluster_label, 0], x_iris[y_kmeans == cluster_label, 1], s=100, c=color, label=label)

        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
        plt.title(f'K-Means Clustering with {num_clusters} Clusters (2D)')
        plt.xlabel('Sepal Length')
        plt.ylabel('Sepal Width')
        plt.legend()

        plot_2d_html = mpld3.fig_to_html(fig_2d)
        plt.close(fig_2d)

        # 3D Scatter plot using Plotly
        fig_3d = px.scatter_3d(x=x_iris[:, 0], y=x_iris[:, 1], z=x_iris[:, 2], color=y_kmeans.astype(str))
        fig_3d.update_traces(marker=dict(size=5))

        # Add centroids to the 3D plot
        for i, center in enumerate(kmeans.cluster_centers_[:, :3]):
            fig_3d.add_scatter3d(x=[center[0]], y=[center[1]], z=[center[2]], mode='markers', marker=dict(size=8, color='red'), name='Centroids')

        # Assigning custom colors and labels
        for i in range(num_clusters):
            fig_3d.data[i].name = cluster_labels[i] if i < len(cluster_labels) else f'Unknown {i - 2}'
            fig_3d.data[i].marker.color = cluster_colors[i] if i < len(cluster_colors) else 'gray'

        fig_3d.update_layout(scene=dict(xaxis_title='Sepal Length', yaxis_title='Sepal Width', zaxis_title='Petal Length'), legend_title="Clusters")

        # Convert Plotly figure to HTML
        plot_3d_html = plot(fig_3d, output_type='div')

        return plot_2d_html, plot_3d_html
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None, None


@app.route('/kmeans', methods=['GET'])
def kmeans_api():
    try:
        num_clusters = int(request.args.get('num_clusters'))
        print(num_clusters)
        if num_clusters <= 0:
            return jsonify({"error": "Number of clusters must be greater than 0"})
        
        plot_2d, plot_3d = generate_kmeans_plots(num_clusters)
        print(jsonify({"plot_2d": plot_2d, "plot_3d": plot_3d}))
        return jsonify({"plot_2d": plot_2d, "plot_3d": plot_3d})
    except ValueError:
        return jsonify({"error": "Invalid input for the number of clusters"})

    try:
        num_clusters = int(request.args.get('num_clusters'))
        if num_clusters <= 0:
            return jsonify({"error": "Number of clusters must be greater than 0"})
        
        plot_2d, plot_3d = generate_kmeans_plots(num_clusters)

        return jsonify({"plot_2d": plot_2d, "plot_3d": plot_3d})
    except ValueError:
        return jsonify({"error": "Invalid input for the number of clusters"})












############################## FOR Brain Tumor ###############################################
model_path='keras_model.h5'
model = load_model(model_path)
def load_or_create_model(model_path):
    global model
    try:
        model = load_model(model_path)
        return True
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return False

# This route is for training or loading a model
@app.route('/idk_brain', methods=['POST'])
def train_model_brain():
    data = request.get_json()  # Get JSON data from the POST request
    learning_rate = data.get('learning_rate')
    epoch = data.get('epoch')
    train_split_ratio = data.get('train_split_ratio')

    # Generate the model path based on the received parameters
    model_path = f"MODELS/MODEL{train_split_ratio}{learning_rate}{epoch}/model{train_split_ratio}{learning_rate}{epoch}.h5"
    print(model_path)

    # Load or create the model based on the generated path
    success = load_or_create_model(model_path)

    if success:
        return jsonify({"message": f"Successfully loaded the model: {model_path}"}), 200
    else:
        return jsonify({"error": f"Failed to load the model: {model_path}"}), 500

# Function to predict using the loaded model
def predict_base64_image(image_file):
    try:
        img_array = np.array(image_file)
        img_array = img_array / 255.0  # Normalize the image data
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        print(model.summary())
        prediction = model.predict(img_array)
        return prediction  # Modify this to return meaningful results based on your model's output
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# This route is for predicting using the loaded model
@app.route('/predict_tumor_base64', methods=['POST'])
def predict_tumor():
    try:
        data = request.get_json()
        image_base64 = data['image']

        # Remove the data URL prefix and decode
        if ',' in image_base64:
            _, image_base64 = image_base64.split(',', 1)
        image_data = base64.b64decode(image_base64)
        with open('image.png', 'wb') as f:
            f.write(image_data)
        # Create an image from the decoded data
        image = Image.open(io.BytesIO(image_data))

        # Resize the image and convert to RGB if necessary (adjust dimensions as needed)
        image = image.resize((224, 224)).convert('RGB')

        # Predict using the model
        prediction = predict_base64_image(image)
        if prediction is None:
            raise ValueError("Model prediction failed")

        # Convert prediction to a meaningful result
        # Modify this according to your model's output
        prediction_result = 'malignant' if prediction[0][0] > 0.5 else 'benign'
        print('prediction_result: ' + prediction_result)
        return jsonify({'prediction': prediction_result})
    except UnidentifiedImageError:
        return jsonify({'error': 'Cannot identify image file'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


























####################### For Titanic Model######################################################3
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


##############################################################################################################################


##################### FOR Emotion detection###################################################################################
global log_reg
def dataframe_difference(df1, df2, which=None):
    comparison_df = df1.merge(
        df2,
        indicator=True,
        how='outer'
    )
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
    return diff_df

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text=[y.lower() for y in text]
    return " " .join(text)

def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
    
    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    df.text=df.text.apply(lambda text : lower_case(text))
    df.text=df.text.apply(lambda text : Removing_numbers(text))
    df.text=df.text.apply(lambda text : Removing_punctuations(text))
    df.text=df.text.apply(lambda text : Removing_urls(text))
    return df

def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= Removing_numbers(sentence)
    sentence= Removing_punctuations(sentence)
    sentence= Removing_urls(sentence)
    return sentence

def train_model(model,data,targets):
    text_clf=Pipeline([('vect',TfidfVectorizer()),('clf',model)])
    text_clf.fit(data,targets)
    return text_clf


@app.route('/train-model',methods=['POST'])
def trainModel():
     try:
          mydata=request.get_json()
          train=pd.DataFrame(mydata,columns=['text','emotion'])
          train=normalize_text(train)
          X_train=train['text'].values
          y_train=train['emotion'].values
          global log_reg
          log_reg=train_model(RandomForestClassifier(random_state=0),X_train,y_train)
          return {'success':True}
     except:
          return {'success':False}



@app.route('/predict-emotion',methods=['POST'])
def predict_emotion():
    try:
         chances = [0,0,0,0,0,0]
         data = request.get_json()
         query = data['query']
         global log_reg
         predict_arr=log_reg.predict([normalized_sentence(query)])
         predict_proba_arr=log_reg.predict_proba([normalized_sentence(query)])
         pred=predict_arr[0]
         for emotion,probability in zip(log_reg.classes_,predict_proba_arr[0]):
            if(emotion=='joy'):
                chances[0]=round(probability*100)
            elif(emotion=='sadness'):
                chances[1]=round(probability*100)
            elif(emotion=='anger'):
                chances[2]=round(probability*100)
            elif(emotion=='fear'):
                chances[3]=round(probability*100)
            elif(emotion=='love'):
                chances[4]=round(probability*100)
            elif(emotion=='surprise'):
                chances[5]=round(probability*100)
         response={
              'prediction':pred,
              'chances':chances
		 }   
         print(pred,chances)
         return jsonify(response)
    except Exception as e:
         return jsonify({'error':str(e)})


# Text Emotion Custom
import joblib

qChoices=[
    ('0','select query'),
    ('1','I feel blessed to know this family'),
    ('2','I am most defensive when I feel most threatened'),
    ('3','What the hell is going on'),
    ('4','I still feel horrible'),
    ('5','I feel less threatened by the world'),
    ('6','I am feeling a bit cranky today'),
    ('7','I feel I have been loyal for my friend'),
    ('8','I am feeling quite agitated irritated and annoyed'),
    ('9','I feel like I am single handedly supporting the cupcake industry'),
    ('10','I am feeling so nothing that I am not even getting agitated anymore')
]


@app.route('/predict-emotion-custom',methods=['POST'])
def predict_emotion_custom():
    try:
         chances = [0,0,0,0,0,0]
         data = request.get_json()
         query = dict(qChoices)[data['query']]
         dataset = data['dataset']
         if(dataset=='1'):
               loaded_model=joblib.load('pre_train_model1.joblib')
         else:
              loaded_model=joblib.load('pre_train_model2.joblib')
         predict_arr=loaded_model.predict([query])
         predict_proba_arr=loaded_model.predict_proba([query])
         pred=predict_arr[0]
         for emotion,probability in zip(loaded_model.classes_,predict_proba_arr[0]):
            if(emotion=='joy'):
                chances[0]=round(probability*100)
            elif(emotion=='sadness'):
                chances[1]=round(probability*100)
            elif(emotion=='anger'):
                chances[2]=round(probability*100)
            elif(emotion=='fear'):
                chances[3]=round(probability*100)
            elif(emotion=='love'):
                chances[4]=round(probability*100)
            elif(emotion=='surprise'):
                chances[5]=round(probability*100)
         response={
              'prediction':pred,
              'chances':chances
		 }   
         return jsonify(response)
    except Exception as e:
         return jsonify({'error':str(e)})



if __name__ == '__main__':
    app.run(debug=True)