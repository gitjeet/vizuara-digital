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
from PIL import Image


app = Flask(__name__)
CORS(app)
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