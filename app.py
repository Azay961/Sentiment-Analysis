from flask import Flask, render_template, request
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import joblib

app = Flask(__name__)

def preprocess(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text) 
        text = re.sub(r'\d+', '', text)
        text = text.lower()
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        clean_text = [word for word in word_tokens if word not in stop_words]

        stemmer = PorterStemmer()
        clean_text = [stemmer.stem(word) for word in clean_text]

        return clean_text
    else:
        return ''

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def predict_sentiment():
    input_text = request.form['input_text']
    selected_model = request.form['model_dropdown']
    model = joblib.load(f"/models/{selected_model}.pkl")
    
    clean_text = preprocess(input_text)
    clean_text = ' '.join(clean_text)
    
    sentiment = model.predict([clean_text])
    sentiment_label = "Positive" if sentiment[0] == 1 else "Negative"
    
    return render_template('index.html', sentiment_label=sentiment_label, input_text=input_text, selected_model=selected_model)

    app.run(debug=True)
    # host="0.0.0.0", port="5002"