from flask import Flask, request, render_template
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models and preprocessing tools
log_model = pickle.load(open('logistic_regression.pkl', 'rb'))
fast_model = pickle.load(open('fast_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
cnn_model = load_model('cnn_model.h5')

max_len = 200  # Should match padding length used during training

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    url = ""
    if request.method == 'POST':
        url = request.form['url']
        
        # Logistic Regression prediction
        vec = vectorizer.transform([url])
        log_pred = 'Malicious' if log_model.predict(vec)[0] == 1 else 'Safe'
        
        # Fast model prediction
        fast_pred = 'Malicious' if fast_model.predict(vec)[0] == 1 else 'Safe'
        
        # CNN prediction
        seq = tokenizer.texts_to_sequences([url])
        padded = pad_sequences(seq, maxlen=max_len, padding='post')
        cnn_output = cnn_model.predict(padded)[0][0]
        cnn_pred = 'Malicious' if cnn_output > 0.5 else 'Safe'

        result = {
            'url': url,
            'cnn': cnn_pred,
            'log': log_pred,
            'fast': fast_pred
        }
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
