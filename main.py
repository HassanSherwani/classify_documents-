#1- Import key modules
from flask import Flask, render_template, url_for, request
import pickle
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfTransformer

# 2- Start script
def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt

app = Flask(__name__)
# load saved models
filename = 'models/linearsvc_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
filename2 = 'models/tfidf.sav'
tfidf = pickle.load(open(filename2, 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        data=pd.DataFrame(data, columns=["body_text"])
        data['tidy_text'] = np.vectorize(remove_pattern)(data['body_text'],"@[\w]*")
        vect = pd.DataFrame(tfidf.transform(data.tidy_text).toarray())
        predict_class = loaded_model.predict(vect)
        score=loaded_model._predict_proba_lr(vect)
        prob=score.max(axis=1)

        # using dataframe to jsonify end point
        result=data
        result=pd.DataFrame(result,columns=["newsheadline"])
        result["news-type"]=predict_class
        result["probability"]=prob
        json_table = result.to_json(orient='records')
    return app.response_class(
        response=json_table,
        status=200,
        mimetype='application/json'
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
