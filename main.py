from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle


app = Flask(__name__)

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
filename2 = 'countvectorizer.sav'
cv = pickle.load(open(filename2, 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = pd.DataFrame(cv.transform(data).toarray())
        my_prediction = loaded_model.predict(vect)
        score=loaded_model.predict_proba(vect)
        prob=score.max(axis=1)

        # using dataframe
        result=data
        result=pd.DataFrame(result,columns=["text"])
        result["news-type"]=my_prediction
        result["news-type"]=result['news-type'].map({1 : "b", 2 : "t", 3 :"e", 4:"m"})
        result["probability"]=prob
        json_table = result.to_json(orient='records')
    return app.response_class(
        response=json_table,
        status=200,
        mimetype='application/json'
    )


if __name__ == '__main__':
    app.run()
