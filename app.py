import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)
model, vect = pickle.load(open('mymodel.pkl', 'rb'))

port = int(os.getenv("PORT"))

file_ = open("name.txt", 'r')
name = file_.readlines()

@app.route('/')
def home():
    return render_template('index.html', name = name[0].upper())

@app.route('/predict',methods=['POST'])
def predict():

    xdata = [x for x in request.form.values()]
    prediction = model.predict(vect.transform(xdata))


    if prediction[0]==0:
       output = "Negative"
    else: 
       output="Positive"

    return render_template('index.html', name = name[0].upper(), input_text = 'The text you entered : {}'.format(xdata[0]), prediction_text='Predicted sentiment : {}'.format( output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True, host = '0.0.0.0', port=port)
