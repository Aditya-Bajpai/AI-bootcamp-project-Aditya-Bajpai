from flask import Flask , render_template , request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("diabetes_classifier.pkl" , "rb"))

@app.route("/")
def main():
    return render_template("main.html")

@app.route("/predict" , methods = ["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template("main.html",text = "The chances of you having diabetes is {}".format(prediction[0]))

