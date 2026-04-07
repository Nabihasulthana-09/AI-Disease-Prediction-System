from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    fever = int(request.form["fever"])
    cough = int(request.form["cough"])
    headache = int(request.form["headache"])
    fatigue = int(request.form["fatigue"])
    nausea = int(request.form["nausea"])
    vomiting = int(request.form["vomiting"])
    sore_throat = int(request.form["sore_throat"])
    body_pain = int(request.form["body_pain"])
    dizziness = int(request.form["dizziness"])
    cold = int(request.form["cold"])

    symptoms = np.array([[
        fever, cough, headache, fatigue,
        nausea, vomiting, sore_throat,
        body_pain, dizziness, cold
    ]])

    result = model.predict(symptoms)

    return render_template("index.html", prediction=result[0])

if __name__ == "__main__":
    app.run(debug=True)
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)