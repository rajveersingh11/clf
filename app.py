from flask import Flask,request, jsonify
import joblib
import numpy as np 

model= joblib.load("insurance_model.pkl")

app=Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        values = data.get("input")
        if values is None:
            raise ValueError("Request JSON must contain an 'input' field.")

        cols = ["age", "bmi", "children", "sex", "smoker", "region"]
        import pandas as pd
        input_df = pd.DataFrame([values], columns=cols)

        prediction = model.predict(input_df)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__=="__main__":
    app.run(debug=True)