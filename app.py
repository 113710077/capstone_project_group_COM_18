from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("rf_model.pkl", "rb"))

FEATURES = ["API_MIN", "API", "vt_detection", "VT_Malware_Deteccao", "AZ_Malware_Deteccao"]

# Homepage
@app.route("/")
def home():
    return render_template("index.html")

# Manual prediction
@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    values = []

    # Safely convert form input to float
    for f in FEATURES:
        val = request.form.get(f, "")
        val = float(val) if val != "" else 0.0
        values.append(val)

    prediction = int(model.predict([values])[0])
    prob = float(max(model.predict_proba([values])[0])) * 100

    prob_percent = round(prob, 2)

    result = "SAFE" if prediction == 0 else "MALICIOUS"
    color = "green" if result == "SAFE" else "red"

    # Cap probability values just in case
    if prob_percent < 0: prob_percent = 0
    if prob_percent > 100: prob_percent = 100

    return render_template(
        "manual_result.html",
        result=result,
        prob_percent=prob_percent,
        color=color
    )

# CSV Prediction
@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    file = request.files.get("csv_file")

    if file is None or file.filename == "":
        return "No file uploaded"

    df = pd.read_csv(file)

    # Ensure required columns exist
    try:
        df = df[FEATURES]
    except KeyError:
        return "CSV missing required columns", 400

    # Drop rows with missing data
    df = df.dropna(subset=FEATURES)

    preds = model.predict(df)
    probs = model.predict_proba(df)

    results = []
    for i in range(len(preds)):
        label = "SAFE" if preds[i] == 0 else "MALICIOUS"
        prob_percent = round(float(max(probs[i])) * 100, 2)
        color = "green" if label == "SAFE" else "red"

        # cap probability
        if prob_percent < 0: prob_percent = 0
        if prob_percent > 100: prob_percent = 100

        results.append({
            "id": i + 1,
            "label": label,
            "prob": prob_percent,
            "color": color
        })

    return render_template("csv_result.html", results=results)

if __name__ == "__main__":
    app.run(debug=True, port=7001)
