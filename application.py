from flask import Flask, render_template, request
import joblib, numpy as np

app = Flask(__name__)

model  = joblib.load("artifacts/models/model.pkl")
scaler = joblib.load("artifacts/processed/scaler.pkl")

# simple mapping from your dropdown values → numeric codes
TREATMENT_MAP = {
    "surgery":   0,
    "chemo":     1,
    "radiation": 2
}

@app.route('/')
def home():
    return render_template("index.html", prediction=None)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # numeric inputs
        costs          = float(request.form["healthcare_costs"])
        tumor          = float(request.form["tumor_size"])
        mortality      = float(request.form["mortality_rate"])

        # map treatment string → integer code
        treatment_str  = request.form["treatment_type"]                # e.g. "surgery"
        treatment_code = TREATMENT_MAP.get(treatment_str, 0)           # default to 0 if somehow missing

        # checkbox: if checked you get "1", otherwise key is missing → default 0
        diabetes_code  = int(request.form.get("diabetes", 0))

        # build your feature vector in the same order your model expects
        features = np.array([[costs, tumor, treatment_code, diabetes_code, mortality]])
        scaled   = scaler.transform(features)

        pred = model.predict(scaled)[0]
        return render_template("index.html", prediction=pred)

    except Exception as e:
        # for real apps you’d log this
        return f"Error processing input: {e}", 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
