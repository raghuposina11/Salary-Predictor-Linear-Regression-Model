from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

# Load model
model_path = os.path.join("model", "salary_model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_salary():
    data = request.get_json()
    years_exp = data.get("YearsExperience") if data else None

    if years_exp is None:
        return jsonify({"error": "YearsExperience is required"}), 400

    prediction = model.predict([[years_exp]])[0]
    return jsonify({"predicted_salary": round(prediction, 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
