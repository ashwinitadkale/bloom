from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get numeric values
    age = float(request.form.get("age"))
    cycle_length = float(request.form.get("cycle_length"))
    last_period_day = float(request.form.get("last_period_day"))

    # Get categorical values
    mood = request.form.get("mood")
    flow = request.form.get("flow")
    symptom = request.form.get("symptom")

    # Encoding maps (must match the order used during training)
    mood_map = {"happy": 0, "sad": 1, "angry": 2, "tired": 3, "neutral": 4}
    flow_map = {"light": 0, "medium": 1, "heavy": 2}
    symptom_map = {"cramps": 0, "headache": 1, "fatigue": 2, "bloating": 3, "nausea": 4}

    mood_encoded = mood_map.get(mood, -1)
    flow_encoded = flow_map.get(flow, -1)
    symptom_encoded = symptom_map.get(symptom, -1)

    # Combine into feature list
    features = [
        age,
        cycle_length,
        last_period_day,
        mood_encoded,
        flow_encoded,
        symptom_encoded,
    ]

    # Fill the remaining features (since the model was trained on 9 features)
    while len(features) < 9:
        features.append(0)

    prediction = model.predict([features])[0]

    return render_template(
        "index.html", prediction=f"Next period in {int(prediction)} days"
    )


if __name__ == "__main__":
    app.run(debug=True)
