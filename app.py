# app.py
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

model = joblib.load("models/model.joblib")
with open("models/meta.json") as f:
    meta = json.load(f)


@app.route("/")
def home():
    return render_template(
        "index.html",
        locations=meta["locations"],
        area_types=meta["area_types"],
        metrics=meta["metrics"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        sqft = float(data["total_sqft"])
        bhk = int(data["bhk"])
        bath = int(data["bath"])
        location = str(data["location"]).strip()
        area_type = str(data["area_type"]).strip()

        if not (300 <= sqft <= 10000):
            return jsonify(error="Area must be between 300 and 10,000 sq ft."), 400
        if not (1 <= bhk <= 10) or not (1 <= bath <= 12):
            return jsonify(error="Please enter valid BHK and bathroom counts."), 400
        if sqft / bhk < 300:
            return jsonify(error="Area is too small for that many bedrooms."), 400
        if location not in meta["locations"]:
            location = "other"
        if area_type not in meta["area_types"]:
            return jsonify(error="Invalid area type."), 400

        X = pd.DataFrame(
            [{"area_type": area_type, "location": location,
              "total_sqft": sqft, "bath": bath, "bhk": bhk}]
        )
        price = float(model.predict(X)[0])
        margin = meta["metrics"]["mdape"]

        return jsonify(
            price=round(price, 2),
            low=round(price * (1 - margin), 2),
            high=round(price * (1 + margin), 2),
            per_sqft=round(price * 100000 / sqft),
        )
    except (KeyError, ValueError, TypeError):
        return jsonify(error="Invalid input. Please check all fields."), 400
    except Exception:
        return jsonify(error="Something went wrong. Please try again."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)