import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# ✅ Load full pipeline (TfidfVectorizer + Model)
classmodel = pickle.load(open("models/best_f1_spam_model.pkl", "rb"))

# Label mapping
label_map = {0: "Not Spam", 1: "Spam"}


@app.route("/")
def home():
    """Render the main homepage."""
    return render_template("home.html")


@app.route("/predict_json", methods=["POST"])
def predict_json():
    """Async prediction route for frontend (AJAX)."""
    message = request.json.get("message", "").strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        # ✅ Predict directly with pipeline
        output = classmodel.predict([message])[0]
        prediction_label = label_map.get(output, str(output))

        # Confidence (if supported)
        confidence = None
        if hasattr(classmodel, "predict_proba"):
            proba = classmodel.predict_proba([message])
            confidence = round(np.max(proba[0]) * 100, 2)

        return jsonify({
            "prediction": prediction_label,
            "confidence": f"{confidence}%" if confidence else None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ping", methods=["GET"])
def ping():
    """Health check route."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
