import json
import joblib
import numpy as np
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# ✅ Load ML model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "ml_models", "fraud_model.pkl")
scaler_path = os.path.join(BASE_DIR, "ml_models", "scaler.pkl")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Fraud model and scaler loaded successfully!")
except Exception as e:
    print("❌ Failed to load fraud model:", e)
    model = None
    scaler = None


@csrf_exempt
def fraud_predict(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode('utf-8'))
            features = np.array(data["features"]).reshape(1, -1)

            scaled = scaler.transform(features)
            prediction = model.predict(scaled)[0]

            result = "fraud" if prediction == 1 else "legit"
            return JsonResponse({"prediction": result})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return JsonResponse({"message": "Only POST method allowed"}, status=405)
