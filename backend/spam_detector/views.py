from rest_framework.views import APIView
from rest_framework.response import Response
import pickle
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'ml_models', 'spam_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'ml_models', 'vectorizer.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

class SpamPredictAPIView(APIView):
    def post(self, request):
        message = request.data.get("message", "")
        cleaned = clean_text(message)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        return Response({"message": message, "prediction": prediction})
