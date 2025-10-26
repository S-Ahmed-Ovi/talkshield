# spam_detector/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import SpamMessageSerializer

class SpamPredictAPIView(APIView):
    def post(self, request):
        serializer = SpamMessageSerializer(data=request.data)
        if serializer.is_valid():
            message = serializer.validated_data['message']

            # Placeholder for ML model prediction
            prediction = "spam" if "offer" in message.lower() else "ham"

            return Response({"message": message, "prediction": prediction})
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
