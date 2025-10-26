from rest_framework import serializers

class SpamMessageSerializer(serializers.Serializer):
    message = serializers.CharField(max_length=500)
