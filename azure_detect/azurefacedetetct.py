import requests
import os
from dotenv import load_dotenv
import cv2

load_dotenv()

class FaceDetectionRecognitionAzure:
    def __init__(self,key,endpoint):
        self.endpoint = endpoint
        self.subscription_key = key

    def detect_faces_azure(self, image):

        url = f"{self.endpoint}/face/v1.0/detect"
    
        headers = {
            "Content-Type": "application/octet-stream",
            "Ocp-Apim-Subscription-Key": self.subscription_key
        }
    
        params = {
            "returnFaceId": 'false',
            "returnFaceLandmarks": "true",
            "detectionModel" : "detection_03"
        }

        ret, my_jpg = cv2.imencode(".jpg", image)

        response = requests.post(url, headers=headers, params=params, data=my_jpg.tobytes())
        #print(response)

        if response.status_code == 200:
            try:
                faces = response.json()
                return len(faces)

            except ValueError:
                print("Error")
            