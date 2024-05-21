import requests
import http.client, urllib.request, urllib.parse, urllib.error, base64

class NLP:
    def __init__(self, endpoint, key):
        self.endpoint = endpoint
        self.key = key

    def conversational_language_understanding(self, project_name, deployment_name, text):
        path = '/language/:analyze-conversations?api-version=2022-10-01-preview'

        url = self.endpoint + path

        headers = {
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': self.key
        }

        body = {
            "kind": "Conversation",
            "analysisInput": {
                "conversationItem": {
                "id": "1",
                "participantId": "1",
                "text": text
                }
            },
            "parameters": {
                "projectName": project_name,
                "deploymentName": deployment_name,
                "stringIndexType": "TextElement_V8"
            }
        }

        response = requests.post(url, headers=headers, json=body)
        print(response)

        if response.status_code == 200:
            print('ok')
            print(response.json())

            category = response.json()["result"]["prediction"]["intents"][0]["category"]
            score = response.json()["result"]["prediction"]["intents"][0]["confidenceScore"]

            return category, score
        