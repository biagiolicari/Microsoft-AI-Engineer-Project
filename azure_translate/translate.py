import uuid

import requests


class Translate:
    def __init__(self, location_translate, key_translate, endpoint_translate) -> None:
        self.location_translate = location_translate
        self.key_translate = key_translate
        self.endpoint_translate = endpoint_translate

    def translate(self, text, detect_language):
        path = 'translate'
        constructed_url = self.endpoint_translate + path
        # print(constructed_url)

        params = {
            'api-version': '3.0',
            'from': detect_language,
            'to': 'it'
        }

        headers = {
            'Ocp-Apim-Subscription-Key': self.key_translate,
            # location required if you're using a multi-service or regional (not global) resource.
            'Ocp-Apim-Subscription-Region': self.location_translate,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

        # You can pass more than one object in body.
        body = [{
            'text': text
        }]

        request = requests.post(constructed_url, params=params, headers=headers, json=body)
        response = request.json()

        res = response[0]["translations"][0]["text"]
        return res
