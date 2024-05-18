import requests
import http.client, urllib.request, urllib.parse, urllib.error, base64


class OCR:
    def __init__(self, endpoint, key):
        self.endpoint = endpoint
        self.key = key

    def ImageOCR(self, image, detect_language):

        path = '/vision/v3.2/ocr'

        url = self.endpoint + path

        headers = {
            'Content-Type': 'application/octet-stream',
            'Ocp-Apim-Subscription-Key': self.key
        }

        params = urllib.parse.urlencode({
            'language': detect_language,
            'detectOrientation': 'true',
            'model-version': 'latest',
        })

        response = requests.post(url, headers=headers, params=params, data=image.tobytes())

        #pprint.pprint(response.json())
        result = ""

        if response.status_code == 200:
            print('ok')
            
            resp = response.json()['regions']

            if len(resp) > 0:
                print(len(resp[0]['lines']))

                for i in range(len(resp[0]['lines'])):
                    for j in range(len(resp[0]['lines'][i]['words'])):
                        result = result + " " + resp[0]['lines'][i]['words'][j]['text']
        
        return result