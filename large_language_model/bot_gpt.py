import requests
import base64
from mimetypes import guess_type


class BotAgent:
    def __init__(self, key, ep, model) -> None:
        self.key = key
        self.ep = ep
        self.model = model
        self.messages = {}

    def create_system_prompt(self, prompt) -> None:
        self.messages["role"] = "system"
        self.messages["content"] = prompt

    def chat(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "api-key": self.key
        }
        url = f"{self.ep}/openai/deployments/{self.model}/chat/completions?api-version=2024-02-01"

        user_input = {"role": "user", "content": prompt}
        payload = {"messages": [self.messages, user_input]}

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            response_json = response.json()

            if "choices" in response_json and len(response_json["choices"]) > 0:
                return response_json["choices"][0]["message"]["content"]
            else:
                return None
        except requests.RequestException as e:
            # Handle request exceptions (e.g., network issues)
            print(f"Request exception: {e}")
            return None
    
    def local_image_to_data_url(image_path):
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"
        
    def food_analyzer(self, prompt,image_jpg,image_path):
        headers= {
            "Content-Type": "application/json",
            "api-key": self.key
        }
        
        url= f"{self.ep}/openai/deployments/{self.model}/chat/completions?api-version=2024-02-01"

        base64_image= base64.b64encode(image_jpg).decode("utf-8")

        user_input= {"role": "user", "content": [{"type":"text","text": prompt},
                                                 {"type":"image_url","image_url":{"url":self.local_image_to_data_url(image_path)}}]}

        print(user_input)
        payload = {"messages": [self.messages, user_input]}

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()

            if "choices" in response_json and len(response_json["choices"]) > 0:
                return response_json["choices"][0]["message"]["content"]
            
            else:
                return None
        
        except requests.RequestException as e:
            print(f"Request exception: {e}")
            return None