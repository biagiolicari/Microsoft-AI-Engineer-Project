import requests
import base64
from mimetypes import guess_type
from open import AzureOpenAI


class BotAgent:
    def __init__(self, key, ep, model, vision_model) -> None:
        self.key = key
        self.vision_model=vision_model
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

    def local_image_to_data_url(self, image_path):
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    def chat_with_vision(self, image_path):
        client = AzureOpenAI(
            api_key=self.key,
            api_version='2023-12-01-preview',
            base_url=f"{self.ep}/openai/deployments/{self.vision_model}"
        )

        response = client.chat.completions.create(
            model=self.vision_model,
            messages=[
                self.messages,
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze the following image and describe the food item depicted. Provide the following "
                            "information:"
                            "1. A description of what you see in the image. "
                            "2. Whether the food item is suitable for consumption. "
                            "3. Additional information about the food item if it is suitable. "
                            "If the image does not show a food item, inform the user that no relevant food item was "
                            "detected.")
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self.local_image_to_data_url(image_path=image_path),
                            "detail": "low"
                        }
                    }
                ]}
            ],
            max_tokens=250,
        )
        return response.choices[0].message.content
