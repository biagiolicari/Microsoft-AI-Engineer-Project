import requests


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
