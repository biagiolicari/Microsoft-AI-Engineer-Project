
import requests
from pprint import pprint




class OpenAI:
    def __init__(self, key, ep, model) -> None:
        self.key = key
        self.ep = ep
        self.model = model
        self.messages = {}

    def create_promt(self, prompt):
        self.messages["role"]="system"
        self.messages["content"]=prompt


    def chat(self, prompt):
        headers= {
            "Content-Type":"application/json",
            "api-key":self.key
        }



        prompt= {
            "messages":[ 
                self.messages,
                {"role": "user", 
                "content": prompt}
            ]
        }
        
        url=f"{self.ep}/openai/deployments/{self.model}/chat/completions?api-version=2024-02-01"
        res= requests.post(url,headers=headers,json=prompt)
        print(f"{res.status_code}: {res.reason}")

        json=res.json()
        pprint(json)

        '''print("---------------------------")
        result=json["choices"][0]["message"]["content"]
        print(result)
        print("---------------------------")'''

        return json

