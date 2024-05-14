import os
from dotenv import load_dotenv

#from AzSpeech.synthesizer import Synthesizer
#from AzSpeech.voice import Voice
#from gpt import OpenAI
from gpt import *

def main():
    load_dotenv()

    KEY = os.getenv('Key')
    ENDPOINT = os.getenv('Endpoint')
    REGION = os.getenv('Region')
    MODEL = os.getenv('OPENAI_MODEL')
    KEY_LLM = os.getenv('KEY')
    ENDPOINT_LLM = os.getenv('ENDPOINT')


    #stt = Voice(REGION, KEY)
    #tts = Synthesizer(REGION, KEY)
    llm= OpenAI(KEY_LLM,ENDPOINT_LLM,MODEL)

    #print(stt.transcribe_command(speech_recognition_language='it-IT'))
    #tts = tts.synthesizer("Test, one, two three", "en-US-AvaMultilingualNeural", 'en-US')
    llm.create_promt("Hello, how are you?")
    llm.chat("I'm fine, thank you!")


if __name__ == '__main__':
    main()