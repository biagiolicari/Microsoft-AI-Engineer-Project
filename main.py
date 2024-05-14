import os

from dotenv import load_dotenv

# from azure_speech.synthesizer import Synthesizer
# from azure_speech.voice import Voice
# from gpt import OpenAI
from large_language_model.bot_gpt import BotAgent


def main():
    load_dotenv()

    KEY = os.getenv('Key')
    ENDPOINT = os.getenv('Endpoint')
    REGION = os.getenv('Region')
    MODEL = os.getenv('OPENAI_MODEL')
    KEY_LLM = os.getenv('KEY')
    ENDPOINT_LLM = os.getenv('ENDPOINT')

    # stt = Voice(REGION, KEY)
    # tts = Synthesizer(REGION, KEY)
    llm = BotAgent(KEY_LLM, ENDPOINT_LLM, MODEL)

    # print(stt.transcribe_command(speech_recognition_language='it-IT'))
    # tts = tts.synthesizer("Test, one, two three", "en-US-AvaMultilingualNeural", 'en-US')
    llm.create_system_prompt("")
    print(llm.chat("hello"))


if __name__ == '__main__':
    main()
