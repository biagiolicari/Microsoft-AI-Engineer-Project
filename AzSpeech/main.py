import os
from dotenv import load_dotenv

from synthesizer import Synthesizer
from voice import Voice

def main():
    load_dotenv()

    KEY = os.getenv('Key')
    ENDPOINT = os.getenv('Endpoint')
    REGION = os.getenv('Region')

    stt = Voice(REGION, KEY)
    tts = Synthesizer(REGION, KEY)

    mystt = stt.recognize_from_microphone(KEY, REGION, ENDPOINT)
    print(mystt)

    mytts = tts.synthesizer("Test, one, two three", "en-US-AvaMultilingualNeural", 'en-US')


if __name__ == '__main__':
    main()