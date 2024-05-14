import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speech_sdk

from synthesizer import Synthesizer

class Voice:
    def __init__(self, location_speech, key_speech, stt) -> None:
        self.location_speech = location_speech
        self.key_speech = key_speech
        self.stt = stt

        self.speech_config = None
        self.speech_synthesizer = None

        self.command = ''

    def transcribe_command(self):
        load_dotenv()
        LoadSynthesizer = Synthesizer(
                location_speech = os.getenv("LOCATION_SPEECH"), 
                key_speech = os.getenv("KEY_SPEECH"), 
                tts = os.getenv("TTS")
            )

        # Configure speech service
        speech_config = speech_sdk.SpeechConfig(self.key_speech, self.location_speech, speech_recognition_language='it-IT')
        #print('Ready to use speech service in:', speech_config.region)

        # Configure speech recognition
        audio_config = speech_sdk.AudioConfig(use_default_microphone=True)
        speech_recognizer = speech_sdk.SpeechRecognizer(speech_config, audio_config)
        print('\nParla ora...')
        LoadSynthesizer.synthesizer(response_text="Parla ora...")


        # Process speech input
        speech = speech_recognizer.recognize_once_async().get()
        if speech.reason == speech_sdk.ResultReason.RecognizedSpeech:
            command = speech.text
            print(command)
        else:
            print(speech.reason)
            if speech.reason == speech_sdk.ResultReason.Canceled:
                cancellation = speech.cancellation_details
                print(cancellation.reason)
                print(cancellation.error_details)

        return command
