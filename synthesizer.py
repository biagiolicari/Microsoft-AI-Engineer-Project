import os
from dotenv import load_dotenv

import azure.cognitiveservices.speech as speech_sdk

class Synthesizer:
    def __init__(self, location_speech, key_speech, tts) -> None:
        self.location_speech = location_speech
        self.key_speech = key_speech
        self.tts = tts

        self.speech_config = None
        self.speech_synthesizer = None

    def synthesizer(self, response_text):

        speech_config = speech_sdk.SpeechConfig(self.key_speech, self.location_speech, speech_recognition_language='it-IT')
        speech_config.speech_synthesis_voice_name = "it-IT-ElsaNeural"
        speech_synthesizer = speech_sdk.SpeechSynthesizer(speech_config)

        # Synthesize spoken output
        responseSsml = " \
            <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='it-IT'> \
                <voice name='it-IT-ElsaNeural'> \
                    {} \
                    <break strength='weak'/> \
                </voice> \
            </speak>".format(response_text)
        
        speak = speech_synthesizer.speak_ssml_async(responseSsml).get()

        if speak.reason != speech_sdk.ResultReason.SynthesizingAudioCompleted:
            print(speak.reason)
