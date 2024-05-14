import os
import azure.cognitiveservices.speech as speech_sdk

from synthesizer import Synthesizer

class Voice:
    def __init__(self, location_speech, key_speech) -> None:
        self.location_speech = location_speech
        self.key_speech = key_speech
        #self.stt = stt

        self.speech_config = None
        self.speech_synthesizer = None

        self.command = ''

    '''def transcribe_command(self, speech_recognition_language):
        LoadSynthesizer = Synthesizer(os.getenv('Region').location_speech, os.getenv('Key'))

        # Configure speech service
        speech_config = speech_sdk.SpeechConfig(self.key_speech, self.location_speech, speech_recognition_language=speech_recognition_language)
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

        return command'''
    
    def recognize_from_microphone(speech_recognition_language, azure_key, azure_location, azure_endpoint):
        try:
            speech_config = speech_sdk.SpeechConfig(subscription=azure_key, region=azure_location)
            speech_config.speech_recognition_language = speech_recognition_language

            audio_config = speech_sdk.audio.AudioConfig(use_default_microphone=True)

            auto_detect_source_language_config = \
                speech_sdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["en-US", "de-DE", "it-IT", "fr-FR"])
            speech_recognizer = speech_sdk.SpeechRecognizer(
                speech_config=speech_config,
                auto_detect_source_language_config=auto_detect_source_language_config,
                audio_config=audio_config)

            print("Speak into your microphone.")
            speech_recognition_result = speech_recognizer.recognize_once()
            if speech_recognition_result.reason == speech_sdk.ResultReason.RecognizedSpeech:
                auto_detect_source_language_result = speech_sdk.AutoDetectSourceLanguageResult(speech_recognition_result)
                detected_language = auto_detect_source_language_result.language
                return speech_recognition_result.text, detected_language
            else:
                print("Speech recognition failed:", speech_recognition_result.reason)
        except Exception as e:
            print("An error occurred during speech recognition:", e)
