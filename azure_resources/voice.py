import azure.cognitiveservices.speech as speech_sdk


class Voice:
    def __init__(self, location_speech, key_speech) -> None:
        self.location_speech = location_speech
        self.key_speech = key_speech

        self.speech_config = speech_sdk.SpeechConfig(subscription=self.key_speech, region=self.location_speech)
        self.speech_synthesizer = None

        self.command = ''

    def transcribe_command(self):
        audio_config = speech_sdk.audio.AudioConfig(use_default_microphone=True)
        auto_detect_source_language_config = speech_sdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=["en-US", "de-DE", "it-IT", "fr-FR"]
            )
        speech_recognizer = speech_sdk.SpeechRecognizer(
                speech_config=self.speech_config,
                auto_detect_source_language_config=auto_detect_source_language_config,
                audio_config=audio_config
            )

        speech_recognition_result = speech_recognizer.recognize_once()
        if speech_recognition_result.reason == speech_sdk.ResultReason.RecognizedSpeech:
            auto_detect_source_language_result = speech_sdk.AutoDetectSourceLanguageResult(speech_recognition_result)
            detected_language = auto_detect_source_language_result.language
            return speech_recognition_result.text, detected_language
        else:
            raise Exception(speech_recognition_result.reason)
