import azure.cognitiveservices.speech as speech_sdk


class Synthesizer:
    def __init__(self, location_speech, key_speech) -> None:
        self.location_speech = location_speech
        self.key_speech = key_speech

        self.speech_config = None
        self.speech_synthesizer = None

    def synthesizer(self, response_text, synthesis_voice_name, speech_recognition_language):

        speech_config = speech_sdk.SpeechConfig(self.key_speech, self.location_speech,
                                                speech_recognition_language=speech_recognition_language)
        speech_config.speech_synthesis_voice_name = synthesis_voice_name
        speech_synthesizer = speech_sdk.SpeechSynthesizer(speech_config)

        speak = speech_synthesizer.speak_text_async(response_text).get()

        if speak.reason == speech_sdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized for text [{}]".format(response_text))
        elif speak.reason == speech_sdk.ResultReason.Canceled:
            cancellation_details = speak.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speech_sdk.CancellationReason.Error:
                if cancellation_details.error_details:
                    print("Error details: {}".format(cancellation_details.error_details))
                    print("Did you set the speech resource key and region values?")
