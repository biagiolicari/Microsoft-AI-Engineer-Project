import random

import azure.cognitiveservices.speech as speech_sdk


def generate_ssml(response_text, voice_name, lang):
    # Choose a random style for express-as
    styles = ["cheerful", "empathetic", "calm", 'advertisement_upbeat', 'affectionate', 'assistant', 'customerservice',
              'embarrassed', 'friendly', 'hopeful', 'whispering', 'unfriendly']
    selected_style = random.choice(styles)

    # Choose a random role (example roles: "default", "youngAdultFemale", "seniorMale")
    roles = ["default", "youngAdultFemale", "seniorMale", 'Boy', 'SeniorFemale']
    selected_role = random.choice(roles)

    # Set a random style degree between 0.5 and 1.5 for variation
    styledegree = round(random.uniform(0.5, 1.5), 2)

    # Generate SSML text with prosody adjustment, express-as styles, style degree, and role
    ssml = f"""
            <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='{lang}'>
                <voice name='{voice_name}'>
                    <prosody rate="medium">
                        <mstts:express-as style="{selected_style}" styledegree="{styledegree}" role="{selected_role}">
                            {response_text}
                        </mstts:express-as>
                    </prosody>
                </voice>
            </speak>
            """
    return ssml


class Synthesizer:
    map_neural_voice_synthesizer = {
        'de-DE': 'de-DE-KatjaNeural',
        'en-US': 'en-US-AvaNeural',
        'fr-FR': 'fr-FR-DeniseNeural',
        'it-IT': 'it-IT-ElsaNeural'
    }

    def __init__(self, location_speech, key_speech) -> None:
        self.location_speech = location_speech
        self.key_speech = key_speech

        self.speech_config = None
        self.speech_synthesizer = None

    def synthesizer(self, response_text, speech_recognition_language):
        speech_config = speech_sdk.SpeechConfig(subscription=self.key_speech, region=self.location_speech)
        speech_config.speech_synthesis_voice_name = self.map_neural_voice_synthesizer.get(speech_recognition_language)

        speech_synthesizer = speech_sdk.SpeechSynthesizer(speech_config=speech_config)

        # Generate SSML text
        ssml_text = generate_ssml(response_text,
                                  self.map_neural_voice_synthesizer.get(speech_recognition_language),
                                  speech_recognition_language)

        speak = speech_synthesizer.speak_ssml_async(ssml_text).get()

        if speak.reason == speech_sdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Speech synthesized for text [{response_text}]")
        elif speak.reason == speech_sdk.ResultReason.Canceled:
            cancellation_details = speak.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speech_sdk.CancellationReason.Error:
                if cancellation_details.error_details:
                    print(f"Error details: {cancellation_details.error_details}")
                    print("Did you set the speech resource key and region values?")

    def get_synthesis_voice_name_from_speech_language_bcp_code(self, speech_language_bcp_code):
        return self.map_neural_voice_synthesizer.get(speech_language_bcp_code, 'it-IT')
