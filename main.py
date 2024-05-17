import os
import time
import cv2
from dotenv import load_dotenv
from azure_detect.azurefacedetect import FaceDetectionRecognitionAzure
from azure_speech.synthesizer import Synthesizer
from azure_speech.voice import Voice
from azure_translate.translate import Translate
from cnn_detection.facenet_detection import FaceDetectionRecognition
from large_language_model.bot_gpt import BotAgent

load_dotenv()

def main():
    Key_multiservice = os.getenv("key_multiservice")
    endpoint_multiservice = os.getenv("endpoint_multiservice")
    region_multiservice = os.getenv("region_multiservice")

    key_openai = os.getenv("key_openai")
    endpoint_openai = os.getenv("endpoint_openai")
    model_openai = os.getenv("model_openai")

    system_prompt = os.getenv("prompt_system")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    bot = BotAgent(key_openai, endpoint_openai, model_openai)
    voice = Voice(region_multiservice, Key_multiservice)
    synthesizer = Synthesizer(region_multiservice, Key_multiservice)
    translate = Translate(region_multiservice, Key_multiservice, 'https://api.cognitive.microsofttranslator.com/')

    nfaces, img_rgb = count_faces(cap, Key_multiservice, endpoint_multiservice)

    if nfaces == 1:
        synthesizer.synthesizer(f"Ciao! Ho visto che sei solo, come posso aiutarti?", "it-IT")
    else:
        synthesizer.synthesizer(f"Ciao! Ho visto che siete in {nfaces}, come posso aiutarvi?", "it-IT")

    bot.create_system_prompt(system_prompt)

    while True:
        try:
            text, detected_lang = voice.transcribe_command()
            print(detected_lang)
            print(detected_lang[0:2])
            if not text:
                break

            elif text.lower() == translate.translate("esci.", detected_lang[0:2]):
                synthesizer.synthesizer(translate.translate("A presto!", detected_lang[0:2]), detected_lang)
                break

            else:
                result_user = bot.chat(text)
                synthesizer.synthesizer(result_user, detected_lang)
        
        except Exception as e:
            print(f"Error: {e}")
            synthesizer.synthesizer(translate.translate("Mi dispiace, non ho capito. Riprova per favore.", detected_lang[0:2]), detected_lang)

def count_faces(cap, Key_multiservice, endpoint_multiservice):
    facenet = FaceDetectionRecognition()
    azurefacedetect = FaceDetectionRecognitionAzure(Key_multiservice, endpoint_multiservice)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        start_time = time.time()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face detection
        boxes, _, nfaces = facenet.detect_faces(img_rgb)
        
        print(f"Volti rilevati da FaceNet: {nfaces}")

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            nfaces = 0
            print("No faces detected.")

        # Calculate and display FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow('Webcam Face Detection', frame)

        print(f"Volti rilevati da Azure: {azurefacedetect.detect_faces_azure(img_rgb)}")

        if nfaces > 0:
            return nfaces, img_rgb

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
    return 0, None  # Ensure returning valid values if loop exits without detection

if __name__ == "__main__":
    main()
