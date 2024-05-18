import os

import cv2
from dotenv import load_dotenv

from azure_detect.azurefacedetect import FaceDetectionRecognitionAzure
from azure_speech.synthesizer import Synthesizer
from azure_speech.voice import Voice
from azure_translate.translate import Translate
from azure_ocr.ocr import OCR
from cnn_detection.facenet_detection import FaceDetectionRecognition
from large_language_model.bot_gpt import BotAgent


def load_env_variables(required_vars):
    load_dotenv()
    env_vars = {var: os.getenv(var) for var in required_vars}

    for key, value in env_vars.items():
        if value is None:
            raise EnvironmentError(f"Environment variable {key} is missing.")

    return env_vars


def setup_camera(width=320, height=240):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise Exception("Error: Could not open webcam.")
    return cap


def initialize_services(env_vars):
    bot = BotAgent(env_vars["key_openai"], env_vars["endpoint_openai"], env_vars["model_openai"])
    voice = Voice(env_vars["region_multiservice"], env_vars["key_multiservice"])
    synthesizer = Synthesizer(env_vars["region_multiservice"], env_vars["key_multiservice"])
    translate = Translate(env_vars["region_multiservice"], env_vars["key_multiservice"],
                          'https://api.cognitive.microsofttranslator.com/')
    azure_face_detect = FaceDetectionRecognitionAzure(env_vars["key_multiservice"],
                                                      env_vars["endpoint_multiservice"])
    ocr = OCR(env_vars["endpoint_multiservice"], env_vars["key_multiservice"])
    return bot, voice, synthesizer, translate, azure_face_detect, ocr


def detect_faces_azure(img_rgb, azurefacedetect):
    azure_faces = azurefacedetect.detect_faces_azure(img_rgb)
    return azure_faces


def detect_faces_facenet(frame, facenet):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, confidences, nfaces = facenet.detect_faces(img_rgb)
    return boxes, confidences, nfaces, img_rgb

def text_from_image(ocr, img_rgb, detected_language):
    return ocr.ImageOCR(img_rgb, detected_language)

def count_faces(cap, batch_size=30):
    facenet = FaceDetectionRecognition()
    frame_buffer = []
    max_frames = batch_size

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            raise Exception("Error: Failed to capture image.")

        # FaceNet detection
        boxes, confidences, nfaces, img_rgb = detect_faces_facenet(frame, facenet)

        # Draw FaceNet detected boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Webcam Face Detection', frame)

        if nfaces > 0:
            frame_buffer.append((nfaces, img_rgb, confidences))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if not frame_buffer:
        return 0, None

    # Select the best frame based on highest number of faces and highest confidence
    best_frame = max(frame_buffer, key=lambda x: (x[0], max(x[2])))
    return best_frame[0], best_frame[1]


def interact_with_user(bot, voice, synthesizer, translate, nfaces):
    if nfaces == 1:
        synthesizer.synthesizer("Ciao! Ho visto che sei solo, come posso aiutarti?", "it-IT")
    else:
        synthesizer.synthesizer(f"Ciao! Ho visto che siete in {nfaces}, come posso aiutarvi?", "it-IT")

    while True:
        try:
            text, detected_lang = voice.transcribe_command()
            if not text:
                return True  # Restart face detection if no speech is detected
            elif text.lower() == translate.translate("esci.", detected_lang[:2]):
                synthesizer.synthesizer(translate.translate("A presto!", detected_lang[:2]), detected_lang)
                return False  # Exit application
            else:
                response = bot.chat(text)
                synthesizer.synthesizer(response, detected_lang)
        except Exception as e:
            print(f"Error: {e}")
            if 'detected_lang' in locals():
                synthesizer.synthesizer(
                    translate.translate("Mi dispiace, non ho capito. Riprova per favore.", detected_lang[:2]),
                    detected_lang)


def main():
    required_vars = ["key_multiservice", "endpoint_multiservice", "region_multiservice",
                     "key_openai", "endpoint_openai", "model_openai", "prompt_system"]

    try:
        env_vars = load_env_variables(required_vars=required_vars)
        cap = setup_camera()

        bot, voice, synthesizer, translate, azure_face_detect, ocr = initialize_services(env_vars)
        bot.create_system_prompt(env_vars["prompt_system"])

        while True:
            nfaces, img_rgb = count_faces(cap)
            if nfaces > 0:
                # Perform Azure face detection for comparison
                azure_faces = detect_faces_azure(img_rgb, azure_face_detect)

                # Compare FaceNet and Azure results
                print(f"FaceNet detected {nfaces} faces.")
                print(f"Azure detected {azure_faces} faces.")

                should_continue = interact_with_user(bot, voice, synthesizer, translate, nfaces)
                if not should_continue:
                    break

    except Exception as e:
        print(f"Fatal Error: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
