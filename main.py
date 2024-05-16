import cv2
import time
import os




from cnn_detection.facenet_detection import FaceDetectionRecognition
from azure_speech.synthesizer import Synthesizer
from azure_speech.voice import Voice 
from large_language_model.bot_gpt import BotAgent
from cnn_detection.facenet_detection import FaceDetectionRecognition
from azure_detect.azurefacedetetct import FaceDetectionRecognitionAzure

from dotenv import load_dotenv

load_dotenv()



def main():

    Key_multiservice= os.getenv("key_multiservice")
    endpoint_multiservice= os.getenv("endpoint_multiservice")
    region_multiservice= os.getenv("region_multiservice")

    key_openai= os.getenv("key_openai")
    endpoint_openai= os.getenv("endpoint_openai")
    model_openai= os.getenv("model_openai")

    system_prompt= os.getenv("prompt_system")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    
    bot = BotAgent(key_openai, endpoint_openai, model_openai)
    voice = Voice(region_multiservice,Key_multiservice)
    synthesizer = Synthesizer(region_multiservice,Key_multiservice)
    

    nfaces, img_rgb = count_faces(cap,Key_multiservice,endpoint_multiservice)

    if nfaces == 1:
        synthesizer.synthesizer(f"Ciao! Ho visto che sei solo, come posso aiutarti?  ", "it-IT-ElsaNeural", "it-IT")
    else:
        synthesizer.synthesizer(f"Ciao! Ho visto che siete in {nfaces} , come posso aiutarvi? ", "it-IT-ElsaNeural", "it-IT")

    bot.create_system_prompt(system_prompt)

    while True:
        text,detetc_language=voice.transcribe_command("it-IT")
        result_user = bot.chat(text)
        synthesizer.synthesizer(result_user, "it-IT-ElsaNeural", "it-IT")



    
    




def count_faces(cap,Key_multiservice,endpoint_multiservice):

    facenet = FaceDetectionRecognition()
    azurefacedetect= FaceDetectionRecognitionAzure(Key_multiservice,endpoint_multiservice)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        start_time = time.time()


        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face detection
        boxes, _, nfaces = facenet.detect_faces(img_rgb)
        print(nfaces)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate and display FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow('Webcam Face Detection', frame)

            
        

        
        print(f"volti detectati da azure: {azurefacedetect.detect_faces_azure(img_rgb)}")

        if nfaces > 0:
            return nfaces, img_rgb

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

    


    


if __name__ == "__main__":
    main()
