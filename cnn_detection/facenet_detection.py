import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np

class FaceDetectionRecognition:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.6, 0.7, 0.7], factor=0.709, min_face_size=20)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        # self.age_gender_model = torch.hub.load('ZhaoJ9014/face.evoLVe.PyTorch', 'vggface2').to(self.device)
        # self.age_gender_model.eval()

    def detect_faces(self, image):
        """
        Detect faces in an image using MTCNN.
        
        Args:
        image (PIL.Image or np.ndarray): The input image.
        
        Returns:
        boxes (np.ndarray): The bounding boxes of the detected faces.
        probs (np.ndarray): The probabilities of the detected faces.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        boxes, probs = self.mtcnn.detect(image)
        return boxes, probs, len(boxes)

    def extract_face(self, image, box):
        """
        Extract the face from the image using the bounding box.
        
        Args:
        image (PIL.Image or np.ndarray): The input image.
        box (np.ndarray): The bounding box of the face.
        
        Returns:
        face (torch.Tensor): The extracted face tensor.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        face = self.mtcnn.extract(image, box, save_path=None)
        return face
    
    # crea un metodo per verificare il numero di volti con aure vision

    def detect_faces_azure(self, image):
        pass

    '''def load_age_model(self):
        """
        Load a pre-trained age estimation model.
        
        Returns:
        model (torch.nn.Module): The age estimation model.
        """
        model = torch.hub.load('yu4u/age-gender-estimation', 'agegender').to(self.device)
        model.eval()
        return model

    def predict_age_gender(self, face):
        """
        Predict the age and gender of a face using the age-gender-estimation model.

        Args:
        face (torch.Tensor): The extracted face tensor.

        Returns:
        age (int): The predicted age of the face.
        gender (str): The predicted gender of the face ('Male' or 'Female').
        """
        face = face.to(self.device)
        age, gender = self.age_gender_model(face.unsqueeze(0))
        age = int(np.floor(age.item()))
        gender = 'Male' if gender.item() < 0.5 else 'Female'
        return age, gender'''

    '''def get_face_embedding(self, face):
        """
        Get the embedding of a face using FaceNet.
        
        Args:
        face (torch.Tensor): The extracted face tensor.
        
        Returns:
        embedding (torch.Tensor): The embedding vector of the face.
        """
        face = face.to(self.device)
        embedding = self.facenet(face.unsqueeze(0))
        return embedding'''

    def preprocess_image(self, image_path):
        """
        Load and preprocess the image.
        
        Args:
        image_path (str): The path to the input image.
        
        Returns:
        image (PIL.Image): The preprocessed image.
        """
        image = Image.open(image_path).convert('RGB')
        return image

    '''def recognize_face(self, image_path):
        """
        Detect faces and extract their embeddings from an image.
        
        Args:
        image_path (str): The path to the input image.
        
        Returns:
        face_embeddings (list of torch.Tensor): The list of face embeddings detected in the image.
        """
        image = self.preprocess_image(image_path)
        boxes, probs = self.detect_faces(image)

        face_embeddings = []
        if boxes is not None:
            for box in boxes:
                face = self.extract_face(image, box)
                if face is not None:
                    embedding = self.get_face_embedding(face)
                    face_embeddings.append(embedding)
        
        return face_embeddings'''

    '''def verify_faces(self, embedding1, embedding2, threshold=0.6):
        """
        Verify if two face embeddings are from the same person.
        
        Args:
        embedding1 (torch.Tensor): The first face embedding.
        embedding2 (torch.Tensor): The second face embedding.
        threshold (float): The threshold for verification.
        
        Returns:
        is_same (bool): True if the faces are from the same person, False otherwise.
        """
        distance = torch.dist(embedding1, embedding2).item()
        is_same = distance < threshold
        return is_same'''