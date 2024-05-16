import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from PIL import Image

from cnn_detection.cnn import CNN

class FaceDetectionCNN:
    def __init__(self, data_dir, input_size=(128, 128), batch_size=64, learning_rate=0.001, epochs=20):
        self.data_dir = data_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(self.input_size[0], scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._prepare_data()
        self._initialize_model()

    def _prepare_data(self):
        dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)

        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def _initialize_model(self):
        self.model = CNN().to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            self.scheduler.step()
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(self.train_loader)}")

            self.validate()

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                preds = (outputs.squeeze() > 0.5).float()
                correct += (preds == labels).sum().item()

        print(f"Validation Loss: {val_loss/len(self.val_loader)}, Accuracy: {correct/len(self.val_loader.dataset)*100}%")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def predict(self, image_path):
        self.model.eval()
        image = Image.open(image_path) #.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(self.device)
        output = self.model(image)
        prediction = torch.sigmoid(output).item()
        return prediction > 0.5

# Example usage
# face_detector = FaceDetectionCNN(data_dir='path_to_your_dataset')
# face_detector.train()
# face_detector.save_model('face_detection_model.pth')
# face_detector.load_model('face_detection_model.pth')
# print(face_detector.predict('path_to_new_image.jpg'))
