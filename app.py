from flask import Flask, render_template, request
import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

# Define your neural network model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(784, 1),
        )

    def forward(self, x):
        return self.model(x)

# Instantiate your model
model = Model()

# Load the model's weights
model.load_state_dict(torch.load('saved_modelcyclonet.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define the Flask application
app = Flask(__name__)

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define route for home page
@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'imagefile' not in request.files:
            return render_template('index.html', prediction="No image uploaded!")

        # Read the image file from the POST request
        imagefile = request.files['imagefile']

        # Check if the file is empty
        if imagefile.filename == '':
            return render_template('index.html', prediction="No image selected!")

        # Save the image to a temporary location
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        # Load and preprocess the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = Image.fromarray(img)  # Convert ndarray to PIL Image
        img = transform(img)  # Apply transformations

        # Add batch dimension
        img = img.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = model(img)
        
        # Extract the predicted value
        prediction = output.item()

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
