from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import cv2

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

app = Flask(__name__)
model = Model()  # Instantiate your PyTorch model here
model.load_state_dict(torch.load('saved_modelcyclonet.pth'))  # Load the model's weights

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Initialize img variable
    img = None
    
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # Check if the image file exists
    if imagefile:
        # Preprocess the input image
        img = cv2.imread(image_path)
        img = np.array(img)
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)  # Change channel order to match PyTorch convention (C, H, W)
        img = img / 255.0  # Normalize to [0, 1] range

        # Resize the image
        resize = transforms.Resize(size=(224, 224))
        img = resize(img)

        # Add batch dimension
        img = img.unsqueeze(0)

    # Perform inference if img is not None
    if img is not None:
        with torch.no_grad():
            output = model(img)
        
        # Extract the float value from the prediction
        prediction = output.item()
    else:
        # Handle case where image file is not uploaded
        prediction = "No image file uploaded!"

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
