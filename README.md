# Cyclone Detection Web App

This is a web application for detecting cyclones using a deep learning model. The application is built using Flask and PyTorch.

## Features

- Allows users to upload an image of a cyclone.
- Utilizes a deep learning model to predict whether the uploaded image contains a cyclone.
- Displays the prediction result on the web page.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/akhil99558/Cyclone
    ```

2. Navigate to the project directory:

    ```bash
    cd cyclone-detection-web-app
    ```

3. Install the required Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Download the pre-trained PyTorch model and place it in the project directory.

## Usage

1. Start the Flask server:

    ```bash
    python app.py
    ```

2. Open a web browser and go to `http://localhost:3000`.

3. Upload an image of a cyclone using the provided form.

4. Wait for the prediction result to be displayed on the web page.

## Model

The cyclone detection model is based on a custom convolutional neural network implemented using PyTorch.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

