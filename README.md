# Sign Language Detection

## Overview
Sign Language Detection is a machine learning project designed to recognize and classify hand gestures in real-time using a webcam. The system uses a trained model to interpret gestures representing different letters of the alphabet (A-Z). This tool aims to assist in bridging communication gaps for individuals using sign language.

## Features
- **Real-Time Gesture Detection**: Detects hand gestures in real-time via webcam.
- **Customizable Dataset**: Collect and preprocess your own sign language data.
- **Machine Learning Model**: Trains and evaluates a Random Forest classifier for gesture classification.
- **Interactive Interface**: Displays the predicted gesture on the screen during live detection.

## Technologies Used
- Python
- OpenCV
- Mediapipe
- Scikit-learn
- NumPy
- Matplotlib

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- Pip

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sign-language-detection.git
   cd sign-language-detection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify that your webcam is functioning properly.

---

## Project Structure
```plaintext
├── data/                  # Directory for collected sign language images
├── collect_data.py        # Script for data collection
├── create_dataset.py      # Script for preprocessing collected data
├── train_model.py         # Script for training the model
├── main.py                # Script for real-time detection
├── model.p                # Trained model (generated after training)
├── data.pickle            # Preprocessed data (generated after preprocessing)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## How to Use

### Step 1: Data Collection
1. Run the `collect_data.py` script to collect images of hand gestures for different classes:
   ```bash
   python collect_data.py
   ```
2. Follow the prompts to record gestures for each class (e.g., letters A-Z).

### Step 2: Preprocessing the Dataset
1. Run the `create_dataset.py` script to extract hand landmarks and preprocess the data:
   ```bash
   python create_dataset.py
   ```
2. This will generate a `data.pickle` file containing the processed data.

### Step 3: Training the Model
1. Run the `train_model.py` script to train a machine learning model:
   ```bash
   python train_model.py
   ```
2. The trained model will be saved as `model.p`.

### Step 4: Real-Time Detection
1. Run the `main.py` script to start real-time detection:
   ```bash
   python main.py
   ```
2. The webcam feed will open, and the detected gesture will be displayed on the screen.

---

## Key Components

### 1. Data Collection (`collect_data.py`)
- Collects images of hand gestures for each class using a webcam.
- Saves images in a structured directory for easy processing.

### 2. Dataset Preprocessing (`create_dataset.py`)
- Uses Mediapipe to extract hand landmarks from images.
- Normalizes the coordinates and saves the processed data in a `data.pickle` file.

### 3. Model Training (`train_model.py`)
- Loads the preprocessed dataset.
- Trains a Random Forest classifier to classify gestures.
- Evaluates the model and saves it for future use.

### 4. Real-Time Detection (`main.py`)
- Loads the trained model and uses the webcam to detect gestures in real-time.
- Displays the detected class on the video feed.

---

## Requirements

List of dependencies (found in `requirements.txt`):
- OpenCV
- Mediapipe
- Scikit-learn
- NumPy
- Matplotlib

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Results
- Achieved **high accuracy** in detecting and classifying hand gestures.
- Provided smooth and real-time gesture recognition with minimal latency.

---

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch-name`).
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments
- **Mediapipe**: For efficient hand landmark detection.
- **Scikit-learn**: For providing powerful machine learning tools.
- Community resources and tutorials that inspired the project.

---

## Contact
For any questions or suggestions, please contact:
- **Name**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [YourUsername](https://github.com/yourusername)

