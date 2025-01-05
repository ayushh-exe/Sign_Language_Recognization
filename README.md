# Sign Language Detection System

This repository provides a complete implementation of a sign language detection system. It captures hand gestures using a webcam, preprocesses the data, trains a machine learning model, and performs real-time gesture recognition.

## Features
- *Data Collection:* Collects images of hand gestures for various sign language classes using a webcam.
- *Preprocessing:* Processes collected data to extract hand landmarks using Mediapipe.
- *Model Training:* Trains a Random Forest Classifier to recognize gestures.
- *Real-Time Detection:* Uses the trained model for live gesture detection via webcam.

---

## Directory Structure
plaintext
|-- data/                # Directory for storing collected hand gesture data
|-- collect_data.py      # Script for collecting hand gesture data
|-- create_dataset.py    # Script for preprocessing data and creating dataset
|-- train_model.py       # Script for training the sign language detection model
|-- main.py              # Script for real-time gesture detection
|-- model.p              # Trained machine learning model
|-- README.md            # Project documentation


---

## Installation

1. Clone the repository:
   bash
   git clone https://github.com/yourusername/sign-language-detection.git
   cd sign-language-detection
   

2. Install dependencies:
   bash
   pip install -r requirements.txt
   

3. Ensure a working webcam is connected to your computer.

---

## Usage

### 1. Collect Data
Run the collect_data.py script to capture hand gesture images for each class:
bash
python collect_data.py

Follow the on-screen instructions to collect data for each class.

### 2. Preprocess Data
Use the create_dataset.py script to preprocess the captured images and extract hand landmarks:
bash
python create_dataset.py

This will generate a data.pickle file containing features and labels.

### 3. Train the Model
Train the machine learning model using the train_model.py script:
bash
python train_model.py

This will save the trained model as model.p.

### 4. Real-Time Detection
Run the main.py script to detect gestures in real-time:
bash
python main.py

The script will display the detected sign on the video feed.

---

## Requirements
- Python 3.8+
- OpenCV
- Mediapipe
- Scikit-learn
- NumPy
- Matplotlib

Install all dependencies using:
bash
pip install -r requirements.txt


---

## How It Works
1. *Data Collection:* Captures images of hand gestures for each class.
2. *Landmark Extraction:* Extracts hand landmarks using Mediapipe and normalizes them.
3. *Model Training:* Trains a Random Forest Classifier on the extracted features.
4. *Real-Time Detection:* Uses the trained model to classify gestures in a live webcam feed.

---

## Future Improvements
- Add support for more sign languages and gestures.
- Improve model accuracy using deep learning techniques.
- Implement a graphical user interface (GUI) for easier interaction.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

---

## Acknowledgments
- [Mediapipe](https://mediapipe.dev/) for hand landmark detection.
- [OpenCV](https://opencv.org/) for real-time image processing.
- [Scikit-learn](https://scikit-learn.org/) for machine learning model training.

---

## Contact
For questions or suggestions, contact [your-email@example.com].
