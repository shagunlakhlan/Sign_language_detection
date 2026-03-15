# Sign Language Detection System

## Overview

This project implements a **real-time Sign Language Detection system** that recognizes hand gestures using a webcam and converts them into sign language alphabets. The system uses computer vision and deep learning techniques to detect and classify hand gestures.

The model is trained using the **ASL Alphabet Dataset** and performs real-time prediction through webcam input.

---

## Features

* Real-time hand gesture detection
* Sign language alphabet recognition
* Webcam-based prediction
* Deep learning model trained on image dataset
* Easy to run and extend for future improvements

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* NumPy

---

## Dataset

The model is trained on the **ASL Alphabet Dataset**, which contains images representing American Sign Language alphabets.

Classes included in the dataset:

* A – Z alphabets
* space
* delete
* nothing

---

## Project Structure

```
sign_lang
│
├── dataset
│   ├── A
│   ├── B
│   ├── C
│   └── ...
│
├── models
│   └── sign_model.h5
│
├── train_model.py
├── detect_gui.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/yourusername/sign-language-detection.git
cd sign-language-detection
```

### 2. Create virtual environment

```
python -m venv venv
```

### 3. Activate the environment

Windows:

```
venv\Scripts\activate
```

### 4. Install dependencies

```
pip install -r requirements.txt
```

---

## Train the Model

Run the training script:

```
python train_model.py
```

This will train the deep learning model and save it in the **models** folder.

---

## Run the Detection System

After training, run:

```
python detect_gui.py
```

The webcam will open and the system will detect sign language gestures in real time.

---

## Future Improvements

* Sentence formation from detected letters
* Voice output for detected text
* Support for more sign language gestures
* Improved graphical interface

---

## Author

Shagun
B.Tech Computer Science (Artificial Intelligence)
