# Capuchinbird Audio Classification Using Deep Learning

This repository contains a project for classifying audio clips to determine whether they contain the sound of a Capuchinbird or not. The project leverages deep learning techniques to process audio signals, convert them into spectrograms, and train a Convolutional Neural Network (CNN) to classify the audio clips.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)

## Introduction

Capuchinbird Audio Classification is crucial for applications in bioacoustics research and environmental monitoring. The project aims to develop a machine learning model that can accurately identify Capuchinbird sounds from audio clips. The model uses audio preprocessing and deep learning to achieve this classification.

## Dataset

The dataset used in this project is from Kaggle's Z by HP Unlocked Challenge 3 - Signal Processing. The dataset is preprocessed to extract waveforms and convert them into spectrograms, which are then used as input for the deep learning model.

- [Kaggle's Z by HP Unlocked Challenge 3 - Signal Processing.](https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing)

- Extract the dataset files and place them inside the data/ folder for processing.

## Installation

To run this project, you need to have Python installed on your machine. You can install the required dependencies using `pip`.

```
pip install tensorflow==2.8.0 tensorflow-gpu==2.8.0 tensorflow-io matplotlib



```

Requirements
Python 3.x
TensorFlow
TensorFlow I/O
Matplotlib

## Usage

1. Clone the repository to your local machine:

```
   git clone https://github.com/srijosh/Capuchinbird-Audio-Classification-Using-Deep-Learning.git
```

2. Navigate to the project directory:
   cd Capuchinbird-Audio-Classification-Using-Deep-Learning

3. Open and run the Jupyter Notebook:
   - jupyter notebook AudioClassifier.ipynb

## Model

The model used in this project is a Convolutional Neural Network (CNN) with the following architecture:

- Convolutional Layers: Extracts features from the spectrograms.
- Dense Layer: Classifies the audio clips into two categories (Capuchinbird sound or not).

### Data Preprocessing

- Waveform to Spectrogram Conversion: The raw audio files are read, and waveforms are converted to spectrograms to visualize frequency changes over time.

### Model Training

The model is compiled and trained with:

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Metrics: Recall and Precision for a comprehensive evaluation of classification performance.

### Evaluation

The model's performance is evaluated using the following metrics:

- Binary Crossentropy Loss: Measures the model's performance in classifying the two classes.
- Precision: Indicates how many of the predicted positive results are true positives.
- Recall: Shows how well the model identifies true positive instances.
