# Explainable AI (XAI) Methods for Reinforcement of Object Recognition and Localization in the Wild Egocentric Videos

This project explores Explainable AI (XAI) methods to enhance object recognition and localization in egocentric videos, focusing on wild and dynamic environments. It aims to improve the interpretability and reliability of object detection models through various explainability techniques, which can be used to reinforce object recognition and localization in real-time video streams.

## Overview

The project applies three key explainability methods to evaluate and analyze object recognition and localization:
- **Object Recognition**: Detects and identifies objects in egocentric (first-person) video frames.
- **Localization**: Finds the position of detected objects in the frames.
- **Explainability**: Provides insights into which features contribute the most to model decisions.

## Folder Structure

Each folder in this project corresponds to a specific task or component. Please follow the README files located within each folder for detailed instructions on how to run the respective parts of the project.

1. **Folder 1**: Data Preparation & Preprocessing
   - Contains scripts for data loading, augmentation, and preprocessing.
   
2. **Folder 2**: Model Training
   - Includes the model architecture, training scripts, and configuration files for object recognition.
   
3. **Folder 3**: Explainability Methods
   - Contains the implementation of various XAI methods used to analyze the model's predictions.
   
4. **Folder 4**: Evaluation & Results
   - Holds scripts for model evaluation and visualization of results, including saliency maps and performance metrics.

## Running the Project

Each folder comes with its own detailed README explaining the steps to run the project. Please ensure you follow the instructions in each folder for the respective task:
- **Data Preparation & Preprocessing**: Prepare your dataset before model training.
- **Model Training**: Train the object detection model using the preprocessed data.
- **Explainability Methods**: Apply different XAI techniques to interpret the results.
- **Evaluation & Results**: Evaluate the model’s performance using metrics and visualize the results.

## Future Work

This project is part of ongoing research to refine object recognition and localization in dynamic, real-world settings. Future improvements include:
- Enhancing the integration of XAI methods to improve model accuracy and decision-making.
- Exploring new explainability techniques for better model transparency and validation.

## Dependencies

The following libraries are required to run the project:

- Python 3.x
- PyTorch
- OpenCV
- Matplotlib
- NumPy

To install dependencies, run:

```bash
pip install -r requirements.txt
