
# Dogs vs Cats Classification using ResNet and Transfer Learning

## Overview
This project classifies images of cats and dogs using the **Dogs vs Cats** dataset from Kaggle. It leverages **ResNet50** via TensorFlow Hub for transfer learning, significantly improving training efficiency and accuracy. The model is trained and evaluated using TensorFlow and Keras.

## Dataset
Dataset used: Dogs vs Cats

- **Training Data**: 25,000 labeled images of cats and dogs.
- **Test Data**: 12,500 unlabeled images.

## Project Structure
```
├── data/
│   ├── train/  # Training images (cats & dogs)
│   ├── test1/  # Test images (unlabeled)
├── models/
│   ├── resnet_model.h5  # Saved ResNet model
├── notebooks/
│   ├── dogs_vs_cats_resnet.ipynb  # Jupyter Notebook with training & evaluation
├── README.md
├── requirements.txt  # Dependencies
```

## Installation
To install required dependencies, run:
```sh
pip install -r requirements.txt
```

## Data Preprocessing
- Extract images from ZIP files.
- Resize images to **224x224** for compatibility with ResNet50.
- Normalize pixel values.
- Split into **training (80%)** and **testing (20%)** sets.

## Model Architecture
The model uses **ResNet50** as a feature extractor with fully connected dense layers and a softmax output for binary classification (dog vs. cat).

## Training
- The model is compiled using **Adam optimizer** and **sparse categorical crossentropy loss**.
- Trained for **5 epochs** with validation.
- Achieved high accuracy on both training and validation datasets.

## Evaluation
The model achieves **96.5% accuracy** on the test set.
Evaluated using standard classification metrics.

## Running the Notebook
To train and test the model, open the Jupyter notebook and run all cells:
```sh
jupyter notebook notebooks/dogs_vs_cats_resnet.ipynb
```

## Future Improvements
- Experiment with **EfficientNet** for better accuracy.
- Implement **data augmentation** to improve generalization.
- Fine-tune the **ResNet50** layers.

## Author
Developed by **[Ali Ure]**  
Contact: **ureali90@gmail.com**

