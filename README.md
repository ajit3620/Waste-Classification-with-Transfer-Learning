# Waste Classification with Transfer Learning


---

## Overview

This project builds an image classifier to distinguish nine types of waste using transfer learning on pre-trained convolutional neural networks (CNNs): ResNet50, ResNet101, EfficientNetB0, and VGG16.

## Dataset

The dataset is organized as follows:
```
data/
  RealWaste/
    1-Cardboard/
    2-Food Organics/
    3-Glass/
    4-Metal/
    5-Miscellaneous Trash/
    6-Paper/
    7-Plastic/
    8-Textile Trash/
    9-Vegetation/
```
- **Training/Test Split**: 80% of images per class for training, 20% for testing.  
- **Validation Set**: 20% of the training set, selected per class.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/<username>/waste-classification.git
    cd waste-classification
    ```
2. **Create a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare the data**: Place the `RealWaste` folder under the `data/` directory.
2. **Run the Jupyter notebook**:
    ```bash
    jupyter notebook
    ```
3. Execute the cells in the following order:
   - Data exploration and preprocessing
   - Data augmentation and generator setup
   - Model building and training
   - Evaluation and plotting

## Model Training Details

- **Transfer Learning**: Freeze all layers of the pre-trained model except the final classification head.  
- **Data Augmentation**: Random cropping, zoom, rotation, flipping, contrast adjustment, and translation.  
- **Model Head**: Fully connected layer with ReLU activation, L2 regularization, batch normalization, dropout (20%), and softmax output.  
- **Optimizer & Loss**: Adam optimizer with categorical crossentropy.  
- **Training Strategy**: Batch size of 5, train for 50–100 epochs, early stopping on validation loss (patience = 10), and checkpoint saving for best weights.

## Evaluation Metrics

After training, report the following on both validation and test sets:
- Precision
- Recall
- F1-score
- ROC AUC (one-vs-rest for multiclass)

Sample results (replace with actual numbers):

| Model          | Test Accuracy | Macro F1 | Macro AUC |
| -------------- | ------------- | -------- | --------- |
| ResNet50       | XX%           | X.XX     | X.XX      |
| ResNet101      | XX%           | X.XX     | X.XX      |
| EfficientNetB0 | XX%           | X.XX     | X.XX      |
| VGG16          | XX%           | X.XX     | X.XX      |

Plots of training/validation loss and accuracy per epoch are saved in the `figures/` directory.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

---

## References

- Keras Documentation: https://keras.io
- Transfer Learning Overview: https://builtin.com/data-science/transfer-learning
- Batch Normalization: https://en.wikipedia.org/wiki/Batch_normalization
