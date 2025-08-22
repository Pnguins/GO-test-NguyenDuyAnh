# Fine-Tuning VGG16 for Image Classification

This project provides a complete and reusable workflow for fine-tuning a VGG16 model for binary image classification using TensorFlow and Keras. The code is structured in an object-oriented manner to be easily configurable and extensible.

## Features
-   **Transfer Learning:** Leverages the pre-trained VGG16 model for high accuracy with smaller datasets.
-   **Data Augmentation:** Uses Keras's `ImageDataGenerator` to artificially expand the training dataset and prevent overfitting.
-   **Comprehensive Evaluation:** Automatically generates and saves:
    -   Training vs. Validation Accuracy/Loss plots.
    -   A detailed Confusion Matrix for the validation set.
-   **Clear Separation:** Includes separate, easy-to-use scripts for both training (`trainer.py`) and inference (`inference.py`).
-   **Configurable:** All parameters like file paths, batch size, and learning rate are managed in a central configuration dictionary.

## Getting Started

Follow these instructions to set up the project and run the training and inference processes.

### 1. Prerequisites
-   Python 3.11+
-   `pip` package manager
-   Finetuned model: https://drive.google.com/file/d/1afz0dz3AyVjSKWImH5X-cC4qOq6yfx5W/view?usp=sharing

### 2. Installation
First, clone the repository and navigate into the project directory:
```bash
git clone https://github.com/Pnguins/GO-test-NguyenDuyAnh.git
cd GO-test-NguyenDuyAnh
```

Next, install all the required Python libraries using the requirements.txt file:
```bash
pip install -r requirements.txt
```
Then, download the finetuned model from the drive link mention in **1. Prerequisites**
3. Dataset Preparation
For the scripts to work correctly, your image dataset must be organized in the following directory structure. The class names (e.g., cats, dogs) are automatically inferred from the folder names.
```code
/input
│
├── train/
│   ├── cats/
│   │   ├── image_a_1.jpg
│   │   └── image_a_2.jpg
│   └── dogs/
│       ├── image_b_1.jpg
│       └── image_b_2.jpg
│
└── test/ (or validation/)
    ├── cats/
    │   ├── image_a_val_1.jpg
    │   └── image_a_val_2.jpg
    └── dogs/
        ├── image_b_val_1.jpg
        └── image_b_val_2.jpg
```
## Usage

### Training the Model

The training process is handled by the `trainer.py` script.

**Step 1: Configure Training Parameters**

Open `trainer.py` and modify the values in the `training_config` dictionary according to the table below.


| Variable Name | File Location | Description |
| :--- | :--- | :--- |
| `PATH_TO_TRAIN_FOLDER` | `trainer.py` | **(Required)** Path to your training dataset folder. |
| `PATH_TO_TEST_FOLDER` | `trainer.py` | **(Required)** Path to your validation/test dataset folder. |
| `epochs` | `trainer.py` | (Optional) The number of times to iterate over the entire dataset. Default is `25`. |
| `batch_size` | `trainer.py` | (Optional) The number of images to process in one batch. Default is `32`. |
| `learning_rate`| `trainer.py` | (Optional) The step size for the optimizer. Default is `0.0001`. |


**Step 2: Run the Training Script**

Execute the script from your terminal. The training process will begin, showing the progress for each epoch.

```bash
python trainer.py
```
## Inference with a Trained Model
- The inference process is handled by the inference.py script.
**Step 1: Configure Inference Parameters** 
Open inference.py and modify the following variables as described in the table.
| Variable Name |	File Location |	Description |
| :--- | :--- | :--- |
| `FINETUNED_MODEL_PATH` |	`inference.py` |	**(Required)** Path to your saved .h5 model file generated during training. |
| `IMAGE_URL` |	`inference.py` |	**(Required)** The full path to the new image you want to classify. |

**Step 2: Run the Inference Script**
Execute the script from your terminal. It will load the model and print the predicted class.

```bash
python inference.py
```
