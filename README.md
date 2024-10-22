# CMR Report Classification Project

## Overview

This project is designed to automatically classify presence of disease in cardiovascular magnetic resonance (CMR) reports into three categories: **negative**, **positive**, and **indeterminate (neutral)**. The system is equipped to use a two-step approach:

1. A negation filter to quickly screen out negative reports.
2. A fine-tuned BERT model to classify reports that are not easily categorized by the negation filter.

Testing on our data has shown that using the negation filter reduces false positives but also significantly increases false negatives. This leads to an overall decreased F1-score when compared to a well trained BERT model alone but can improve results if a fine-tuned model is struggling with negative sample exclusion. Depending on the use case, this trade-off may or may not be desirable. Deployment of the negation filter is a toggleable option in the GUI.

The project aims to assist medical professionals by automating the process of categorizing CMR reports, making the workflow more efficient and providing a reliable basis for further review.

## Table of Contents
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [How to Download and Set Up the Model](#how-to-download-and-set-up-the-model)
- [Using the Application](#using-the-application)
  - [Step 1: Running the GUI](#step-1-running-the-gui)
  - [Step 2: Processing Reports](#step-2-processing-reports)
  - [Understanding the Output](#understanding-the-output)
- [Evaluation](#evaluation)
- [Fine-Tuning](#fine-tuning-the-model)
- [FAQ](#faq)
- [Contact](#contact)

## Installation

To set up and use this project, follow these steps:

### 1. Prerequisites

Ensure that the following software is installed on your computer:

- **Python 3.8+**: Download from [python.org](https://www.python.org/).
- **Git**: Download from [git-scm.com](https://git-scm.com/).
- **Virtual Environment (Optional)**: It is recommended to use a virtual environment to manage dependencies.

### 2. Clone the Repository

Clone the repository to your local machine by running the following command in your terminal:

```bash
git clone <repository_url>
cd HeartScan_Classifier
```

### 3. Install Dependencies

Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Dependencies include:

- `transformers` (for BERT model)
- `scikit-learn` (for evaluation metrics)
- `spaCy` and `negspacy` (for the negation filter)
- `pandas` (for data manipulation)
- `Tkinter` (for GUI functionality)
- `PyTorch` (for model loading, training, and optimization). Install the version appropriate for your machine using the guide at [PyTorch](https://pytorch.org/get-started/locally/)

## Data Requirements

To use this project, you will need a CSV file with the following structure:

| Patient_ID | Report_Text         |
|------------|---------------------|
| P00001     | "The CMR scan shows..." |
| P00002     | "No evidence of..."     |

- **Patient_ID**: A unique identifier for each patient.
- **Report_Text**: The text of the CMR report to be classified.

Ensure that the CSV file is saved in UTF-8 encoding and contains these two columns, with appropriate CMR report text.

## How to Download and Set Up the Model

### 1. Download the Fine-Tuned Model

The fine-tuned model is hosted on google drive. You can download it using this [link](https://drive.google.com/uc?export=download&id=1iBakduy20hk5ZxEuZlQKuXlZx2dqCWLm)

Once downloaded, extract the model files into the `models/` directory of the project.

Ensure that the model files (`vocab.txt`, `config.json`, `checkpoint-{} directory`, etc.) are located in the `models/fine_tuned_bert/` directory.

## Using the Application

The application provides an easy-to-use Graphical User Interface (GUI) for classifying the CMR reports.

### Step 1: Running the GUI

To start the GUI, run the main script:

```bash
python scripts/main.py
```

This will open a window where you can select the input CSV file, define where you would like to save the results, specify the disease/diseases of interest, and toggle use of the negation filter.

### Step 2: Processing Reports

- **Upload CSV**: Click the "Upload CSV" button to select the file containing CMR reports.
- **Set Output Path**: Specify the path where the classified reports should be saved.
- **Specify Disease**: Identify the disease or diseases (comma seperated) the model should classify on.
- **Negation Filter**: Toggle the negation filter. If stereo button is selected, the negation filter will be applied prior to the BERT classifier. Off by default.
- **Start Classification**: Click the "Start Processing" button to process the reports.

If negation filter is engaged, the application will first apply a negation filter to classify obvious negatives. Reports that are not flagged as negative will be further classified by the BERT model into negative, positive, or neutral categories.

Run time will depend on the whether the negation filter is used (faster when engaged), the number of reports to be classified, and the hardware available on your machine.

### Understanding the Output

The processed reports are saved in three separate CSV files for convenience:

- **Negative Reports**: Reports identified as negative.
  - Saved as `<output_file>_negative.csv`
- **Positive Reports**: Reports with a positive indication of the disease.
  - Saved as `<output_file>_positive.csv`
- **Neutral Reports**: Indeterminate or inconclusive reports.
  - Saved as `<output_file>_neutral.csv`

These files will contain the `Patient_ID` and `Report_Text` for each report.

## Evaluation

To evaluate the model's performance on a labeled dataset, use the evaluation script:

```bash
python scripts/evaluate_model.py --model_path models/fine_tuned_bert --test_data {path to your test data}.csv
```

### Input Data Structure for Evaluation

Ensure that your test data CSV file has the following structure:

| Patient_ID | Report_Text         | Label |
|------------|---------------------|-------|
| P00001     | "The CMR scan shows..." | 1     |
| P00002     | "No evidence of..."     | 0     |

- **Label**: The correct classification of the report (0 = Negative, 1 = Positive, 2 = Neutral).

### Output

The evaluation script generates a classification report that includes **precision**, **recall**, **F1-score**, and **accuracy** for each class. The report is saved to `outputs/classification_report.txt`.

## Fine-Tuning the Model

To fine-tune the provided ClinicalBERT checkpoint for CMR report classification, follow these steps:

### Data Preparation
Ensure your labeled data is stored in a CSV file named `labeled_data.csv` located in the `data/` directory. The CSV should have the following structure:

- **Evaluated_Disease**: The disease being evaluated (e.g., "amyloidosis").
- **Report_Text**: The text of the CMR report.
- **Label**: The classification label (`0 = Negative`, `1 = Positive`, `2 = Indeterminate`).

### Run the Fine-Tuning Script
Use the provided Python script to fine-tune the model. The script will:

1. Load the labeled data and preprocess it.
2. Split the data into training and validation sets (90% for training, 10% for validation).
3. Tokenize the input using the ClinicalBERT tokenizer.
4. Fine-tune the ClinicalBERT model using the specified hyperparameters.

You can run the fine-tuning script as follows:

```bash
python scripts/fine_tune.py
```
By default, this script will fine-tune the origional pre-trained ClinicalBERT model. If you wish to perform further finetuning on a specific checkpoint, be sure to change the model and tokenizer paths within the script.

### Training Arguments
By default the training script uses the following arguments:

**Batch Size:** 32 for both training and evaluation.

**Learning Rate:** 2e-5 to ensure a stable training process.

**Number of Epochs:** 5 to balance training performance without overfitting.

**Mixed Precision Training:** Enabled (using fp16=True) for faster training on compatible GPUs.

**Logging:** Logs are saved every 50 steps, and evaluation occurs at the end of each epoch.

Feel free to change these as necessary for your training goals.

## Saving the Model
After fine-tuning, the script saves the model and tokenizer in the models/fine_tuned_bert/ directory. This includes the model weights, configuration, and tokenizer files.

## Using the Fine-Tuned Model
Once the model is fine-tuned, it can be used for inference. You can run the classifier script with the fine-tuned model to classify new CMR reports.

## Compute Metrics
During training, the model's performance is evaluated using accuracy, precision, recall, and F1 score. These metrics provide insights into how well the model is performing on the validation data.

## FAQ

1. **What do I do if the model does not load correctly?**
   
   Ensure that all required files (`vocab.txt`, `config.json`, `checkpoint-{} directory`, etc.) are located in the `models/fine_tuned_bert/` directory.

2. **How can I customize the classification categories?**
   
   You can modify the `generate_synthetic_data.py` script to add or update classification templates to suit your specific needs.

3. **How can I retrain the model with new data?**
   
   Use the `fine_tune_bert.py` script to retrain the model with your new labeled dataset:

   ```bash
   python scripts/fine_tune_bert.py
   ```

   Ensure that your dataset follows the same structure as the `labeled_data.csv` file and model + tokenizer paths point to the model you wish to retrain.

## Contact

For any issues or questions, please reach out to the project maintainer at [parker.martin@osumc.edu].
