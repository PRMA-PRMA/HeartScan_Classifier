# scripts/evaluate_model.py

import os
import pandas as pd
from data_loader import preprocess_text
from negation_filter import negation_filter
from bert_classifier import BERTClassifier
from sklearn.metrics import classification_report
import argparse

def evaluate_model(model_path, test_data_file, disease_terms):
    # Load test data
    df = pd.read_csv(test_data_file)

    # Preprocess the text if necessary
    df['Report_Text'] = df['Report_Text'].apply(preprocess_text)
    true_labels = df['Label'].tolist()  # Use labels from test data

    # Initialize the classifier
    classifier = BERTClassifier(model_path=model_path)

    predicted_labels = []
    misclassified_reports = []  # To store misclassified examples
    for index, row in df.iterrows():
        # Get the disease and report text from each row
        disease = row['Evaluated_Disease']
        text = row['Report_Text']
        true_label = row['Label']

        # Classify the report for the evaluated disease
        predicted_class = classifier.classify(disease, text)
        predicted_labels.append(predicted_class)

        # Check if the prediction is incorrect
        if predicted_class != true_label:
            misclassified_reports.append({
                'Report_Text': text,
                'True_Label': true_label,
                'Predicted_Label': predicted_class,
                'Disease': disease
            })

    # Generate classification report
    target_names = ['Negative', 'Positive', 'Neutral']
    report = classification_report(true_labels, predicted_labels, target_names=target_names)
    print(report)

    # Save report to file
    with open('outputs/classification_report.txt', 'w') as f:
        f.write(report)
    print("Classification report saved to outputs/classification_report.txt")

    # Save misclassified reports to a CSV file
    if misclassified_reports:
        misclassified_df = pd.DataFrame(misclassified_reports)
        misclassified_df.to_csv('outputs/misclassified_reports.csv', index=False)
        print(f"Misclassified reports saved to outputs/misclassified_reports.csv")
    else:
        print("No misclassified reports found.")

    # Generate binary classification report for Negative vs Not Negative
    binary_true_labels = [0 if label == 0 else 1 for label in true_labels]  # 0: Negative, 1: Not Negative
    binary_predicted_labels = [0 if label == 0 else 1 for label in predicted_labels]  # 0: Negative, 1: Not Negative

    binary_target_names = ['Negative', 'Not Negative']
    binary_report = classification_report(binary_true_labels, binary_predicted_labels, target_names=binary_target_names)
    print(binary_report)

    # Save binary report to file
    with open('outputs/binary_classification_report.txt', 'w') as f:
        f.write(binary_report)
    print("Binary classification report saved to outputs/binary_classification_report.txt")


def main():
    parser = argparse.ArgumentParser(description='Evaluate BERT Classifier')
    parser.add_argument('--model_path', type=str, required=False, help='Path to BERT model', default=None)
    parser.add_argument('--test_data', type=str, required=False, help='Path to test data CSV',
                        default='data/test_data.csv')
    parser.add_argument('--disease_terms', type=str, required=False, nargs='+', help='Disease terms',
                        default=['microvascular disease', 'MVD',
                                 "cardiac amyloid", "amyloidosis",
                                 "sarcoid", "sarcoidosis",
                                 "hypertrophic cardiomyopathy", "HCM",
                                 "HOCM", "hypertrophic obstructive cardiomyopathy"
                                 ])

    args = parser.parse_args()

    evaluate_model(args.model_path, args.test_data, args.disease_terms)


if __name__ == '__main__':
    main()

'''
Example Usage:
python scripts/evaluate_model.py --test_data data/test_data.csv
python scripts/evaluate_model.py --model_path models/fine_tuned_bert --test_data data/test_data.csv
python scripts/evaluate_model.py --model_path models/fine_tuned_bert --test_data data/combined_amyloid.csv --disease_terms amyloid amyloidosis
'''
