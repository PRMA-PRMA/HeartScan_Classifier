# scripts/main.py

import os
import pandas as pd
from data_loader import load_data, preprocess_text
from negation_filter import negation_filter
from bert_classifier import BERTClassifier
import tkinter as tk
from tkinter import filedialog, messagebox

def main():
    # Initialize the main window
    root = tk.Tk()
    root.title("CMR Diagnosis Identifier")
    root.geometry("500x350")

    # Variable to track the negation filter
    neg_filter_var = tk.BooleanVar(value=False)

    # Function to select the input CSV file
    def select_input_file():
        input_file_path = filedialog.askopenfilename(
            title="Select Input CSV File",
            filetypes=[("CSV Files", "*.csv")]
        )
        input_entry.delete(0, tk.END)
        input_entry.insert(0, input_file_path)

    # Function to select the output CSV file
    def select_output_file():
        output_file_path = filedialog.asksaveasfilename(
            title="Select Output CSV File Directory Location",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )
        output_entry.delete(0, tk.END)
        output_entry.insert(0, output_file_path)

    # Function to start processing
    def start_processing():
        input_file = input_entry.get()
        output_file = output_entry.get()
        disease_terms_input = disease_entry.get()
        neg_filter = neg_filter_var.get()  # Get the value of the negation filter checkbox

        if not os.path.isfile(input_file):
            messagebox.showerror("Error", "Please select a valid input CSV file.")
            return

        if not output_file:
            messagebox.showerror("Error", "Please select a valid output CSV file.")
            return

        if not disease_terms_input.strip():
            messagebox.showerror("Error", "Please enter at least one disease term.")
            return

        # Split disease terms by commas and strip whitespace
        disease_terms = [term.strip().lower() for term in disease_terms_input.split(',')]

        # Proceed with processing
        process_reports(input_file, output_file, disease_terms, neg_filter)

    # UI Elements
    tk.Label(root, text="Input CSV File:").pack(pady=5)
    input_frame = tk.Frame(root)
    input_entry = tk.Entry(input_frame, width=50)
    input_entry.pack(side=tk.LEFT)
    input_button = tk.Button(input_frame, text="Browse", command=select_input_file)
    input_button.pack(side=tk.LEFT, padx=5)
    input_frame.pack()

    tk.Label(root, text="Output CSV File Directory Location:").pack(pady=5)
    output_frame = tk.Frame(root)
    output_entry = tk.Entry(output_frame, width=50)
    output_entry.pack(side=tk.LEFT)
    output_button = tk.Button(output_frame, text="Browse", command=select_output_file)
    output_button.pack(side=tk.LEFT, padx=5)
    output_frame.pack()

    tk.Label(root, text="Disease Terms (comma-separated):").pack(pady=5)
    disease_entry = tk.Entry(root, width=50)
    disease_entry.pack()

    # Add negation filter checkbutton
    neg_filter_checkbox = tk.Checkbutton(root, text="Apply Negation Filter", variable=neg_filter_var)
    neg_filter_checkbox.pack(pady=5)

    start_button = tk.Button(root, text="Start Processing", command=start_processing)
    start_button.pack(pady=20)

    root.mainloop()


def process_reports(input_file, output_dir, disease_terms, neg_filter):
    # Load and preprocess data
    df = load_data(input_file)
    df['Report_Text'] = df['Report_Text'].apply(preprocess_text)

    # Initialize classifier
    classifier = BERTClassifier(model_path='../models/fine_tuned_bert')

    # Convert list of disease terms into a single string
    disease_terms_str = ' '.join(disease_terms)

    # Lists to store results
    positive_reports = []
    neutral_reports = []
    negative_reports = []

    # Process each report
    for index, row in df.iterrows():
        text = row['Report_Text']
        patient_id = row['Patient_ID']

        # Apply negation filter if indicated
        if neg_filter:
            is_negative = negation_filter(text.lower(), disease_terms)
        else:
            is_negative = False

        if is_negative:
            # Negation filter classified as negative
            negative_reports.append({'Patient_ID': patient_id, 'Report_Text': text})
        else:
            # Use BERT classifier to further classify
            predicted_class = classifier.classify(disease_terms_str, text)

            if predicted_class == 1:
                # Positive diagnosis
                positive_reports.append({'Patient_ID': patient_id, 'Report_Text': text, 'Disease': disease_terms_str})
            elif predicted_class == 2:
                # Neutral/Indeterminate diagnosis
                neutral_reports.append({'Patient_ID': patient_id, 'Report_Text': text, 'Disease': disease_terms_str})
            elif predicted_class == 0:
                # Negative diagnosis caught by BERT
                negative_reports.append({'Patient_ID': patient_id, 'Report_Text': text, 'Disease': disease_terms_str})

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate proper file paths for saving positive, neutral, and negative CSV files within the output directory
    positive_file = os.path.join(output_dir, "positive.csv")
    neutral_file = os.path.join(output_dir, "neutral.csv")
    negative_file = os.path.join(output_dir, "negative.csv")

    # Save reports to CSV
    if positive_reports:
        positive_df = pd.DataFrame(positive_reports)
        positive_df.to_csv(positive_file, index=False)

    if neutral_reports:
        neutral_df = pd.DataFrame(neutral_reports)
        neutral_df.to_csv(neutral_file, index=False)

    if negative_reports:
        negative_df = pd.DataFrame(negative_reports)
        negative_df.to_csv(negative_file, index=False)

    # Show message box to indicate processing completion
    messagebox.showinfo(
        "Processing Complete",
        f"Positive reports saved to:\n{positive_file}\n"
        f"Neutral reports saved to:\n{neutral_file}\n"
        f"Negative reports saved to:\n{negative_file}"
    )


if __name__ == '__main__':
    main()
