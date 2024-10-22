#scripts/generate_synthetic_data.py

import pandas as pd
import random
import yaml
import numpy as np
import os

"""
This script generates synthetic clinical reports for training a model to classify cardiac MR reports.
We use weighted random selection to choose diseases based on estimated prevalence.
The weights are assigned as follows:
- Microvascular Disease (MVD): 50%
- Hypertrophic Cardiomyopathy (HCM, HOCM): 25%
- Cardiac Amyloidosis: 15%
- Sarcoid (Sarcoidosis): 10%
These weights are exaggerated approximations based on known prevalence rates in the population while ensuring there is representation for more rare diseases.
"""

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


def load_templates():
    # Get the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the correct path to the templates.yml file
    file_path = os.path.join(base_dir, '../data/templates.yml')

    # Load template strings from a YAML file.
    with open(file_path, 'r', encoding='utf-8') as f:
        templates = yaml.safe_load(f)

    #print(templates)  # Add this line to check if the data is loaded correctly
    return templates


def generate_synthetic_reports(num_samples, disease_terms, templates):
    # Load positive, negative, and neutral templates from the provided dictionary.
    positive_templates = templates['positive']
    negative_templates = templates['negative']
    neutral_templates = templates['neutral']
    additional_text_options = templates['additional']

    # Define disease terms and their aliases.
    disease_aliases = {
        "microvascular disease": ["microvascular disease", "MVD"],
        "cardiac amyloid": ["cardiac amyloid", "amyloidosis"],
        "sarcoid": ["sarcoid", "sarcoidosis"],
        "hypertrophic cardiomyopathy": ["hypertrophic cardiomyopathy", "HCM", "HOCM",
                                        "hypertrophic obstructive cardiomyopathy"]
    }

    # Define weights for disease selection based on prevalence.
    disease_weights = {
        "microvascular disease": 0.50,  # Most common among the listed diseases
        "hypertrophic cardiomyopathy": 0.25,  # Relatively common genetic heart disease
        "cardiac amyloid": 0.15,  # Less common, often underdiagnosed
        "sarcoid": 0.10  # Rare compared to the others
    }

    evidence_options = [
        "subendocardial late gadolinium enhancement",
        "increased native T1 mapping values",
        "elevated extracellular volume fraction",
        "diffuse myocardial uptake",
        "abnormal gadolinium kinetics",
        "myocardial edema/inflammation",
        "concentric left ventricular hypertrophy",
        "reduced global longitudinal strain",
        "biatrial enlargement",
        "pericardial effusion"
    ]

    ef_values = ["45", "50", "55", "60"]
    symptoms = ["shortness of breath", "chest pain", "palpitations", "fatigue"]
    aorta_measurements = ["3.5", "4.0", "4.5", "5.0"]
    diastolic_grades = ["I", "II", "III"]

    data = []
    for i in range(num_samples):
        # Generate a unique patient ID.
        patient_id = f"P{i + 1:05d}"

        # Randomly select a disease based on the defined weights.
        disease = np.random.choice(list(disease_aliases.keys()), p=list(disease_weights.values()))
        disease_variant = random.choice(disease_aliases[disease])  # Select a variant or alias for the disease.
        disease_capitalized = disease_variant.capitalize()

        # Set the evaluated disease as the chosen variant
        evaluated_disease = disease_variant

        # Select evidence if needed
        evidence = random.choice(evidence_options)
        ef_value = random.choice(ef_values)
        symptom = random.choice(symptoms)
        aorta_measurement = random.choice(aorta_measurements)
        diastolic_grade = random.choice(diastolic_grades)

        # Prepare a dictionary for formatting
        format_dict = {
            "disease": disease_variant,
            "Disease": disease_capitalized,
            "evidence": evidence,
            "ef_value": ef_value,
            "symptom": symptom,
            "aorta_measurement": aorta_measurement,
            "diastolic_grade": diastolic_grade
        }

        # Randomly select label
        label = random.choice([0, 1, 2])  # 0: Negative, 1: Positive, 2: Neutral

        if label == 1:
            template = random.choice(positive_templates)
        elif label == 0:
            template = random.choice(negative_templates)
        else:  # label == 2
            template = random.choice(neutral_templates)

        # Format the template with the selected disease term.
        report_text = template.format(**format_dict)

        # Introduce variability by adding additional text to 40% of the reports.
        if random.random() < 0.4:
            num_additional_texts = random.randint(1, 3)  # Add 1 to 3 additional texts
            additional_texts = random.sample(additional_text_options, num_additional_texts)
            additional_text = " ".join(additional_texts)
            report_text += " " + additional_text

        # Split the report into sentences and randomly shuffle them
        sentences = report_text.split('. ')
        random.shuffle(sentences)
        report_text = '. '.join(sentences)

        # Append the generated report to the dataset.
        data.append({
            "Patient_ID": patient_id,
            "Report_Text": report_text.strip(),
            "Evaluated_Disease": evaluated_disease,
            "Label": label  # Label for the evaluated disease
        })

    # Return the generated dataset as a Pandas DataFrame.
    return pd.DataFrame(data)


def main():
    # Load templates from YAML file.
    templates = load_templates()
    disease_terms = [
        "microvascular disease",
        "cardiac amyloid",
        "sarcoid",
        "hypertrophic cardiomyopathy"
    ]

    # Ensure the output directory exists
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate fine-tuning dataset (~5,000 samples).
    fine_tune_df = generate_synthetic_reports(20000, disease_terms, templates)
    fine_tune_df.to_csv(os.path.join(output_dir, 'labeled_data.csv'), index=False, encoding='utf-8')
    print("Fine-tuning dataset generated and saved to data/labeled_data.csv")

    # Generate testing dataset (~200 samples).
    test_df = generate_synthetic_reports(500, disease_terms, templates)
    # Keep labels for evaluation.
    test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False, encoding='utf-8')
    print("Testing dataset generated and saved to data/test_data.csv")


if __name__ == '__main__':
    main()

"""
Rationale:
The script generates synthetic cardiac MR reports to create labeled datasets for training and testing a machine learning model (e.g., ClinicalBERT).
We used weighted disease selection to simulate realistic prevalence in clinical settings.

Purpose:
To generate labeled synthetic data for training ClinicalBERT to classify cardiac MR reports based on specific diseases.

Inputs/Outputs:
- Input: YAML file containing templates for positive, negative, and additional text.
- Output: CSV files ('labeled_data.csv' and 'test_data.csv') containing synthetic patient reports with labels.

Functionality:
1. Load report templates from an external YAML file.
2. Generate synthetic patient reports using positive and negative templates, with weighted disease selection.
3. Add variability to the generated reports by appending additional text.
4. Save the generated reports to CSV files for use in model training and evaluation.

Example Usage:
Run the script to generate synthetic datasets for fine-tuning and testing machine learning models for cardiac MR report classification.
"""