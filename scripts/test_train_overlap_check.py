import pandas as pd
from tkinter import Tk, filedialog


def upload_and_analyze_csv_files():
    # Initialize Tkinter root
    root = Tk()
    root.withdraw()  # Hide the Tkinter root window

    # Upload the first CSV file
    print("Select the test CSV file.")
    first_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    first_df = pd.read_csv(first_file_path)

    # Upload the second CSV file
    print("Select the training CSV file.")
    second_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    second_df = pd.read_csv(second_file_path)

    # Extract the second column from each file (assuming the statements are in the second column)
    first_statements = first_df.iloc[:, 1].dropna().tolist()
    second_statements = second_df.iloc[:, 1].dropna().tolist()

    # Analyze repeated statements within the first file
    repeated_in_first = len(first_statements) - len(set(first_statements))
    repeated_in_first_percent = (repeated_in_first / len(first_statements)) * 100

    # Analyze repeated statements within the second file
    repeated_in_second = len(second_statements) - len(set(second_statements))
    repeated_in_second_percent = (repeated_in_second / len(second_statements)) * 100

    # Analyze statements from the first file that also appear in the second file
    repeated_between_files = len([statement for statement in first_statements if statement in second_statements])
    repeated_between_percent = (repeated_between_files / len(first_statements)) * 100

    # Print the results
    print("\nAnalysis Results:")
    print(f"File 1: {repeated_in_first} repeated statements ({repeated_in_first_percent:.2f}%)")
    print(f"File 2: {repeated_in_second} repeated statements ({repeated_in_second_percent:.2f}%)")
    print(
        f"Between Files: {repeated_between_files} statements from File 1 also appear in File 2 ({repeated_between_percent:.2f}%)")


# Run the function
if __name__ == "__main__":
    upload_and_analyze_csv_files()
