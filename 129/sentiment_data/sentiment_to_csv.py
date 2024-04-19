import csv
import os

# Directory containing your text files
input_directory = '.'
# Path to your output CSV file
output_file_path = 'output.csv'

# Open the output CSV file
with open(output_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    # Write the header
    writer.writerow(['Sentence', 'Sentiment'])

    # Loop through all text files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):  # Check if the file is a text file
            with open(os.path.join(input_directory, filename), 'r', encoding='utf-8') as txt_file:
                lines = txt_file.readlines()

                # Process each line in the current text file
                for line in lines:
                    if line.strip():  # Ensure the line isn't empty
                        # Split the line into sentence and sentiment based on the tab separator
                        sentence, sentiment = line.split('\t')
                        # Write to the CSV file
                        writer.writerow([sentence.strip(), sentiment.strip()])

print("Conversion complete. The data from all text files has been saved to", output_file_path)