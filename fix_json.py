def fix_json(filename):
    try:
        # Open the original JSON file
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Open a new file to write the corrected JSON
        with open('fixed_' + filename, 'w') as file:
            file.write('[\n')  # Start of the JSON array
            for line in lines[:-1]:  # Iterate through all lines except the last one
                if line.strip():  # If the line is not empty
                    file.write(line.strip() + ',\n')  # Add a comma at the end
            file.write(lines[-1].strip() + '\n')  # Write the last line without a comma
            file.write(']')  # End of the JSON array

        print("JSON fixed successfully!")
    except Exception as e:
        print("An error occurred:", e)

# Usage example, replace 'path_to_your_file.json' with the path to your JSON file
fix_json('mental_health_convo_set.json')