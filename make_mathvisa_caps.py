import json

# Sample input file path (replace with your actual file path)
input_file_path = '/usr/xtmp/mxy/VLM-Poisoning/data/clean_data/captions_mathvista/captions_vitgpt2.json'
output_file_path = '/usr/xtmp/mxy/VLM-Poisoning/data/task_data/MathVista_base_hamburgerFries_target/base_train/cap.json'

# Load the existing JSON
with open(input_file_path, 'r') as f:
    data = json.load(f)

# Initialize the list of annotations for the desired output
annotations = []

# Iterate through the texts in the "texts" field to create new annotations
for image_id, text in data["texts"].items():
    annotation = {
        "image_id": image_id,  # Use the text key as image_id
        "caption": text  # Use the corresponding text as the caption
    }
    annotations.append(annotation)

# Create the output dictionary
output = {
    "annotations": annotations
}

# Save the result to a new JSON file
with open(output_file_path, 'w') as f:
    json.dump(output, f, indent=4)

print("Transformation complete. Output saved to", output_file_path)
