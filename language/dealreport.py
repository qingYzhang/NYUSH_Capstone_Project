import os
import csv
import json
import random
from sklearn.model_selection import train_test_split

# Define paths
path_to_csv = "./label_report/experiment_2/results.csv"  # Replace with actual CSV path
root_directory = "../xr_knee_sample_20241202/images"  # Replace with actual root directory path
output_train_json = "./label_report/train.json"
output_test_json = "./label_report/test.json"

# Define labels and initialize data
labels = [
    "fracture", "osteoarthritis", "joint effusion", "healing/healed fracture", 
    "soft tissue swelling", "orif", "arthroplasty", "enthesopathy", 
    "intra-articular fracture", "heterotopic ossification", "chondrocalcinosis", 
    "osteochondral injury", "intraarticular body", "osteotomy", "No findings"
]

# Read the CSV file and process data
accession_data = {}
with open(path_to_csv, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        accession = row['accession']  # Adjust key if different in CSV
        label_list = row['gpt_output'].split(", ")  # Adjust delimiter if needed
        one_hot_target = [1 if label in label_list else 0 for label in labels]
        if not any(one_hot_target):  # Skip if all values in one_hot_target are 0
            continue
        accession_data[accession] = one_hot_target
        # print(accession_data)

# Map images to accession numbers
data = []
for accession, target in accession_data.items():
    accession_folder = os.path.join(root_directory, accession)
    if os.path.isdir(accession_folder):
        for img_file in os.listdir(accession_folder):
            if img_file.endswith(('.png', '.jpg', '.jpeg', '.dcm')):  # Add more extensions if necessary
                img_path = os.path.join(accession_folder, img_file)
                data.append({"img_path": img_path, "target": target})

# Split data into train and test sets by accession
accession_list = list(accession_data.keys())
train_accessions, test_accessions = train_test_split(accession_list, test_size=0.2, random_state=42)

train_data = [item for item in data if any(train_accession in item['img_path'] for train_accession in train_accessions)]
test_data = [item for item in data if any(test_accession in item['img_path'] for test_accession in test_accessions)]

# Save data to JSON files
with open(output_train_json, 'w') as train_file:
    json.dump(train_data, train_file, indent=4)

with open(output_test_json, 'w') as test_file:
    json.dump(test_data, test_file, indent=4)

print(f"Train and test JSON files generated:\nTrain: {output_train_json}\nTest: {output_test_json}")


