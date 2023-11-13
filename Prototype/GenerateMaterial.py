# Simple Script to Generate CE Material
# 1. Reads BOM CSV
# 2. Sums up all the values per material
# 3. Outputs to separate CSV

import csv

# Specify the path to your CSV file
csv_input_path = 'CE Input/Prototype Materials Input.csv'

csv_out_path = 'CE Output/Output-CEModelMaterial.csv'


output_data = [["ID", "Material", "MatTotalQuantity"]]


def generateMaterial():
    # Open the CSV file in read mode
    with open(csv_input_path, 'r', newline='') as input_file:
        # Create a CSV reader object
        csv_reader = csv.reader(input_file)

        MatQuantity = {}
        # Iterate through each row in the CSV file
        for row in csv_reader:
            # Find the largest value for each Subcontractor
            materialName = row[1].strip()
            quantity = row[2]
            if materialName == 'Material BOM' or materialName == '':
                continue
            if MatQuantity.get(materialName) == None:
                MatQuantity[materialName] = quantity
            else:
                MatQuantity[materialName] += quantity

    # The CSV file is automatically closed when the 'with' block is exited

    count = 0
    for material, quantity in MatQuantity.items():
        count += 1
        output_data.append(
            [count, material, quantity])

    # Write the processed data to the output CSV file
    with open(csv_out_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)

        # Write the processed data to the output file
        csv_writer.writerows(output_data)
