# Simple Script to Generate CE Resource
# 1. Reads Schedule CSV
# 2. Finds the largest Manpower per subcontractor
# 3. Outputs to separate CSV

import csv

# Specify the path to your CSV file
csv_input_path = 'CE Input/Prototype Schedule Input.csv'

csv_out_path = 'CE Output/Output-CEModelResource.csv'


output_data = [["ID", "Resource", "ResQuantity", "ResCompany"]]


def generateResource():
    # Open the CSV file in read mode
    with open(csv_input_path, 'r', newline='') as input_file:
        # Create a CSV reader object
        csv_reader = csv.reader(input_file)

        ResQuantity = {}
        # Iterate through each row in the CSV file
        for row in csv_reader:
            # Find the largest value for each Subcontractor
            contractorName = row[4]
            quantity = row[6]
            if contractorName == 'Subcontractor' or contractorName == '':
                continue
            if ResQuantity.get(contractorName) == None:
                ResQuantity[contractorName] = quantity
            else:
                value = ResQuantity.get(contractorName)
        if quantity > value:
            ResQuantity[contractorName] = quantity

    # The CSV file is automatically closed when the 'with' block is exited

    count = 0
    for subcontractor, quantity in ResQuantity.items():
        count += 1
        output_data.append(
            [count, "Manpower (" + subcontractor + ")", quantity, subcontractor])

    # Write the processed data to the output CSV file
    with open(csv_out_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)

        # Write the processed data to the output file
        csv_writer.writerows(output_data)
