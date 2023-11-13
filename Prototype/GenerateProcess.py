# Simple Script to Generate CE Resource
# 1. Reads Schedule CSV
# 2. Finds the largest Manpower per subcontractor
# 3. Outputs to separate CSV

import csv
from datetime import datetime

# Specify the path to your CSV file
csv_schedule_input = 'CE Input/Prototype Schedule Input.csv'
csv_bom_input = 'CE Input/Prototype Materials Input.csv'

csv_out_path = 'CE Output/Output-CEModelProcess.csv'

# Find Materials, and their Values, Associated with each ProcStep.


def findMaterialAssignment() -> dict:
    assignment = {}

    with open(csv_bom_input, 'r', newline='') as input_file:
        csv_reader = csv.reader(input_file)

        # Key = activityDescription, value = [ [list of materials], [list of values] ]
        value = [[], []]
        prevActivityDescription = ''
        for row in csv_reader:
            activityDescription = row[0].strip()
            material = row[1].strip()
            # while (material != ''):
            quantity = row[2].strip()
            if (material != ''):
                if (activityDescription != ''):
                    # print(activityDescription)
                    assignment[prevActivityDescription] = value
                    value = [[], []]
                    prevActivityDescription = activityDescription
                value[0].append(material)
                value[1].append(quantity)
        assignment[prevActivityDescription] = value
        assignment.pop('')

    return assignment


def findLevels():
    maxLevel = 0
    with open(csv_schedule_input, 'r', newline='') as input_file:
        # Create a CSV reader object
        csv_reader = csv.reader(input_file)
        for row in csv_reader:
            level = row[1].split("-")[1].strip()
            if level > maxLevel:
                maxLevel = level
    return maxLevel


date_format = "%m/%d/%Y"

materialAssignment = findMaterialAssignment()


def createOutputData():
    # Open the CSV file in read mode
    with open(csv_schedule_input, 'r', newline='') as input_file:
        # Create a CSV reader object
        csv_reader = csv.reader(input_file)
        # Skip the header row
        next(csv_reader)

        # Key = ProcStep
        #Value = [ProcStepSuccessor, ProcStepDuration, ProcMaterialAssignment, ProcMaterialBatchSize, ProcResourceAssignment, ProcResourceQuantity]
        data = {}
        ID = 1
        # Iterate through each row in the CSV file
        for row in csv_reader:
            procStep = row[1].split("-")[0].strip()
            if procStep == '':
                continue
            startDate = datetime.strptime(row[2].strip(), date_format)
            endDate = datetime.strptime(row[3].strip(), date_format)
            subcontractor = row[4].strip()
            manpower = row[6].strip()
            if procStep == 'Activity Description' or procStep == '':
                continue

            # Build the data.
            if data.get(procStep) == None:
                # ProcStepProcessor
                ID += 1
                ProcStepProcessor = ID

                # ProcStepDuration
                ProcStepDuration = str(endDate-startDate).split(',')[0]

                try:
                    # Attempt to access a key that doesn't exist
                    materialsList = materialAssignment[procStep][0]
                    materialsQuantity = materialAssignment[procStep][1]
                except KeyError as e:
                    materialsList = []
                    materialsQuantity = []
                    pass

                # ProcMaterial Assignment
                # materialsList = materialAssignment[procStep][0]
                ProcMaterialAssignment = ""
                for i in materialsList:
                    ProcMaterialAssignment = ProcMaterialAssignment + i + ", "

                # ProcMaterialBatchSize
                # materialsQuantity = materialAssignment[procStep][1]
                ProcMaterialBatchSize = ""
                for i in materialsQuantity:
                    ProcMaterialBatchSize = ProcMaterialBatchSize + \
                        str(int(i)/3) + ", "

                # ProcResourceAssignment
                ProcResourceAssignment = "Manpower (" + subcontractor + ")"

                # ProcResourceQuantity
                ProcResourceQuantity = manpower

                # Add Data to Output
                data[procStep] = [ProcStepProcessor, ProcStepDuration, ProcMaterialAssignment,
                                  ProcMaterialBatchSize, ProcResourceAssignment, ProcResourceQuantity]
    return data


def generateProcess():
    output_data = [["ID", "ProcessOutput", "ProcStep", "ProcStepSuccessor", "ProcStepDuration",
                    "ProcMaterialAssignment", "ProcMaterialBatchSize", "ProcResourceAssignment", "ProcResourceQuantity"]]
    ID = 1
    info = createOutputData()
    for procStep, data in info.items():
        output_data.append([ID, "LEVEL", procStep, data[0],
                           data[1], data[2], data[3], data[4], data[5]])
        ID += 1

    with open(csv_out_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)

        # Write the processed data to the output file
        csv_writer.writerows(output_data)
