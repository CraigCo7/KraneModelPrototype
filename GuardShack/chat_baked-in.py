# $ pip install openai
import openai
import pandas as pd
import csv

# Load your API key from an environment variable or secret management service
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "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"


csv_in_path = 'CE Input/GuardShackSchedule.csv'
csv_out_path = 'CE Output/Output-CEModelProcess.csv'
df = pd.read_csv(csv_in_path)


# def assignConstructionSoft():
#     df.dropna(subset=['Activity Name'], inplace=True)
#     activities = df['Activity Name'].tolist()
#     prompt = "You are a construction foreman. You will be given a list of processes, and need to associate each one with a product in the construction schedule. Assign the following processes, given in the following list, to a product. Processes: " + \
#         ', '.join(map(str, activities)) + "HI"
#     response = openai.Completion.create(
#         model="gpt-3.5-turbo-instruct",
#         prompt=prompt,
#         max_tokens=1000,
#         temperature=0,
#         best_of=1
#     )
#     return response['choices'][0]['text']


# Divides chatGPT return into a dictionary.


def prompt_generation():
    df.dropna(subset=['Activity Name'], inplace=True)
    activities = df['Activity Name'].tolist()
    # print(activities)
    prompt = "You are given a list of activities associated with construction. Sort them out into 4 process outputs: base, wall (which should include paint and windows), roof, eletrical. Make sure all activities have been sorted! Return in this specific format. *Groupname*: *list* /newline. Here is the list of activities: " + \
        ', '.join(map(str, activities))
    print("\nPROMPT:\n", prompt, "\n")
    return prompt


def createResponse():
    prompt = prompt_generation()
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1000,
        temperature=0,
        best_of=1
    )
    return response['choices'][0]['text']


# Divides chatGPT return into a dictionary.
def readResponse():
    output = {}
    text = createResponse()
    text = text.split('\n')
    for category in text:
        if category == '':
            continue
        process = category.split(':')[0].strip()
        activityList = category.split(':')[1].strip().split(',')
        for activity in activityList:
            if output.get(process) == None:
                output[process] = [activity.strip()]
            else:
                output[process].append(activity.strip())

    for x, y in output.items():
        print(x, y)
    return output


def generateProcess():
    output_data = [["ID", "ProcessOutput", "ProcStep", "ProcStepSuccessor", "ProcStepDuration",
                    "ProcMaterialAssignment", "ProcMaterialBatchSize", "ProcResourceAssignment", "ProcResourceQuantity"]]
    ID = 1
    info = readResponse()

    with open(csv_in_path, 'r', newline='') as input_file:
        csv_reader = csv.reader(input_file)
        ind = 0
        for row in csv_reader:
            activityName = row[1].strip()
            if activityName == '' or activityName == 'Activity Name':
                continue

            key_list = list(info.keys())
            val_list = list(info.values())

            for returnList in val_list:
                if activityName in returnList:
                    ind = val_list.index(returnList)
                # else:
                #     ind = 0
            processOutput = key_list[ind]
            output_data.append([ID, processOutput, activityName, None,
                                None, None, None, None, None])
            ID += 1

    with open(csv_out_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)

        # Write the processed data to the output file
        csv_writer.writerows(output_data)


readResponse()
