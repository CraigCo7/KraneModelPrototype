import openai
import pandas as pd
import csv
import os

from feature import general

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")


csv_in_path = 'CE Input/GuardShackSchedule.csv'
csv_out_path = 'CE Output/Output-CEModelProcess.csv'
# df = pd.read_csv(csv_in_path)


def assignConstructionSoft(csv_in_path):
    df = pd.read_csv(csv_in_path)
    df.drop_duplicates(subset=['Activity Name'])
    df.dropna(subset=['Activity Name'], inplace=True)
    activities = df['Activity Name'].tolist()
    prompt = "You are a construction foreman. You will be given a list of processes, and need to associate each one with a general product in the construction schedule. Assign the following processes, given in the following list, to a general product. Some of these processes belong to the same general product. Processes: " + \
        ', '.join(map(str, activities))
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1000,
        temperature=0,
        best_of=1
    )
    return response['choices'][0]['text']


def readAssignConstruction(csv_in_path):
    output = {}
    text = assignConstructionSoft(csv_in_path)
    text = text.split('\n')
    for category in text:
        if category == '':
            continue
        process = category.split('-')[0].strip()

        # cat = category.split('-')[1].strip()
        # Split into first word of assigned category
        cat = category.split('-')[1].strip().split(' ')[0].strip()
        if output.get(cat) == None:
            output[cat] = [process.strip()]
        else:
            output[cat].append(process.strip())

    for x, y in output.items():
        print(x, y)
    return output


def testOne():
    print(readAssignConstruction("CE Input/GuardShackSchedule.csv"))


def sortByGroup(csv_in_path, num_groups):
    df = general(csv_in_path)
    activities = df['Activity Name'].tolist()
    print(activities)
    prompt = "Here is a list of activities: " + \
        ', '.join(map(str, activities)) + "\nSort them into " + \
        str(num_groups) + " groups based on the type of activity."
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1000,
        temperature=0,
        best_of=1
    )
    return response['choices'][0]['text']


print(sortByGroup(csv_in_path, 4))
