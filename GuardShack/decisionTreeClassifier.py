from sklearn.tree import DecisionTreeClassifier
from feature import featureEngineer, createLabel

dataset = featureEngineer()

# Create pairs of tasks, excluding self pairs
task_pairs = [(task1, task2) for task1 in dataset['ID']
              for task2 in dataset['ID'] if task1 != task2]

# Initialize lists to store features and labels
features = []
labels = []

# Generate features and labels for task pairs
# for task1, task2 in task_pairs:
#     duration_difference = dataset.loc[dataset['ID'] == task1, 'Duration'].values[0] - \
#         dataset.loc[dataset['ID'] == task2, 'Duration'].values[0]
#     order_label = 1 if duration_difference < 0 else 0
#     features.append([duration_difference])
#     labels.append(order_label)
for task1, task2 in task_pairs:
    start_day_task1 = dataset.loc[dataset['ID'] == task1, 'StartDay'].values[0]
    start_month_task1 = dataset.loc[dataset['ID'] == task1, 'StartMonth'].values[0]
    start_year_task1 = dataset.loc[dataset['ID'] == task1, 'StartYear'].values[0]
    end_day_task1 = dataset.loc[dataset['ID'] == task1, 'FinishDay'].values[0]
    end_month_task1 = dataset.loc[dataset['ID'] == task1, 'FinishMonth'].values[0]
    end_year_task1 = dataset.loc[dataset['ID'] == task1, 'FinishYear'].values[0]

    start_day_task2 = dataset.loc[dataset['ID'] == task2, 'StartDay'].values[0]
    start_month_task2 = dataset.loc[dataset['ID'] == task2, 'StartMonth'].values[0]
    start_year_task2 = dataset.loc[dataset['ID'] == task2, 'StartYear'].values[0]
    end_day_task2 = dataset.loc[dataset['ID'] == task2, 'FinishDay'].values[0]
    end_month_task2 = dataset.loc[dataset['ID'] == task2, 'FinishMonth'].values[0]
    end_year_task2 = dataset.loc[dataset['ID'] == task2, 'FinishYear'].values[0]

    features.append([start_day_task1, start_month_task1, start_year_task1, end_day_task1, end_month_task1, end_year_task1,
                     start_day_task2, start_month_task2,  start_year_task2, end_day_task2, end_month_task2, end_year_task2])
    
    order_label = createLabel(task1, task2)
    labels.append(order_label)
# Features and target variable
# Want the feature to be start date of first task, end date of first task, start date of second task, end date of second task?
X = features
y = labels

# Create and fit a Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Predict the order of tasks for a new example (provide the duration difference as a feature)

startday1 = 1
startmonth1 = 1
startyear1 = 2023
endday1 = 5
endmonth1 = 10
endyear1 = 2023

startday2 = 6
startmonth2 = 1
startyear2 = 2023
endday2 = 10
endmonth2 = 1
endyear2 = 2023

new_example = [[startday1, startmonth1, startyear1, endday1, endmonth1, endyear1, startday2, startmonth2, startyear2, endday1, endmonth2, endyear2]]  # Example duration difference
prediction = clf.predict(new_example)

# if prediction[0] == 1:
#     print("Task 1 precedes Task 2.")
# else:
#     print("Task 2 precedes Task 1.")

print(prediction)

count = 0
for index, row in dataset.iterrows():
    if count == 0:
        task1 = row['ID']

        startDay1 = row['StartDay']
        startMonth1 = row['StartMonth']
        startYear1 = row['StartYear']
        endDay1 = row['FinishDay']
        endMonth1 = row['FinishMonth']
        endYear1 = row['FinishYear']
        count += 1
    else:        
        startDay2 = startDay1
        startMonth2 = startMonth1
        startYear2 = startYear1
        endDay2 = endDay1
        endMonth2 = endMonth1
        endYear2 = endYear1
        task2 = task1

        startDay1 = row['StartDay']
        startMonth1 = row['StartMonth']
        startYear1 = row['StartYear']
        endDay1 = row['FinishDay']
        endMonth1 = row['FinishMonth']
        endYear1 = row['FinishYear']
        task1 = row['ID']

    new_example = [[startday1, startmonth1, startyear1, endday1, endmonth1, endyear1, startday2,
                    startmonth2, startyear2, endday1, endmonth2, endyear2]]  # Example duration difference
    prediction = clf.predict(new_example)
    actual = createLabel(task1,task2)

    print("TASK1: ", task1, "\tTASK2: ", task2)
    print("PREDICTION VS ACTUAL:\t", prediction[0], actual)


        

