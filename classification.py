# %%

# import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

df_R1 = pd.read_excel("Deliverable_table.xlsx", sheet_name="R1")
df_R91 = pd.read_excel("Deliverable_table.xlsx", sheet_name="R91")
df_R186 = pd.read_excel("Deliverable_table.xlsx", sheet_name="R186")
df_R187 = pd.read_excel("Deliverable_table.xlsx", sheet_name="R187")

# if not os.path.exists("saxs_waxs_figures"):
#     os.makedirs("saxs_waxs_figures")

# %%
# Drop the first two columns
df_R1 = df_R1[1:22].drop(["Data point ", "Sample "], axis=1)
df_R91 = df_R91[1:22].drop(["Data point ", "Sample "], axis=1)
df_R186 = df_R186[1:22].drop(["Data point ", "Sample "], axis=1)
df_R187 = df_R187[1:22].drop(["Data point ", "Sample "], axis=1)

# %%


df_R1_norm = (df_R1 - df_R1.min()) / (df_R1.max() - df_R1.min())
df_R91_sub_norm = (df_R91 - df_R91.min()) / (
    df_R91.max() - df_R91.min()
)
df_R186_norm = (df_R186 - df_R186.min()) / (
    df_R186.max() - df_R186.min()
)
df_R187_norm = (df_R187 - df_R187.min()) / (
    df_R187.max() - df_R187.min()
)

# the second 7 rows are selected
df_R1_sub = df_R1_norm[7:14]
df_R91_sub = df_R91_sub_norm[7:14]
df_R186_sub = df_R186_norm[7:14]
df_R187_sub = df_R187_norm[7:14]


# %%
list(df_R1_sub.columns)

# %%

# Select the saxs data
df_R1_sub_selected = df_R1_sub[["SAXS1", "Unnamed: 3", "SAXS2", "Unnamed: 5"]]
df_R91_sub_selected = df_R91_sub[["SAXS1", "Unnamed: 3", "SAXS2", "Unnamed: 5"]]
df_R186_sub_selected = df_R186_sub[["SAXS1", "Unnamed: 3", "SAXS2", "Unnamed: 5"]]
df_R187_sub_selected = df_R187_sub[["SAXS1", "Unnamed: 3", "SAXS2", "Unnamed: 5"]]

# Add a new column 'label' to each DataFrame
df_R1_sub_selected.loc[:, "label"] = 0
df_R91_sub_selected.loc[:, "label"] = 1
df_R186_sub_selected.loc[:, "label"] = 2
df_R187_sub_selected.loc[:, "label"] = 3

# Concatenate the selected columns from each DataFrame
df_x_saxs = pd.concat(
    [
        df_R1_sub_selected,
        df_R91_sub_selected,
        df_R186_sub_selected,
        df_R187_sub_selected,
    ],
    axis=0,
)

# Create the target variable y
y_saxs = df_x_saxs["label"]

# Remove the 'label' column from df_x_saxs
x_saxs = df_x_saxs.drop("label", axis=1)

# %%

# Select the waxs data
df_R1_sub_selected = df_R1_sub[
    [
        "WAXS M1",
        "Unnamed: 7",
        "Unnamed: 8",
        "Unnamed: 9",
        "Unnamed: 10",
        "WAXS M2",
        "Unnamed: 12",
        "Unnamed: 13",
        "Unnamed: 14",
        "Unnamed: 15",
        "WAXS E1",
        "Unnamed: 17",
        "Unnamed: 18",
        "Unnamed: 19",
        "Unnamed: 20",
        "WAXS E2",
        "Unnamed: 22",
        "Unnamed: 23",
        "Unnamed: 24",
        "Unnamed: 25",
    ]
]
df_R91_sub_selected = df_R91_sub[
    [
        "WAXS M1",
        "Unnamed: 7",
        "Unnamed: 8",
        "Unnamed: 9",
        "Unnamed: 10",
        "WAXS M2",
        "Unnamed: 12",
        "Unnamed: 13",
        "Unnamed: 14",
        "Unnamed: 15",
        "WAXS E1",
        "Unnamed: 17",
        "Unnamed: 18",
        "Unnamed: 19",
        "Unnamed: 20",
        "WAXS E2",
        "Unnamed: 22",
        "Unnamed: 23",
        "Unnamed: 24",
        "Unnamed: 25",
    ]
]
df_R186_sub_selected = df_R186_sub[
    [
        "WAXS M1",
        "Unnamed: 7",
        "Unnamed: 8",
        "Unnamed: 9",
        "Unnamed: 10",
        "WAXS M2",
        "Unnamed: 12",
        "Unnamed: 13",
        "Unnamed: 14",
        "Unnamed: 15",
        "WAXS E1",
        "Unnamed: 17",
        "Unnamed: 18",
        "Unnamed: 19",
        "Unnamed: 20",
        "WAXS E2",
        "Unnamed: 22",
        "Unnamed: 23",
        "Unnamed: 24",
        "Unnamed: 25",
    ]
]
df_R187_sub_selected = df_R187_sub[
    [
        "WAXS M1",
        "Unnamed: 7",
        "Unnamed: 8",
        "Unnamed: 9",
        "Unnamed: 10",
        "WAXS M2",
        "Unnamed: 12",
        "Unnamed: 13",
        "Unnamed: 14",
        "Unnamed: 15",
        "WAXS E1",
        "Unnamed: 17",
        "Unnamed: 18",
        "Unnamed: 19",
        "Unnamed: 20",
        "WAXS E2",
        "Unnamed: 22",
        "Unnamed: 23",
        "Unnamed: 24",
        "Unnamed: 25",
    ]
]

# Add a new column 'label' to each DataFrame
df_R1_sub_selected.loc[:, "label"] = 0
df_R91_sub_selected.loc[:, "label"] = 1
df_R186_sub_selected.loc[:, "label"] = 2
df_R187_sub_selected.loc[:, "label"] = 3

# Concatenate the selected columns from each DataFrame
df_x_waxs = pd.concat(
    [
        df_R1_sub_selected,
        df_R91_sub_selected,
        df_R186_sub_selected,
        df_R187_sub_selected,
    ],
    axis=0,
)

# Create the target variable y
y_waxs = df_x_waxs["label"]

# Remove the 'label' column from df_x_waxs
x_waxs = df_x_waxs.drop("label", axis=1)


# %%

# Naive Bayes Classifier

# Create a Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Create a LeaveOneOut object
loo = LeaveOneOut()

accuracies = []
cms = []

# labels
labels = [0, 1, 2, 3]

# Perform the leave-one-out cross-validation
for train_index, test_index in loo.split(x_saxs):
    x_train, x_test = x_saxs.iloc[train_index], x_saxs.iloc[test_index]
    y_train, y_test = y_saxs.iloc[train_index], y_saxs.iloc[test_index]

    # Train the classifier
    gnb.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = gnb.predict(x_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cms.append(cm)

# Calculate the average accuracy
average_accuracy = sum(accuracies) / len(accuracies)

print(f"Average accuracy: {average_accuracy}")

# Calculate the sum of confusion matrices
sum_cm = sum(cms)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(sum_cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Truth")

# %%

# Random Forest Classifier

# Create a RandomForestClassifier
rfc = RandomForestClassifier()

# Create a LeaveOneOut object
loo = LeaveOneOut()

accuracies = []
cms = []

# labels
labels = [0, 1, 2, 3]

# Perform the leave-one-out cross-validation
for train_index, test_index in loo.split(x_saxs):
    x_train, x_test = x_saxs.iloc[train_index], x_saxs.iloc[test_index]
    y_train, y_test = y_saxs.iloc[train_index], y_saxs.iloc[test_index]

    # Train the classifier
    rfc.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = rfc.predict(x_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cms.append(cm)

# Calculate the average accuracy
average_accuracy = sum(accuracies) / len(accuracies)

print(f"Average accuracy: {average_accuracy}")

# Calculate the sum of confusion matrices
sum_cm = sum(cms)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(sum_cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Truth")
