import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# import sklearn
TRAINING_LINE_NUMBER = 8000000  # Number of lines to be read from input files
# List of years for training and testing
YEARS = ['2002', '2003', '2004', '2005', '2006', '2007', '2008']
INPUT_FILE_PATH = ".\\flights\\"  # Unix path
SKIP_FIRST_LINE = True  # To skip the first line, as its the header

master = []
print("Reading into Pandas frame...")
try:
    for year in YEARS:
        path = os.path.join(INPUT_FILE_PATH, '%d.csv' % int(year))
        print("\n", path)
        dfPart = pd.read_csv(
            path, nrows=TRAINING_LINE_NUMBER, encoding="ISO-8859-1", skiprows=0, usecols=[
                u'Year',
                u'Month',
                u'DayofMonth',
                u'DayOfWeek',
                u'UniqueCarrier',
                u'DepTime',
                u'TailNum',
                u'Origin',
                u'Dest',
                u'DepDelay',
                # u'ArrDelay',
                u'Cancelled',
                #                 u'ArrTime',
                #                 u'ArrDelay',
                #                 u'Distance'
            ])
        print(len(dfPart))
        # Removing cancelled flights from each year
        dfPart = dfPart[dfPart['Cancelled'] == 0]
        rows = np.random.choice(
            np.random.permutation(dfPart.index.values), len(dfPart) // 3,
            replace=False)  # 33% sampling of training data
        print(rows)
        sampled_dfPart = dfPart.loc[rows]
        sampled_dfPart = dfPart
        master.append(sampled_dfPart)
        print
except Exception as e:
    print("Supplemental Data Import failed", e)

dfMaster = pd.concat(master, ignore_index=True)
master = []
dfPart = []

print("Total length - ", len(dfMaster))
del dfMaster['Cancelled']  # Column not needed

dfMaster.fillna(0, inplace=True)

print(dfMaster.head())

dfMaster['Year'] = dfMaster['Year'].astype('int')
dfMaster['Month'] = dfMaster['Month'].astype('int')
dfMaster['DayofMonth'] = dfMaster['DayofMonth'].astype('int')
dfMaster['DayofWeek'] = dfMaster['DayOfWeek'].astype('int')
dfMaster['DepTime'] = dfMaster['DepTime'].astype('int')
dfMaster['DepDelay'] = dfMaster['DepDelay'].astype('int')

df = dfMaster

print("Calculating classification label...")
df['label'] = 0
df.label[df.DepDelay >= 15] = 1
df.label[df.DepDelay < 15] = 0
print("Actual delayed flights  -", np.sum(dfMaster['label']) / len(dfMaster['label']))

del df['DepDelay']
del df['TailNum']

print("Dataframe shape - ", df.shape)
print("Columns -", df.columns)

print("Converting categorical data to numeric...")
for col in set(df.columns):
    if df[col].dtype == np.dtype('object'):
        print("Converting...", col)

        if col == 'UniqueCarrier':
            s = np.unique(df[col].values)
            UniqueCarrier = pd.Series([x[0] for x in enumerate(s)], index=s)
        if col == 'Dest':
            s = np.unique(df[col].values)
            Dest = pd.Series([x[0] for x in enumerate(s)], index=s)
        if col == 'Origin':
            s = np.unique(df[col].values)
            Origin = pd.Series([x[0] for x in enumerate(s)], index=s)


def getDest(inDest):
    out = []
    for x, y in inDest.iteritems():
        out.append(Dest._get_value(y))
    return out


#
# Function: getOrigin()
# Description: This function will convert the input categorical value to corresponding numeric key.
# Input: categorical value you want to convert
# Output: a numeric value corresponding to the value passed. It uses the list created previously for lookup.
#


def getOrigin(inOrign):
    out = []
    for x, y in inOrign.iteritems():
        out.append(Origin._get_value(y))
    return out


#
# Function: getCarrier()
# Description: This function will convert the input categorical value to corresponding numeric key.
# Input: categorical value you want to convert
# Output: a numeric value corresponding to the value passed. It uses the list created previously for lookup.
#


def getCarrier(inCarrier):
    out = []
    for x, y in inCarrier.iteritems():
        out.append(UniqueCarrier._get_value(y))
    return out


# Converting UniqueCarrier
df['UniqueCarrier'] = getCarrier(df['UniqueCarrier'])
print("UniqueCarrier completed.")

# Converting Dest
df['Dest'] = getDest(df['Dest'])
print("Dest completed.")

# Converting Origin
df['Origin'] = getOrigin(df['Origin'])
print("Origin completed.")

print("Conversion to numeric completed.")

print(df.info())

X = df[['Year', 'Month', 'DayofMonth', 'DayofWeek', 'DepTime', 'UniqueCarrier', 'Origin', 'Dest']]

Y = df['label']

print(X.head())
print(Y.head())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

classifier = RandomForestClassifier(random_state=42, n_estimators=100)

results = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred

reg = accuracy_score(y_test, y_pred)
print("accuracy:", reg)
matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

matrix = confusion_matrix(y_test, y_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16, 7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
            cmap=plt.cm.Greens, linewidths=0.2)

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()


print(matrix)
print(classification_report(y_test, y_pred))
print(reg)
