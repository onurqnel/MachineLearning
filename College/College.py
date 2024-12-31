import numpy as np 
import pandas as pd
from matplotlib.pyplot import subplots, show

# Read the college data set.
College = pd.read_csv('College.csv')

# Rename the first column to 'College' and set it as the index.
CollegeData = College.rename({'Unnamed: 0': 'College'}, axis=1)
CollegeData = CollegeData.set_index('College')

# Describe the college data set.
print(CollegeData.describe())

# Create a scatter plot marix of columns 'Top10perc', 'Apps', and 'Enroll'.
pd.plotting.scatter_matrix(CollegeData[['Top10perc', 'Apps', 'Enroll'],bins=100)])
show()

# Create a boxplots of Outstate versus Private.
CollegeData.boxplot(column='Outstate', by='Private')
show()

#Categorizes schools as 'Elite' ('Yes' or 'No') based on whether 'Top10perc' is above 0.5.
print(CollegeData['Top10perc'].describe())
CollegeData['Elite'] = pd.cut(CollegeData['Top10perc'], [0, 50, 100], labels=['No', 'Yes'])
EliteCount = CollegeData['Elite'].value_counts()
print(EliteCount)

# Create a boxplots of Outstate versus Elite.
CollegeData.boxplot(column='Outstate', by='Elite')
show()

# Create a histogram of all quantitative variables.
QuantitativeVars = CollegeData.columns[1:]
CollegeData[QuantitativeVars].hist(bins=100)
show()