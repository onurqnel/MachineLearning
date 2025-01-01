import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the college data set.
College = pd.read_csv('College.csv')

# Rename the first column to 'College' and set it as the index.
CollegeData = College.rename({'Unnamed: 0': 'College'}, axis=1)
CollegeData = CollegeData.set_index('College')
print(CollegeData.describe())

# Acceptence rate
CollegeData['AcceptenceRate'] = (CollegeData['Accept'] / CollegeData['Apps']) * 100

# Total Student Count 
CollegeData['StudentCount'] = CollegeData['F.Undergrad'] + CollegeData['P.Undergrad'] + CollegeData['Enroll']

def classify_size(row):
    if row['StudentCount'] > 5925:
        return 'Large'
    elif row['StudentCount'] < 1502:
        return 'Small'
    else:
        return 'Medium'
CollegeData['InstutionSize'] = CollegeData.apply(classify_size, axis=1)

sns.scatterplot(x=CollegeData['Top10perc'], y=CollegeData['AcceptenceRate'], hue=CollegeData['Private'])
plt.title('Acceptance Rate vs Top 10% Enrollment Rate')
plt.xlabel('Top 10% Enrollment Rate')
plt.ylabel('Acceptance Rate of Instution')
plt.show()

sns.scatterplot(x=CollegeData['Top10perc'], y=CollegeData['AcceptenceRate'], hue=CollegeData['InstutionSize'])
plt.title('Acceptance Rate vs Top 10% Enrollment Rate')
plt.xlabel('Top 10% Enrollment Rate')
plt.ylabel('Acceptance Rate of Instution')
plt.show()

sns.scatterplot(x=CollegeData['Outstate'], y=CollegeData['Expend'], hue=CollegeData['InstutionSize'])
plt.title('International Student Tuition vs Instructional Expenditure Per Student')
plt.xlabel('International Student Tuition')
plt.ylabel('Instructional Expenditure Per Student')
plt.show()

sns.scatterplot(x=CollegeData['Outstate'], y=CollegeData['Expend'], hue=CollegeData['Private'])
plt.title('International Student Tuition vs Instructional Expenditure Per Student')
plt.xlabel('International Student Tuition')
plt.ylabel('Instructional Expenditure Per Student')
plt.show()

sns.scatterplot(x=CollegeData['PhD'], y=CollegeData['S.F.Ratio'], hue=CollegeData['Private'])
plt.title('Student/Instructor Ratio vs Instution PhD Rate')
plt.xlabel('Instution PhD Rate')
plt.ylabel('Student/Instructor Ratio')
plt.show()

sns.scatterplot(x=CollegeData['PhD'], y=CollegeData['S.F.Ratio'], hue=CollegeData['InstutionSize'])
plt.title('Student/Instructor Ratio vs Instution PhD Rate')
plt.xlabel('Instution PhD Rate')
plt.ylabel('Student/Instructor Ratio')
plt.show()


