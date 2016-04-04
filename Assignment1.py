import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

data = pd.read_csv("mock_student_data.csv")
data.describe()
counts = data.count()
missings = counts - 1000

plt.figure()
data[['Age']].dropna(axis=0).hist()
plt.savefig('graphics/Age.png')
data[['GPA']].dropna(axis=0).hist()
print data['GPA'].dropna(axis=0)
plt.savefig('graphics/GPA.png')
data[['Days_missed']].dropna(axis=0).hist()
plt.savefig('graphics/Days_missed.png')

plt.figure()
data.dropna(axis=0, subset = [['Gender']]).groupby('Gender').size().plot(kind='bar')
plt.savefig('graphics/Gender.png')
plt.figure()
data.dropna(axis=0, subset = [['Graduated']]).groupby('Graduated').size().plot(kind='bar')
plt.savefig('graphics/Graduated.png')

url_base = "https://api.genderize.io/?name="
'''
def gender_fill(x):
    if np.isnan(x[5]) == True:
        first_name = x[2]
        r = requests.get(url_base + first_name)
        pred_gen = r.json()['gender']
        x[5] = pred_gen
        print 'watwat'
        print x[0]

data.apply(gender_fill, axis = 1)
data.to_csv("dataoutput/gender_fill_data.csv")
'''
print data.iloc[:,6:9]
data.update(data.iloc[:,6:9].fillna(data.iloc[:,6:9].mean()))
print data.iloc[:,6:9]

data.to_csv("dataoutput/Unconditional_means_data.csv")

age_means = data.groupby(['Graduated'])['Age'].mean()
gpa_means = data.groupby(['Graduated'])['GPA'].mean()
Days_missed_means = data.groupby(['Graduated'])['Days_missed'].mean()

data.set_index(['Graduated'])

data[['Age']] = data[['Age']].fillna(age_means)
data[['GPA']] = data[['GPA']].fillna(gpa_means)
data[['Days_missed']] = data[['Days_missed']].fillna(Days_missed_means)

data.to_csv("dataoutput/Conditional_means_data.csv")





