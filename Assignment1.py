import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
#load data
data = pd.read_csv("mock_student_data.csv")
stats = data.describe()
print stats.to_latex()
counts = data.count()
missings = np.subtract(counts, 1000)
print missings
modes = data.mode(axis=0)
print modes.to_latex()

#plot histograms
plt.figure()
data[['Age']].dropna(axis=0).hist()
plt.savefig('graphics/Age.png')
data[['GPA']].dropna(axis=0).hist()
plt.savefig('graphics/GPA.png')
data[['Days_missed']].dropna(axis=0).hist()
plt.savefig('graphics/Days_missed.png')

#plot bar charts
plt.figure()
data.dropna(axis=0, subset = [['Gender']]).groupby('Gender').size().plot(kind='bar')
plt.savefig('graphics/Gender.png')
plt.figure()
data.dropna(axis=0, subset = [['Graduated']]).groupby('Graduated').size().plot(kind='bar')
plt.savefig('graphics/Graduated.png')
plt.figure()
data.dropna(axis=0, subset = [['State']]).groupby('State').size().plot(kind='bar')
plt.savefig('graphics/State.png')

#define base url
url_base = "https://api.genderize.io/?name="
#gender fill function. arg = row of dataframe
def gender_fill(x):
    if np.isnan(x[5]) == True:
        first_name = x[2]
        r = requests.get(url_base + first_name)
        pred_gen = r.json()['gender']
        x[5] = pred_gen
#apply gender fill
data.apply(gender_fill, axis = 1)
data.to_csv("dataoutput/gender_fill_data.csv")

#uncondotional means
unconmeans = data.copy()
unconmeans.update(unconmeans.iloc[:,5:8].fillna(unconmeans.iloc[:,5:8].mean()))

unconmeans.to_csv("dataoutput/A.csv")
#define conditional means. arg = list of column names
def conditional_means(by_list):
    conmeans = data.copy()
    age_means = conmeans.groupby(by_list)['Age'].mean()
    gpa_means = conmeans.groupby(by_list)['GPA'].mean()
    Days_missed_means = conmeans.groupby(by_list)['Days_missed'].mean()
    conmeans = conmeans.set_index(by_list)
    conmeans.update(conmeans['Age'].fillna(age_means))
    conmeans.update(conmeans['GPA'].fillna(gpa_means))
    conmeans.update(conmeans['Days_missed'].fillna(Days_missed_means))
    return conmeans

#run functions save output
conmeans = conditional_means(['Graduated'])
conmeans.to_csv("dataoutput/B.csv")
data.update(data['State'].fillna('None'))
extraconmeans = conditional_means(['Graduated','Gender','State'])
extraconmeans.to_csv("dataoutput/C.csv")