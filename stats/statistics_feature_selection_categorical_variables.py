
# Commented out IPython magic to ensure Python compatibility.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

churn_master_df = pd.read_csv('https://raw.githubusercontent.com/nabeel-ds/CSVs/main/churn_data_st.csv')

churn_df = churn_master_df.copy()

churn_df.head()

"""customerID - Customer ID

Gender - Male or Female

SeniorCitizen - Whether the customer is a senior citizen or not (1, 0)

tenure - Number of months the customer has stayed with the company

ServiceCount - Number of services/product customer has availed

Contract - The contract term of the customer (Month-to-month, One year, Two year)

PaperlessBilling - Whether the customer has paperless billing or not (Yes, No)

MonthlyCharges - The amount charged to the customer monthly

TotalCharges - The total amount charged to the customer

Churn - Whether the customer churned or not (Yes or No)
"""

churn_df.info()

print ("Rows     : " ,churn_df.shape[0])
print ("Columns  : " ,churn_df.shape[1])
print ("\nFeatures : \n" ,churn_df.columns.tolist())
print ("\nUnique values :  \n",churn_df.nunique())

"""## Null Hypothesis - There is no relationship between two categorical values

## Alternate Hypothesis - There is a relationship between the two categorical values

Questions we are going to answer through statistical test

Is there any relationship between user who have Churned to Gender of the user?

Is there any relationship between users who have opted for paperless billing to user who have churned?
"""

churn_df['gender'].value_counts()

sns.set(style="darkgrid")
sns.set_palette("hls", 3)
fig, ax = plt.subplots(figsize=(20,5))
ax = sns.countplot(x="gender", hue="Churn", data=churn_df)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/churn_df.shape[0]),
            ha="center")

"""## Note: Statistics alone cannot prove anything. All the tests we do are based on relative likelihood**"""

gender_churn_ct=pd.crosstab(index=churn_df['gender'],columns=churn_df['Churn'])

gender_churn_ct

gender_churn_ct.iloc[0].values

"""*** scipy.stats.chi2_contingency(observed, correction=True, lambda_=None)[source]

> Indented block






"""

from scipy import stats
(chi2, p, dof,_) = stats.chi2_contingency([gender_churn_ct.iloc[0].values,gender_churn_ct.iloc[1].values])

print ("chi2     : " ,chi2)
print ("p-value  : " ,p)
print ("Degree for Freedom : " ,dof)

"""Df |	0.5 |	0.10 |	0.05 | 0.02 
--| -- | -- | -- | --
1	| 0.455 |	2.706	| 3.841 |	5.412	
2	| 1.386 |	4.605 |	5.991	| 7.824	
3	| 2.366 |	6.251	| 7.815	| 9.837

*** chi-square statistics

****** X^2 = sum((Observed – Expected)^2 / Expected)
"""

pd.crosstab(index=churn_df['gender'],columns=churn_df['Churn'], margins=True)

"""** Is there any relationship between users who have opted for paperless billing to user who have churned? **"""

churn_df['PaperlessBilling'].value_counts()

sns.set(style="darkgrid")
sns.set_palette("hls", 3)
fig, ax = plt.subplots(figsize=(20,5))
ax = sns.countplot(x="PaperlessBilling", hue="Churn", data=churn_df)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/churn_df.shape[0]),
            ha="center")

pb_churn_ct=pd.crosstab(index=churn_df['PaperlessBilling'],columns=churn_df['Churn'])

pb_churn_ct

pb_churn_ct.iloc[0].values

(chi2, p, dof,_) = stats.chi2_contingency([pb_churn_ct.iloc[0].values,pb_churn_ct.iloc[1].values])

print ("chi2     : " ,chi2)
print ("p-value  : " ,p)
print ("Degree for Freedom : " ,dof)

