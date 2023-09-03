#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries 

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[12]:


train=pd.read_csv("S:\\Desktop\\New folder\\Data Science\\titanic\\train.csv",encoding='unicode_escape')
df=train.copy()
test=pd.read_csv("S:\\Desktop\\New folder\\Data Science\\titanic\\test.csv",encoding='unicode_escape')
train
test


# # Exploratory Data Analysis

# In[14]:


print(df.shape) 
df.head()
print(test.shape) 
test.head()


# In[17]:


df.info()
test.info()


# In[24]:


df.describe().style.background_gradient(cmap="Purples")


# In[25]:


df.describe(include="object")


# In[27]:


test.describe().style.background_gradient(cmap="Purples")


# In[29]:


test.describe(include="object")


# In[44]:


df.dtypes


# In[43]:


df.isnull().sum()


# In[45]:


test.isnull().sum()


# In[47]:


plt.figure()
sns.heatmap(df.corr().abs(),annot=True)


# In[49]:


df.Survived.value_counts()


# In[56]:


plt.figure(figsize=(5,10))
sns.histplot(df.Survived);
plt.title('Probability of survival')


# In[59]:


fare_class_cabin = df[['Pclass', 'Fare', 'Cabin']]
fare_class_cabin.Cabin.fillna('missing', inplace=True)
for i in range(len(fare_class_cabin)):
    if fare_class_cabin.Cabin.iloc[i] == 'missing':
        fare_class_cabin.Cabin.iloc[i] = 0
    else:
        fare_class_cabin.Cabin.iloc[i] = 1
        
fare_class_cabin.Cabin = fare_class_cabin.Cabin.astype('int')


# In[60]:


sns.stripplot(x='Pclass', y='Fare', data=fare_class_cabin)


# In[66]:


pd.crosstab(fare_class_cabin.Cabin, fare_class_cabin.Pclass)


# In[68]:


pd.crosstab(fare_class_cabin.Cabin, df.Survived)


# In[70]:


plt.figure();
sns.histplot(x=fare_class_cabin.Pclass, hue=df.Survived)


# In[73]:


#Sex
pd.crosstab(df.Sex, df.Survived)


# In[74]:


plt.figure(figsize=(8,6))
sns.histplot(hue=df.Sex, x=df.Survived)
plt.title('relation of sex and survival')


# In[75]:


plt.figure()
sns.kdeplot(data=df, x = 'Age',hue='Survived', fill=True)


# In[77]:


plt.figure(figsize=(10,6))
sns.regplot(data = df, x='Age', y = 'Survived')


# In[78]:


plt.figure(figsize=(10,6))
sns.swarmplot(data = df, x='Survived', y = 'Age')


# In[79]:


sns.pairplot(df, kind="scatter")


# In[80]:


sns.heatmap(df.corr() , annot = True)


# In[82]:


# Survival percentages of passengers by gender

labels = [("died", "male"),
          ("died", "female"),
          ("survived", "male"),
          ("survived", "female")]
sizes = survived_sex.values
colors = ['#c2c2f0', '#ffcc99', '#66b3ff', '#ff9999']
# Create Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,colors=colors, shadow=True, wedgeprops={'edgecolor': 'black'})
plt.show()


# In[83]:


plt.subplots(figsize=(10, 5))
sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", common_norm=False)
plt.xlabel('Age')
plt.ylabel('Number of Passangers')
plt.title('Survival Status by Age')
plt.legend(title='Survived', labels=['Died', 'Survived'])
plt.show()


# ### Death percentage of passenger based on their ticket class

# In[89]:


data = {
    'Pclass': [1, 1, 2, 2, 3, 3],
    'Sex': ['male', 'female', 'male', 'female', 'male', 'female'],
    'Count': [122, 94, 108, 76, 347, 144]
}

# Convert the data into a DataFrame
data_f = pd.DataFrame(data)

# Group the data by 'Pclass' and 'Sex', then calculate the total passenger count
grouped_data = data_f.groupby(['Pclass', 'Sex'])['Count'].sum().reset_index()

# Calculate the total passenger count for each Pclass
total_passengers = grouped_data.groupby('Pclass')['Count'].transform('sum')

# Calculate the percentage of passengers and add it to the DataFrame
grouped_data['Percentage'] = (grouped_data['Count'] / total_passengers) * 100

# Create a bar plot using sns.barplot
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=grouped_data, x='Pclass', y='Percentage', hue='Sex', palette='Set1')

# Add percentage values on top of each bar
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')

# Customize the plot
plt.xlabel('Pclass')
plt.ylabel('Percentage (%)')
plt.title('Pclass by Gender')

# Set the Y axis limit to 100
plt.ylim(0, 100)

# Show the plot
plt.show()


# #### Family (Siblings & Parents)

# In[92]:


pd.crosstab(df.SibSp, df.Survived)


# In[94]:


pd.crosstab(df.Parch, df.Survived)


# In[97]:


df['Family'] = df.SibSp + df.Parch
test['Family'] = test.SibSp + test.Parch
pd.crosstab(tdf.Family, df.Survived)


# In[100]:


pd.crosstab(df.Embarked, df.Survived)


# passenger from port C are more likely to survive

# # Modelling

# In[109]:


df = train.copy()


# In[110]:


from sklearn.ensemble import RandomForestClassifier

y = df["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(df[features])
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# In[111]:


output

