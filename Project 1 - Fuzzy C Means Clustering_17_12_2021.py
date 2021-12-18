#!/usr/bin/env python
# coding: utf-8

# In[260]:


# import needed libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pylab as pl
import seaborn as sns 


# In[ ]:


# Understanding the Dataset
# age (#)
# sex : 1 = Male, 0 = Female (Binary)
# (cp) chest pain [type (4 values, Ordinal)]: 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic
# (trestbps) resting blood pressure (#)
# (chol) serum cholestoral in mg/dl (#)
# (fbs) fasting blood sugar > 120 mg/dl (Binary) [1 = true; 0 = false]
# (restecg) resting electrocardiographic results [values 0,1,2]
# (thalach) maximum heart rate achieved (#)
# (exang) exercise induced angina (Binary) [1 = yes; 0 = no]
# (oldpeak) = ST depression induced by exercise relative to rest (#)
# (slope) of the peak exercise ST segment (Ordinal) [ 1: upsloping, 2: flat , 3: downsloping)
# (ca) number of major vessels (0-3, Ordinal) colored by fluoroscopy
# (thal) maximum heart rate achieved (Ordinal) [3 = normal; 6 = fixed defect; 7 = reversable defect]


# In[261]:


# upload data 
df=pd.read_csv("C:/Users/j.elachkar/Desktop/data.csv")
df.head(3)


# # Preprocessing 

# In[262]:


# remove spaces from columns names
df.columns=df.columns.str.replace(' ','')
print(df.columns)


# In[263]:


# rename num column to become Target
df=df.rename(columns={'num':'Target'})
print (df.head(3))


# In[264]:


# check for number of rows/instances and columns/variables
df.shape


# In[265]:


# check data types of variables 
df.dtypes


# In[266]:


# check for imbalance data 
df.Target.value_counts()


# In[267]:


count_0 = len(df[df['Target']==0])
print(count_0)


# In[268]:


count_1=len(df[df['Target']==1])
print(count_1)


# In[269]:


Percentage_of_class_1=round((count_1/(count_1 + count_0))*100,2)
print(f"Percentage_of_class_1 is {Percentage_of_class_1}%")


# In[270]:


# replace special characters such as ? with 0 
df=df.replace('?',0)
df.head()


# In[271]:


# remove rows with value 0 related to features trestbps, chol,thalach
df.drop(index=df[(df['trestbps']==0) |(df['chol']==0) | (df['thalach']==0)].index,inplace=True)


# In[272]:


# remove duplicate rows 
df=df.drop_duplicates()
df.head()


# In[273]:


# age bracket histogram 
df['age'].hist()
pl.suptitle("Histogram by Age Group")


# In[274]:


# converting all datafrane column to the type of int64 dtype
df=df.astype(int)


# In[275]:


# visualized relationship between cholesterol , resting blood pressure (trestbps) and heart attack 
# No relationship 
sns.relplot(x=df['chol'],y=df['trestbps'],hue=df['Target'])


# In[276]:


# More of Data Exploration 
get_ipython().run_line_magic('matplotlib', 'inline')
Chest_Pain = ['typical_angina','atypical_angina','non-anginal_pain','asymptomatic']
conditions = [df['cp']==1,df['cp']==2,df['cp']==3,df['cp']==4]
df['chest_pain_description']=np.select(conditions,Chest_Pain)

chest_pain_count=df['chest_pain_description'].value_counts()
sns.set(style='darkgrid')
sns.barplot(chest_pain_count.index,chest_pain_count.values,alpha=0.9)
plt.title("Frequency of Chest Pain Types")
plt.ylabel('Number of occurences',fontsize=12)
plt.xlabel('Chest Pain Type',fontsize=12)
plt.show()


# In[277]:


# let us re-drop now the added column 
df=df.drop('chest_pain_description',axis=1)


# In[278]:


# drop target variable to plot the correlation matrix 
df1=df.drop('Target',axis=1)


# In[279]:


# check for correlation betweeen features ( high correlation > 0.95 ) can be removed ( no added value, it creates noise)
cor_mat=df1.corr()
f, ax =plt.subplots(figsize=(12,9))
sns.heatmap(cor_mat,vmax=.8,square=True,annot=True)


# In[280]:


# splitting data between features and target variable 
X=df.drop('Target',axis=1)
y=df[['Target']]


# In[237]:


get_ipython().system('pip install fuzzy-c-means')


# In[238]:


from fcmeans import FCM


# In[239]:


fcm=FCM(n_clusters=2)


# In[240]:


# to train X using Fuzzy C Means, we need to convert data type from data frome to array
X=X.to_numpy()
print(X)


# In[241]:


# according to our data, we have 2 clusters ( 0 and 1)
plt.figure(figsize=(5,5))
plt.scatter(X[:,0],X[:,1],alpha=.1)
plt.show()


# In[242]:


fcm.fit(X)


# In[243]:


# outputs 
fcm_centers=fcm.centers
fcm_labels=fcm.predict(X)


# In[244]:


from seaborn import scatterplot as scatter


# In[245]:


fcm_centers


# In[246]:


np.unique(fcm_labels)


# In[247]:


# plot result
f, axes = plt.subplots(1, 2, figsize=(11,5))
axes[0].scatter(X[:,0], X[:,1], alpha=.1)
axes[1].scatter(X[:,0], X[:,1], c=fcm_labels, alpha=.1)
axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='w')
plt.show()


# # Fitting the fuzzy C Means with multiple Clusters

# In[126]:


#create models with 2, 3, 4, 5, 6 and 6 centers
n_clusters_list = [2, 3, 4, 5, 6, 7]
models = list()
for n_clusters in n_clusters_list:
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(X)
    models.append(fcm)


# In[128]:


# outputs
#The FPC is defined on the range from 0 to 1, with 1 being best. 
#It is a metric which tells us how cleanly our data is described by a certain model
#The partition entropy (PE) [16] measures the fuzzy degree of final divided clusters by means
#of the fuzzy partition matrix, and the smaller its value, 
#the better the partition result. MPC (modified partition coefficient)
num_clusters = len(n_clusters_list)
rows = int(np.ceil(np.sqrt(num_clusters)))
cols = int(np.ceil(num_clusters / rows))
f, axes = plt.subplots(rows, cols, figsize=(11,16))
for n_clusters, model, axe in zip(n_clusters_list, models, axes.ravel()):
    # get validation metrics
    pc = model.partition_coefficient
    pec = model.partition_entropy_coefficient
    
    fcm_centers = model.centers
    fcm_labels = model.predict(X)
    # plot result
    axe.scatter(X[:,0], X[:,1], c=fcm_labels, alpha=.1)
    axe.scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='black')
    axe.set_title(f'n_clusters = {n_clusters}, PC = {pc:.3f}, PEC = {pec:.3f}')
plt.show()


# # KNN Model 

# In[281]:


#K is the number of nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=5)


# In[282]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# In[283]:


# feature scaling 
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[291]:


knc.fit(X_train,y_train.values.ravel())


# In[292]:


y_predict=knc.predict(X_test)
accuracy=accuracy_score(y_test,y_predict)
print(round(accuracy,2))


# In[287]:


from sklearn.metrics import plot_confusion_matrix,confusion_matrix


# In[290]:


print(confusion_matrix(y_test,y_predict))


# In[289]:


plot_confusion_matrix(knc,X_test,y_test.round())

