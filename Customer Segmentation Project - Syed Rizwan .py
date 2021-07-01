#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


df = pd.read_csv("C:/Users/hp/Desktop/Mall_Customers.csv")
df.head
df.info
df.tail


# In[13]:


#Now the data is loaded and based on the given we are going to get the number of people who are male and female from the dataset 


# In[14]:


genders = df.Gender.value_counts()


# In[15]:


plt.figure(figsize=(10,20))


# In[16]:


sns.barplot(x=genders.index , y=genders.values)
plt.show()


# In[17]:


#Now we will split the groups based on the age so that we can clearly understand the specific age group customers are high


# In[18]:


age18_25 = df.Age[(df.Age<=25)&(df.Age>=18)]
age26_35 = df.Age[(df.Age<=35)&(df.Age>=26)]
age36_45 = df.Age[(df.Age<=45)&(df.Age>=36)]
age46_55 = df.Age[(df.Age<=55)&(df.Age>=46)]
age55above = df.Age[(df.Age>=56)]

x=["18-25","26-35","36-45","46-55","Above55"]
y=[len(age18_25.values), len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]

plt.figure(figsize=(15,6))
plt.title=("Number of customers and ages")
plt.xlabel=("Ages")
plt.ylabel=("Number of customers")
sns.barplot(x=x,y=y)
plt.show()


# In[19]:


#Now we are going to see the highest spending scores among the customers 


# In[20]:


ss1_20= df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=1) &(df["Spending Score (1-100)"]<=20)]
ss21_40= df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=21) &(df["Spending Score (1-100)"]<=40)]
ss41_60= df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=41) &(df["Spending Score (1-100)"]<=60)]
ss61_80= df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=61) &(df["Spending Score (1-100)"]<=80)]
ss81_100= df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=81) &(df["Spending Score (1-100)"]<=100)]

x=["1-20","21-40","41-60","61-80","81-100"]
y=[len(ss1_20.values), len(ss21_40.values),len(ss41_60.values),len(ss61_80.values),len(ss81_100.values)]

sns.barplot(x=x , y=y)
plt.figure(figsize=(10,20))
plt.title=("Spending scores of the customers")
plt.xlabel=("Spending Scores")
plt.ylabel=("score of customers")
plt.show()


# In[26]:


ai0_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=0)&(df["Annual Income (k$)"]<=30)]
ai31_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=31)&(df["Annual Income (k$)"]<=60)]
ai61_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=61)&(df["Annual Income (k$)"]<=90)]
ai91_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=91)&(df["Annual Income (k$)"]<=120)]
ai121_150 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=121)&(df["Annual Income (k$)"]<=150)]

x=["0-30","31-60", "61-90","91-120","121-150"]
y=[len(ai0_30.values), len(ai31_60.values), len(ai61_90.values),len(ai91_120.values), len(ai121_150.values)]


plt.figure(figsize=(15,6))
sns.barplot(x=x,y=y,)
plt.title=("Annual Income of customers")
plt.xlabel("Annual Income in k$")
plt.ylabel("Number of customers")
plt.show()

                                                               


# In[ ]:


#So the above graph numberr of customers with their annual income so there are 80% customers
#who have their annual income around 61-90$ who visit the mall 


# In[ ]:


#Now we will analyze the 2 variables which are Spending Scores and Annual Income


# In[27]:


x=df.iloc[:,[3,4]].values #Locating the values of Spending scores and Annual Income in the variable x


# In[28]:


x # values of spending scores and Annual Income


# In[29]:


#Now we have to find the number of clusters to be used the fundamental method which goes with unsupervised method is Elbow Method
#Elbow method is used to find out the optimal value to be used in Kmeans In the line chart it resembles the arm with the elbow 
#The inflection in the curve resembles the underlined model fits best at that point


# In[30]:


#Kmeans Algorithm is used with unsupervised learning in which unlabelled data is classified in various clusters 
#If the K value is 3 it is classified in to 3 different clusters


# In[52]:



from sklearn.cluster import KMeans
wcss=[] #Within cluster sum of squares



for i in range(1,11):
    kmeans=KMeans(n_clusters=i , init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    



plt.figure(figsize=(12,6))    
plt.grid()
plt.plot(range(1,11),wcss)
plt.xlabel('KValue')
plt.xticks(np.arange(1,11,1))
plt.ylabel('WCSS')
plt.show()


# In[ ]:


#so based on the above line chart the maximum curve is in (5) so we can take 5 clusters in our k means


# In[61]:


kmeansmodel=KMeans(n_clusters=5, init='k-means++', random_state=0)
y_kmeans=kmeansmodel.fit_predict(x)
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=80,c='red',label='customer1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=80,c='blue',label='customer2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=80,c='green',label='customer3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=80,c='orange',label='customer4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=80,c='pink',label='customer5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c="black",label="centroids")
plt.title(str('cluster of customers'))
plt.xlabel(str('Annual Income'))
plt.ylabel(str('Spending Scores'))
plt.legend()
plt.show()




# In[ ]:


#so we have classified the customers into 5 clusters through which we can see that customer1 is having average spending scores
#with the average income so this range of customers can be targeted in order to increase sales

