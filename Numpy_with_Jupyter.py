#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Numpy
help(len)


# In[11]:


help(help)


# In[13]:


import numpy as np
numpy.__version__


# In[10]:


# List 
L=list(range(10))


# In[16]:


print(L)
type(l)


# In[21]:


l= [str(i) for i in L]
print(l)
type(l)
type(l[0])


# In[19]:


type(l[0])


# In[24]:


l3= [1,'2', False, 5.0]
[type(i) for  i in l3]


# In[26]:


np.array( [1,2,3,4,5])


# In[28]:


np.array([1,2,3,4,5],dtype='float32')


# In[29]:


#multidimensional array
np.array([range(i,i+4) for i in [2,3,4]])


# In[31]:


np.zeros(20,dtype=int)


# In[34]:


np.ones((5,5),dtype='float32')


# In[37]:


np.arange(0,20,2)


# In[41]:


# different dimensional array and their size
np.random.seed(0)
a1=np.random.randint(10,size=6)#one-dimensional
a2=np.random.randint(10,size=(3,4))#two-dimensional
a3=np.random.randint(10,size=(3,4,5))#three-dimensional
print(a3.ndim,"\n")
print(a3.shape,"\n")
print(a3.size,"\n")
print(a1,"\n")
print(a2,"\n")
print(a3,"\n")


# In[45]:


x=np.arange(9)
print(x)


# In[43]:


x[7::-3]


# In[46]:


grid=x.reshape(3,3)


# In[47]:


grid


# In[49]:


#reciprocals
np.random.seed(1)
def reciprocals_(values):
    output=np.empty(len(values))
    for i in range(len(values)):
        output[i]=1/values[i]
    return output
values=np.random.randint(1,10,size=5)
reciprocals_(values)
        


# In[70]:


# Selection Sorting
def selection_sort(x):
    for i in range(len(x)):
        swap= i + np.argmin(x[i:])
        (x[i],x[swap])=(x[swap],x[i])
        
    return x
            


# In[71]:


x=np.array([3,2,5,1,6,7])
selection_sort(x)


# In[ ]:


# data Manipulation with Pandas


# In[72]:


import pandas
pandas.__version__


# In[73]:


import pandas as pd


# In[74]:


data = pd.Series([5,6,7,8])
data


# In[75]:


data.values


# In[76]:


data.index


# In[78]:


# Creating a series and dataframe
rng=np.random.RandomState(42)
ser=pd.Series(rng.randint(0,10,4))
ser


# In[79]:


df=pd.DataFrame(rng.randint(0,10,(3,4)),columns=['A','B','c','D'])
df


# In[84]:


# index alignment
A=pd.DataFrame(rng.randint(0,10,(2,2)),columns=['A','B'])
A


# In[85]:


B=pd.DataFrame(rng.randint(0,10,(3,4)),columns=['A','B','C','D'])
B


# In[86]:


A+B


# In[ ]:





# In[88]:


#Null values
data=pd.Series([1,np.nan,'hello',None])
data.isnull()


# In[4]:


# Data Visualisation Library
# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# In[5]:


plt.style.use('classic')


# In[6]:


# Basic plotting example
x=np.linspace(0,10,100)
fig=plt.figure()
plt.plot(x,np.sin(x),'-')
plt.plot(x,np.cos(x),'--')


# In[7]:


fig.savefig('my_figure.png')


# In[24]:






plt.plot(x,np.sin(x),':g',label='Sinx')
plt.axis('equal')
plt.title("An example")
plt.xlabel("Ofcourse x")
plt.ylabel("ofcourse y")

plt.legend()


# In[11]:


plt.style.use('seaborn-whitegrid')


# In[12]:


fig=plt.figure()
ax=plt.axes()


# In[25]:


#Machine Learning
# Iris dataset
import seaborn as sns
iris=sns.load_dataset('iris')
iris.head()


# In[26]:


X_iris=iris.drop('species',axis=1)
X_iris.shape
X_iris.head()


# In[27]:


Y_iris=iris['species']
Y_iris.shape


# In[35]:


#simple regression
import numpy as np
rng=np.random.RandomState(42)
x=10*rng.rand(50)
y=2*x+rng.rand(50)
plt.scatter(x,y)


# In[2]:


#Machine Learning Project California Datasets
import pandas as pd
housing=pd.read_csv(r'C:\Users\MAYANK\Downloads\blackjack\housing1.csv\housing.csv')
housing.head()


# In[43]:


housing.shape


# In[44]:


housing.info()


# In[45]:


housing['ocean_proximity'].value_counts()


# In[46]:


housing.describe()


# In[55]:


#plotting histogram
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
plt.show()


# In[3]:


# Testing and Training Data
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing, test_size=0.2,random_state=42)


# In[69]:


#Making a copy to explore
housing=train_set.copy()
housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,
             s=housing["population"]/100
             , label="population", figsize=(10,7),
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()


# In[70]:


#correlation among attribute
corr_matrix=housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# In[77]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[78]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[7]:


# Data Cleaning
#Categorical Attributes
housing_cat = housing["ocean_proximity"]
housing_cat.head(10)


# In[8]:


housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]


# In[ ]:



#Classification
#MNIST dataset


# In[9]:





# In[ ]:




