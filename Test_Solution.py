#!/usr/bin/env python
# coding: utf-8

# # - Using “Test_Data.csv”:
# - ------------------------------------------------------------------------------------------------ 

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


from sklearn.manifold import TSNE


# In[5]:


import seaborn as sns 


# In[6]:


Test_Data = pd.read_csv("Test_Data.csv")


# - ------------------------------------------------------------------------------------------------
# ## - 1.	Extract columns ‘A’,‘B’,’C’,’G’,’R’,’X6’ and ’X8’ for different classes using dictionary and save as a csv for each class. 

# In[7]:


Test_Data.head()


# ### - Extrcting Column A:

# In[8]:


A = Test_Data['A'] # Extracting column A
A = dict(A) # As given in problem statement converting A data into dictionary

A_df = pd.DataFrame(A,index=[0]) 

A = A_df.transpose()

A_df.to_csv('A.csv')

A

A.head()


# ### - Extrcting Column B:

# In[9]:


B = Test_Data['B'] # Extracting column A
B = dict(B) # As given in problem statement converting A data into dictionary

B_df = pd.DataFrame(B,index=[0]) 

B = B_df.transpose()

B_df.to_csv('B.csv')

B

B.head()


# ### - Extrcting Column C:

# In[10]:


C = Test_Data['C'] # Extracting column A
C = dict(C) # As given in problem statement converting A data into dictionary

C_df = pd.DataFrame(C,index=[0]) 

C = C_df.transpose()

C_df.to_csv('C.csv')

C

C.head()


# ### - Extrcting Column G:

# In[11]:


G = Test_Data['G'] # Extracting column A
G = dict(G) # As given in problem statement converting A data into dictionary

G_df = pd.DataFrame(G,index=[0]) 

G = G_df.transpose()

G_df.to_csv('G.csv')

G

G.head()


# ### - Extrcting Column R:

# In[12]:


R = Test_Data['R'] # Extracting column A
R = dict(R) # As given in problem statement converting A data into dictionary

R_df = pd.DataFrame(R,index=[0]) 

R = R_df.transpose()

R_df.to_csv('R.csv')

R

R.head()


# ### - Extrcting Column X6:

# In[13]:


X6 = Test_Data['X6'] # Extracting column A
X6 = dict(X6) # As given in problem statement converting A data into dictionary

X6_df = pd.DataFrame(X6,index=[0]) 

X6 = X6_df.transpose()

X6_df.to_csv('X6.csv')

X6

X6.head()


# ### - Extrcting Column X8:

# In[14]:


X8 = Test_Data['X8'] # Extracting column A
X8 = dict(X8) # As given in problem statement converting A data into dictionary

X8_df = pd.DataFrame(X8,index=[0]) 

X8 = X8_df.transpose()

X8_df.to_csv('X8.csv')

X8

X8.head()


# ## - 2.	Do a complete EDA on each class file. 

# In[14]:


A.info()


# In[67]:


B.info()


# In[68]:


C.info()


# In[69]:


G.info()


# In[70]:


R.info()


# In[71]:


X6.info()


# In[72]:


X8.info()


# In[16]:


A.shape


# In[61]:


B.shape


# In[62]:


C.shape


# In[63]:


G.shape


# In[64]:


R.shape


# In[65]:


X6.shape


# In[66]:


X8.shape


# In[17]:


A.describe()


# In[60]:


B.describe() 


# In[21]:


A[0].unique()


# In[58]:


B[0].unique()


# In[22]:


A[0].value_counts()


# In[59]:


B[0].value_counts()


# In[28]:


plt.hist(A[0],bins=10)
plt.xlabel('Values')
plt.ylabel('Parameter')
plt.title("Histogram for A")
plt.show()


# In[49]:


plt.plot(B[0],'o',markersize=2,alpha=0.2)
plt.title("Plot for B")
plt.xlabel("Parameter")
plt.ylabel("Values")
plt.show()


# In[50]:


sns.violinplot(data=C,inner=None)
plt.title('Violin plot for C')
plt.xlabel('Parameter')
plt.ylabel('values')
plt.show()


# In[52]:


sns.boxenplot(data=B)
plt.title('Boxen plot for B')
plt.xlabel('Parameter')
plt.ylabel('values')
plt.show()


# In[55]:


sns.boxplot(data=G,whis=10)
plt.title('Box plot for G')
plt.xlabel('Parameter')
plt.ylabel('values')
plt.show()


# In[74]:


plt.plot(X6[0],'o',markersize=2)
plt.title("Plot for X6")
plt.xlabel("Parameter")
plt.ylabel("Values")
plt.show()


# In[73]:


np.corrcoef(X6[0])


# ## - 3.	Build a classification model by using dimensionality reduction and feature selection techniques

# In[77]:


print(np.corrcoef(A[0]))
print(np.corrcoef(X6[0]))
print(np.corrcoef(X8[0]))


# In[83]:


sns.pairplot(A,hue=None,diag_kind='hist')
plt.show()


# In[89]:


A_TSNE = TSNE(A)
print(A_TSNE)


# In[90]:


B_TSNE = TSNE(B)
print(B_TSNE)


# In[91]:


C_TSNE = TSNE(C)
print(A_TSNE)


# In[92]:


X6_TSNE = TSNE(X6)
print(X6_TSNE)


# In[93]:


X8_TSNE = TSNE(X8)
print(X8_TSNE)


# ## - 4.	Create a ROC curve of the model.

# In[105]:


def pdf(x, std, mean):
    cons = 1.0 / np.sqrt(2*np.pi*(std**2))
    pdf_normal_dist = cons*np.exp(-((x-mean)**2)/(2.0*(std**2)))
    return pdf_normal_dist
x = np.linspace(0, 1, num=100)
good_pdf = pdf(x,0.1,0.4)
bad_pdf = pdf(x,0.1,0.6)

def plot_pdf(good_pdf, bad_pdf, ax):
    ax.fill(x, good_pdf, "g", alpha=0.5)
    ax.fill(x, bad_pdf,"r", alpha=0.5)
    ax.set_xlim([0,1])
    ax.set_ylim([0,5])
    ax.set_title("Probability Distribution", fontsize=14)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_xlabel('P(X="bad")', fontsize=12)
    ax.legend(["good","bad"])


# In[109]:


fig, ax = plt.subplots(1,1, figsize=(10,5))
plot_pdf(good_pdf, bad_pdf, ax)


# TRP = (True Positive/Total positive)
# 
# FPR = (False Positive/Total Negative)

# In[110]:


def plot_roc(good_pdf, bad_pdf, ax):
    #Total
    total_bad = np.sum(bad_pdf)
    total_good = np.sum(good_pdf)
    #Cumulative sum
    cum_TP = 0
    cum_FP = 0
    #TPR and FPR list initialization
    TPR_list=[]
    FPR_list=[]
    #Iteratre through all values of x
    for i in range(len(x)):
        #We are only interested in non-zero values of bad
        if bad_pdf[i]>0:
            cum_TP+=bad_pdf[len(x)-1-i]
            cum_FP+=good_pdf[len(x)-1-i]
        FPR=cum_FP/total_good
        TPR=cum_TP/total_bad
        TPR_list.append(TPR)
        FPR_list.append(FPR)
    #Calculating AUC, taking the 100 timesteps into account
    auc=np.sum(TPR_list)/100
    #Plotting final ROC curve
    ax.plot(FPR_list, TPR_list)
    ax.plot(x,x, "--")
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_title("ROC Curve", fontsize=14)
    ax.set_ylabel('TPR', fontsize=12)
    ax.set_xlabel('FPR', fontsize=12)
    ax.grid()
    ax.legend(["AUC=%.3f"%auc])


# In[111]:


fig, ax = plt.subplots(1,1, figsize=(10,5))
plot_roc(good_pdf, bad_pdf, ax)

