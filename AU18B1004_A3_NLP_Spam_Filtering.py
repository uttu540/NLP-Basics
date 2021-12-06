#!/usr/bin/env python
# coding: utf-8

# # Assignment -3 
# 
# #### Title : Prepare Bag of words to perform Spam Filtering in the given Text passage provided.
# 
# #### Description : The students are required to analyse the text passage provided, tokenize and Lementise the text. Now apply technique to prepare Bag of Words and apply spam filtering methods to provide spam filtering solution.
# 
# ##### Objective: Familiarity with NLP operations using NLTK and application in semantic analysis.
# 
# ##### Domain : Natural Language Processing.
# 
# Steps to be taken:
# 
# 1) Perform cleaning and tokenization.
# 
# 2) Perform lementization after doing POS (parts of speech identifications) to avoid errors.
# 
# 3) Prepare Bag of Words and perform Spam filtering from database of emails.

# ### Importing Libraries

# In[1]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt, re
import nltk


# #### Reading text file which contains the spam text

# In[2]:


text = open(r'C:\Users\utkar\OneDrive\Desktop\Machine Learning\SpamCollection.txt').read()


# Converting the text file into a csv file so that we can have columns and we can divide the text and the spam, non-spam label.

# In[3]:


df = pd.read_csv(r'C:\Users\utkar\OneDrive\Desktop\Machine Learning\SpamCollection.txt',sep='\t',header=None,names=['label','text'])
df.head()


# Converting ham and spam values of label column to 0 and 1, here we assign 0 for ham(not-spam) and 1 for spam

# In[4]:


df['label'] = df['label'].apply(lambda x:0 if x == "ham" else 1)


# In[5]:


df.head()


# In[6]:


df.shape


# Importing two python files which we created for simpyfying our task
# 
# nlp_tools contains function for lemmatization
# 
# contractions contains functions for expanding short forms into complete words ex. don't = do not

# In[7]:


import nlp_tools
import contractions


# Here we create a new column and apply contractions file function as well as nlp_tools file function on the given text

# In[8]:


df['clean_text'] = df['text'].apply(contractions.expand_contraction)


# In[9]:


df['clean_text'] = df['clean_text'].apply(nlp_tools.lemmatization_sentence)


# ##### Now we will create a list of the clean_text column and use CountVectorizer

# In[10]:


spam_filter = df['clean_text'].tolist()


# In[11]:


from sklearn.feature_extraction.text import CountVectorizer


# In[12]:


cv = CountVectorizer()


# In[13]:


cv.fit(spam_filter)


# In[14]:


X = cv.transform(spam_filter).toarray()


# In[15]:


len(cv.get_feature_names())


# In[16]:


y = df['label'].values


# Using train_test_split library to split the data into train and test dataset

# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y)


# We will use Naive Bayes algorithm for classifying the text into spam and not-spam

# In[18]:


from sklearn.naive_bayes import MultinomialNB


# In[19]:


model = MultinomialNB()


# In[20]:


model.fit(x_train,y_train)


# In[21]:


y_pred = model.predict(x_test)


# In[22]:


from sklearn import metrics
cr = metrics.classification_report(y_test,y_pred)
print(cr)


# ### The overall accuracy we achieve using naive bayes algorithm is 98 percent which is pretty good.

# In[23]:


testspam = "date wed NUMBER aug NUMBER NUMBER NUMBER NUMBER NUMBER from chris garrigues cwg dated NUMBER NUMBER"


# In[25]:



clean_spam = contractions.expand_contraction(testspam)

lemma_spam = nlp_tools.lemmatization_sentence(clean_spam)
vector_spam = cv.transform([lemma_spam]).toarray()
spam = model.predict(vector_spam)
prob = model.predict_proba(vector_spam)


# In[26]:


spam


# In[27]:


prob


# ### The test text is a spam text as we can see and the model predicts the same with a really good probability of 99%. Our model's testing with a testing dataset as well as the test text is done with good and accurate results.
