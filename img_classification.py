#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


# load resnet extracted features
# resnet_train = np.load('./data/extracted_features/resnet_train.npz')
# resnet_test = np.load('./data/extracted_features/resnet_test.npz')
resnet_train = np.load('./features/resnet_train.npz')
resnet_test = np.load('./features/resnet_test.npz')
resnet_train_features = resnet_train['arr_0']
resnet_train_labels = resnet_train['arr_1']
resnet_test_features = resnet_test['arr_0']
resnet_test_labels = resnet_test['arr_1']

# In[3]:


# load scattering nets extracted features
scatter_train = np.load('./features/scattering_train.npz')
scatter_test = np.load('./features/scattering_test.npz')
scatter_train_features = scatter_train['arr_0']
scatter_train_labels = scatter_train['arr_1']
scatter_test_features = scatter_test['arr_0']
scatter_test_labels = scatter_test['arr_1']


# In[4]:


print(scatter_train_features.shape, scatter_train_labels.shape, scatter_test_features.shape, scatter_test_labels.shape)


# In[5]:


from torchvision import models, datasets, transforms
from ipywidgets import IntProgress


# In[6]:


# load raw features
transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# # MNIST dataset
# train_dataset = datasets.MNIST('./data/MNIST', train=True, download=True,
#                                transform=transform)
# test_dataset = datasets.MNIST('./data/MNIST', train=False,
#                                transform=transform)

# # Fashion MNIST dataset
# train_dataset = datasets.FashionMNIST('./data/MNIST_fashion', train=True, download=True,
#                                transform=transform)
# test_dataset = datasets.FashionMNIST('./data/MNIST_fashion', train=False,
#                                transform=transform)

# Fashion MNIST dataset
train_dataset = datasets.FashionMNIST('./features/MNIST_fashion', train=True, download=True,
                               transform=transform)
test_dataset = datasets.FashionMNIST('./features/MNIST_fashion', train=False,
                               transform=transform)


# In[7]:


raw_train_features = train_dataset.data.reshape(-1,784)
raw_train_labels = train_dataset.train_labels
raw_test_features = test_dataset.data.reshape(-1,784)
raw_test_labels = test_dataset.train_labels


# In[8]:


raw_train_features.shape


# In[9]:


from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[10]:


resnet_features[:60000,:].shape


# In[11]:


# image classification - raw features
train_features = raw_train_features
train_labels = raw_train_labels
test_features = raw_test_features
test_labels = raw_test_labels

# logistic regression - raw features
lr = LogisticRegression()
lr.fit(train_features, train_labels)
lr_predict = lr.predict(test_features)
print("[Logistic Regression - raw features]\n\taccuracy score: %.4f" % accuracy_score(lr_predict, test_labels))
print("\tClassification report:\n%s\n" % classification_report(lr_predict, test_labels))
# Support Vector Machine - raw features
svm = SVC()
svm.fit(train_features, train_labels)
svm_predict = svm.predict(test_features)
print("[Support Vector Machine - raw features]\n\taccuracy score: %.4f" % accuracy_score(svm_predict, test_labels))
print("\tClassification report:\n%s\n" % classification_report(svm_predict, test_labels))
# Linear Discriminant Analysis - raw features
lda = LinearDiscriminantAnalysis()
lda.fit(train_features, train_labels)
lda_predict = lda.predict(test_features)
print("[Linear Discriminant Analysis - raw features]\n\taccuracy score: %.4f" % accuracy_score(lda_predict, test_labels))
print("\tClassification report:\n%s\n" % classification_report(lda_predict, test_labels))
# Random Forest - raw features
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(train_features, train_labels)
rfc_predict = rfc.predict(test_features)
print("[Random Forest - raw features]\n\taccuracy score: %.4f" % accuracy_score(rfc_predict, test_labels))
print("\tClassification report:\n%s\n" % classification_report(rfc_predict, test_labels))


# In[21]:


# image classification - resnet
# train_features = resnet_features[:60000,:]
# train_labels = resnet_labels[:60000]
# test_features = resnet_features[60000:,:]
# test_labels = resnet_labels[60000:]
train_features = resnet_train_features
train_labels = resnet_train_labels
test_features = resnet_test_features
test_labels = resnet_test_labels

# logistic regression - resnet extracted features
lr = LogisticRegression()
lr.fit(train_features, train_labels)
lr_predict = lr.predict(test_features)
print("[Logistic Regression - resnet extracted features]\n\taccuracy score: %.4f" % accuracy_score(lr_predict, test_labels))
print("\tClassification report:\n%s\n" % classification_report(lr_predict, test_labels))
# Support Vector Machine - resnet extracted features
svm = SVC()
svm.fit(train_features, train_labels)
svm_predict = svm.predict(test_features)
print("[Support Vector Machine - resnet extracted features]\n\taccuracy score: %.4f" % accuracy_score(svm_predict, test_labels))
print("\tClassification report:\n%s\n" % classification_report(svm_predict, test_labels))
# Linear Discriminant Analysis - resnet extracted features
lda = LinearDiscriminantAnalysis()
lda.fit(train_features, train_labels)
lda_predict = lda.predict(test_features)
print("[Linear Discriminant Analysis - resnet extracted features]\n\taccuracy score: %.4f" % accuracy_score(lda_predict, test_labels))
print("\tClassification report:\n%s\n" % classification_report(lda_predict, test_labels))
# Random Forest - resnet extracted features
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(train_features, train_labels)
rfc_predict = rfc.predict(test_features)
print("[Random Forest - resnet extracted features]\n\taccuracy score: %.4f" % accuracy_score(rfc_predict, test_labels))
print("\tClassification report:\n%s\n" % classification_report(rfc_predict, test_labels))


# In[13]:


# image classification - scattering nets
train_features = scatter_train_features
train_labels = scatter_train_labels
test_features = scatter_test_features
test_labels = scatter_test_labels

# logistic regression - scattering net extracted features
lr = LogisticRegression()
lr.fit(train_features, train_labels)
lr_predict = lr.predict(test_features)
print("[Logistic Regression - scattering net extracted features]\n\taccuracy score: %.4f" % accuracy_score(lr_predict, test_labels))
print("\tClassification report:\n%s\n" % classification_report(lr_predict, test_labels))
# Support Vector Machine - scattering net extracted features
svm = SVC()
svm.fit(train_features, train_labels)
svm_predict = svm.predict(test_features)
print("[Support Vector Machine - scattering net extracted features]\n\taccuracy score: %.4f" % accuracy_score(svm_predict, test_labels))
print("\tClassification report:\n%s\n" % classification_report(svm_predict, test_labels))
# Linear Discriminant Analysis - scattering net extracted features
lda = LinearDiscriminantAnalysis()
lda.fit(train_features, train_labels)
lda_predict = lda.predict(test_features)
print("[Linear Discriminant Analysis - scattering net extracted features]\n\taccuracy score: %.4f" % accuracy_score(lda_predict, test_labels))
print("\tClassification report:\n%s\n" % classification_report(lda_predict, test_labels))
# Random Forest - scattering net extracted features
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(train_features, train_labels)
rfc_predict = rfc.predict(test_features)
print("[Random Forest - scattering net features]\n\taccuracy score: %.4f" % accuracy_score(rfc_predict, test_labels))
print("\tClassification report:\n%s\n" % classification_report(rfc_predict, test_labels))


# In[ ]:




