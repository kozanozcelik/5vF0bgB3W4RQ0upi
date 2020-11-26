#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:45:22 2020

@author: kozanozcelik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

df = pd.read_csv(r'/Users/kozanozcelik/Documents/term-deposit-marketing-2020.csv')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df.head(10)
df.describe()

df['default'] = [1 if x == 'yes' else 0 for x in df.default]
df['housing'] = [1 if x == 'yes' else 0 for x in df.housing]
df['loan'] = [1 if x == 'yes' else 0 for x in df.loan]
df['y'] = [1 if x == 'yes' else 0 for x in df.y]

for i in range(len(df)):
    if df['month'][i] == 'jan':
        df['month'][i] = 1
    elif df['month'][i] == 'feb':
        df['month'][i] = 2
    elif df['month'][i] == 'mar':
        df['month'][i] = 3
    elif df['month'][i] == 'apr':
        df['month'][i] = 4
    elif df['month'][i] == 'may':
        df['month'][i] = 5
    elif df['month'][i] == 'jun':
        df['month'][i] = 6
    elif df['month'][i] == 'jul':
        df['month'][i] = 7
    elif df['month'][i] == 'aug':
        df['month'][i] = 8
    elif df['month'][i] == 'sep':
        df['month'][i] = 9
    elif df['month'][i] == 'oct':
        df['month'][i] = 10
    elif df['month'][i] == 'nov':
        df['month'][i] = 11
    elif df['month'][i] == 'dec':
        df['month'][i] = 12
    


df = pd.get_dummies(df, columns=['job','marital','education','contact','day','month'])


Y = df['y'].values.reshape(-1,1)
X = df.drop(columns='y')

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.3, random_state=42)

#%% SVM
from sklearn import svm

## Run classifier
clf = svm.SVC(kernel='rbf', C=10, gamma=0.1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

scores = cross_val_score(clf, X_test, y_test, cv=5)
mean_cross_val_score = scores.mean()
print('\n5-fold CV Score = %s'%round(mean_cross_val_score,2))

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix \n%s'%cm)

#%% Bayesian Ridge
from sklearn.linear_model import RidgeClassifierCV

## Run classifier
clf = RidgeClassifierCV()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

scores = cross_val_score(clf, X_test, y_test, cv=5)
mean_cross_val_score = scores.mean()
print('\n5-fold CV Score = %s'%round(mean_cross_val_score,2))

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix \n%s'%cm)

#%% SGD
from sklearn.linear_model import SGDClassifier

## Run classifier
clf = SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

scores = cross_val_score(clf, X_test, y_test, cv=5)
mean_cross_val_score = scores.mean()
print('\n5-fold CV Score = %s'%round(mean_cross_val_score,2))

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix \n%s'%cm)

#%% Perceptron
from sklearn.linear_model import SGDClassifier

## Run classifier
clf = SGDClassifier(loss='perceptron', eta0=1, learning_rate='constant', penalty=None)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

scores = cross_val_score(clf, X_test, y_test, cv=5)
mean_cross_val_score = scores.mean()
print('\n5-fold CV Score = %s'%round(mean_cross_val_score,2))

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix \n%s'%cm)

#%% Decision Tree
from sklearn import tree

## Run classifier
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

scores = cross_val_score(clf, X_test, y_test, cv=5)
mean_cross_val_score = scores.mean()
print('\n5-fold CV Score = %s'%round(mean_cross_val_score,2))

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix \n%s'%cm)

#%% AdaBoost
from sklearn.ensemble import AdaBoostClassifier

## Run classifier
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

scores = cross_val_score(clf, X_test, y_test, cv=5)
mean_cross_val_score = scores.mean()
print('\n5-fold CV Score = %s'%round(mean_cross_val_score,2))

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix \n%s'%cm)

#%% GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier

## Run classifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

scores = cross_val_score(clf, X_test, y_test, cv=5)
mean_cross_val_score = scores.mean()
print('\n5-fold CV Score = %s'%round(mean_cross_val_score,2))

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix \n%s'%cm)

#%% Multi-layer Perceptron (MLP)
from sklearn.neural_network import MLPClassifier

## Run classifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

scores = cross_val_score(clf, X_test, y_test, cv=5)
mean_cross_val_score = scores.mean()
print('\n5-fold CV Score = %s'%round(mean_cross_val_score,2))

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix \n%s'%cm)

#%%  Correlation
import seaborn as sns

correlation = df.corr()
fig, ax = plt.subplots(figsize=(20,16))
sns.heatmap(correlation, annot=False, linewidths=.01, ax=ax, cmap="YlGnBu")

# Variable Selection For Clustering
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

skb = SelectKBest(chi2, k=20).fit(X_scaled, Y_scaled)
X_new_scaled = skb.transform(X_scaled)
print(skb.get_support(indices=True))
X.iloc[:,skb.get_support(indices=True)].columns

# 10 variables that have the highest importance for the dependent variable
correlation_2 = X.iloc[:,skb.get_support(indices=True)].corr()
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(correlation_2, annot=False, linewidths=.01, ax=ax, cmap="YlGnBu")

#%%  Clustering
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.cm as cm

skor_silhouette = []
skor_inertia = []
for n_clusters in [2,3,4,5,6]:
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(12, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X_new_scaled)
    silhouette_avg = silhouette_score(X_new_scaled, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    skor_silhouette.append(silhouette_score(X_new_scaled, cluster_labels))
    skor_inertia.append(clusterer.inertia_)
    sample_silhouette_values = silhouette_samples(X_new_scaled, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.suptitle(("Silhouette Analysis for KMeans Clustering "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.show()

fig, (ax1) = plt.subplots(1, 1)
fig.set_size_inches(12, 7)
ax1.set_title("Elbow Method")
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("Silhouette Score")
plt.plot(np.arange(2,7), skor_silhouette)

fig, (ax1) = plt.subplots(1, 1)
fig.set_size_inches(12, 7)
ax1.set_title("Elbow Method")
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("Inertia")
plt.plot(np.arange(2,7), skor_inertia)

#%% Cluster Properties for n_clusters = 4

n_clusters = 4

clusterer = KMeans(n_clusters=n_clusters, random_state=10)
cluster_labels = clusterer.fit_predict(X_new_scaled)
silhouette_avg = silhouette_score(X_new_scaled, cluster_labels)
print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

clusters = clusterer.predict(X_new_scaled)
clusters = pd.DataFrame(clusters, columns=['clusters'])

X = X.merge(clusters, left_index=True, right_index=True)

# Cluster-0 has the highest average duration
X.pivot_table(values='duration', index='clusters', aggfunc='mean', fill_value=np.nan)

# People with housing loans in Cluster-0 have the highest duration, which shows high inclination for investment
X.pivot_table(values='duration', columns='housing', index='clusters', aggfunc='mean', fill_value=np.nan)

# People with personal loans in Cluster-0 have the highest duration, which shows high inclination for investment
X.pivot_table(values='duration', columns='loan', index='clusters', aggfunc='mean', fill_value=np.nan)

# Cluster-0 does not have the highest nor the lowest average balance which makes sense
X.pivot_table(values='balance', index='clusters', aggfunc='mean', fill_value=np.nan)

# Blue-collars in Cluster-0 have the highest duration, which shows high inclination for investment
X.pivot_table(values='duration', columns='job_blue-collar', index='clusters', aggfunc='mean', fill_value=np.nan)

# Single people in Cluster-3 have the highest duration, which shows high inclination for investment
X.pivot_table(values='duration', columns='marital_married', index='clusters', aggfunc='mean', fill_value=np.nan)

# Cluster-0 on the 1st month of the year tend to be more interested in investment
X.pivot_table(values='duration', columns='month_1', index='clusters', aggfunc='mean', fill_value=np.nan)

# Cluster-0 on the 10th month of the year tend to be more interested in investment
X.pivot_table(values='duration', columns='month_10', index='clusters', aggfunc='mean', fill_value=np.nan)








