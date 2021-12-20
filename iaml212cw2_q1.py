
##########################################################
#  Python script template for Question 1 (IAML Level 10)
#  Note that
#  - You should not change the name of this file, 'iaml212cw2_q1.py', which is the file name you should use when you submit your code for this question.
#  - You should write code for the functions defined below. Do not change their names.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define additional functions, do not define them here, but put them in a separate Python module file, "iaml212cw2_my_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission.
##########################################################

#--- Code for loading modules and the data set and pre-processing --->
# NB: You can edit the following and add code (e.g. code for loading sklearn) if necessary.

import numpy as np
import scipy
from scipy.stats import multivariate_normal
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.stats.stats import pearsonr
from numpy import linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from iaml_cw2_helpers import *
#from iaml212cw2_my_helpers import *

X, Y = load_Q1_dataset()
#print('X:', X.shape, 'Y:', Y.shape)
Xtrn = X.iloc[100:, :]
Ytrn = Y.iloc[100:]    #training data set
Xtst = X.iloc[0:100, :]
Ytst = Y.iloc[0:100]  #test data set

# Create arrays from Xtrn and Ytrn dataframes
Xtrn_arr = Xtrn.to_numpy()
Xtst_arr = Xtst.to_numpy()
Ytrn_arr = Ytrn.to_numpy()
Ytst_arr = Ytst.to_numpy()

# Q1.1
def iaml212cw2_q1_1():
    fig, ax = plt.subplots(3, 3, figsize=(7, 7))
    ax = ax.flatten()
    fig.subplots_adjust(hspace=0.45, wspace=0.4)
    #fig.suptitle('')
    trn_df = pd.concat([Xtrn, Ytrn],axis=1)
    #print(Xtrn.shape[1])
    Xa = trn_df[trn_df.Y == 0].drop(['Y'], axis=1)
    Xb = trn_df[trn_df.Y == 1].drop(['Y'], axis=1)
    print('Xa:', Xa.shape, 'Xb:', Xb.shape)
    handles = []
    labels = ['Class 0', 'Class 1']
    for i in range(Xtrn.shape[1]):
        ax[i].hist([Xa.iloc[:,i], Xb.iloc[:,i]], bins=15)
        ax[i].set_title('A' + str(i))
        ax[i].grid(True)
        handle = ax[i].get_legend_handles_labels()[0]
        handles.append(handle)
        #labels.append(label)
    ax[3].set_ylabel("Frequency", fontsize=12, labelpad=8)
    ax[7].set_xlabel("Attribute values", fontsize=12, labelpad=7)
    #print(handles)
    #print(labels)
    #ax.legend(loc='upper right', ncol=2)
    fig.legend(handles, labels=labels, loc='lower right', ncol=2)
    plt.suptitle("Attribute Histograms", fontsize=13)
    plt.show()

#iaml212cw2_q1_1()   # comment this out when you run the function

# Q1.2
def iaml212cw2_q1_2():
    corr_coeff = np.zeros(Xtrn.shape[1])
    for i in range(Xtrn.shape[1]):
        corr_coeff[i] = Xtrn['A'+str(i)].corr(Ytrn)
        print('Correlation coefficient between attribute A' + str(i) + ' of Xtrn and Ytrn is ' + str(corr_coeff[i]))
#iaml212cw2_q1_2()   # comment this out when you run the function

# Q1.4
def iaml212cw2_q1_4():
    #ub_var = np.zeros(Xtrn.shape[1])
    #for i in range(Xtrn.shape[1]):
    #    ub_var[i] = np.var(Xtrn.iloc[:,i], ddof=1)
    #   print('Unbiased sample variance of attribute A' + str(i) + ' of Xtrn is ' + str(ub_var[i]))
    print("Unbiased sample variance of each attribute of Xtrn is:")
    Xtrn_var = np.var(Xtrn, ddof=1)
    print(Xtrn_var)
    #ub_var_descending = np.sort(ub_var)[::-1]
    #print(ub_var_descending)
    Xtrn_var_sum = np.sum(Xtrn_var)
    print("Sum of variances: ", str(Xtrn_var_sum))
    print(Xtrn_var.shape)

    attributes_names = Xtrn.columns.tolist()
    var_tuples = tuple(zip(attributes_names, Xtrn_var))
    var_tuples_descending = sorted(var_tuples, key=lambda x: x[1], reverse=True)
    print(var_tuples_descending)
    attributes_list = []
    #ratios_list = []
    ratios_list = np.zeros(Xtrn_var.shape)
    # for attribute, variance in var_tuples_descending:
    #     attributes_list.append(attribute)
    #     ratios_list.append(variance/Xtrn_var_sum)
    for idx, tpl in enumerate(var_tuples_descending):
        attributes_list.append(tpl[0])
        if idx==0:
            ratios_list[idx] = tpl[1]/Xtrn_var_sum
        else:
            ratios_list[idx] = ratios_list[idx-1] + tpl[1]/Xtrn_var_sum

    explained_variance_ratios = tuple(zip(attributes_list,ratios_list))
    print(explained_variance_ratios)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax = ax.flatten()

    unzipped_list = list(zip(*var_tuples_descending))
    ax[0].plot(unzipped_list[0], unzipped_list[1])
    ax[0].set_title('Explained variance per attribute')
    ax[0].set_xlabel("Attributes")
    ax[0].set_ylabel("Explained variance")
    unzipped_list2 = list(zip(*explained_variance_ratios))
    ax[1].plot(unzipped_list2[0], unzipped_list2[1])
    ax[1].set_title('Cumulative explained variance ratio')
    ax[1].set_xlabel("Attributes")
    ax[1].set_ylabel("Explained variance ratio")
    ax[0].grid(True)
    ax[1].grid(True)

    plt.show()

#iaml212cw2_q1_4()   # comment this out when you run the function

# Q1.5
def iaml212cw2_q1_5():
    pca = PCA()
    pca.fit(Xtrn)
    print(pca.explained_variance_ratio_)
    print("The unbiased sample variance explained by the whole set of "
         "principal components is: " + str(sum(pca.explained_variance_)))

    cum_expl_var_ratio = np.zeros(pca.explained_variance_ratio_.shape)
    for i in range(len(pca.explained_variance_ratio_)):
       if i == 0:
           cum_expl_var_ratio[i] = pca.explained_variance_ratio_[i]
       else:
           cum_expl_var_ratio[i] = cum_expl_var_ratio[i-1] + pca.explained_variance_ratio_[i]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax = ax.flatten()
    x_list = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9']
    ax[0].plot(x_list, pca.explained_variance_)
    ax[0].set_title('Explained variance per PC')
    ax[0].set_xlabel('Principal Components')
    ax[0].set_ylabel("Explained variance")
    ax[1].plot(x_list, cum_expl_var_ratio)
    ax[1].set_title('Cumulative explained variance ratio')
    ax[1].set_xlabel('Principal Components')
    ax[1].set_ylabel("Explained variance ratio")
    #ax[1].plot(x_list, pca.explained_variance_ratio_)
    ax[0].grid(True)
    ax[1].grid(True)
    plt.show()

    cov_matrix = pca.get_covariance()
    eigenvalues, eigenvectors = LA.eig(cov_matrix)

    PC1 = Xtrn_arr.dot(eigenvectors[:,0])
    PC2 = Xtrn_arr.dot(eigenvectors[:,1])

    #Ytrn and Xtrn share indices so we can use the Ytrn indices to split the PC's into the 2 classes
    PC1_label0 = PC1[Ytrn_arr == 0]
    PC1_label1 = PC1[Ytrn_arr == 1]
    PC2_label0 = PC2[Ytrn_arr == 0]
    PC2_label1 = PC2[Ytrn_arr == 1]

    plt.scatter(PC1_label0, PC2_label0, color= 'blue', label='Class 0')
    plt.scatter(PC1_label1, PC2_label1, color= 'red', label='Class 1')
    plt.suptitle("Instances in Xtrn on the space spanned by PC1 and PC2")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()
    plt.legend()
    plt.show()

    #Create a 9x2 matrix M where M[i,j] is the correlation coefficient of attribute i and PC j.
    corr_matrix = np.zeros((Xtrn_arr.shape[1], 2))
    for i in range(Xtrn_arr.shape[1]):
        corr_matrix[i][0] = pearsonr(Xtrn_arr[:,i], PC1)[0]
        corr_matrix[i][1] = pearsonr(Xtrn_arr[:,i], PC2)[0]
    print(corr_matrix)

#iaml212cw2_q1_5()   # comment this out when you run the function

scaler = StandardScaler().fit(Xtrn)
Xtrn_s = scaler.transform(Xtrn)
Xtst_s = scaler.transform(Xtst)

# Q1.6
def iaml212cw2_q1_6():

    pca = PCA()
    pca.fit(Xtrn_s)
    print(pca.explained_variance_ratio_)
    print("The unbiased sample variance explained by the whole set of "
          "principal components is: " + str(sum(pca.explained_variance_)))
    cum_expl_var_ratio = np.zeros(pca.explained_variance_ratio_.shape)
    for i in range(len(pca.explained_variance_ratio_)):
       if i == 0:
           cum_expl_var_ratio[i] = pca.explained_variance_ratio_[i]
       else:
           cum_expl_var_ratio[i] = cum_expl_var_ratio[i-1] + pca.explained_variance_ratio_[i]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax = ax.flatten()
    x_list = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9']
    ax[0].plot(x_list, pca.explained_variance_)
    ax[0].set_title('Explained variance per PC')
    ax[0].set_xlabel('Principal Components')
    ax[0].set_ylabel("Explained variance")
    ax[1].plot(x_list, cum_expl_var_ratio)
    ax[1].set_title('Cumulative explained variance ratio')
    ax[1].set_xlabel('Principal Components')
    ax[1].set_ylabel("Explained variance ratio")
    ax[0].grid(True)
    ax[1].grid(True)
    plt.show()

    cov_matrix = pca.get_covariance()
    eigenvalues, eigenvectors = LA.eig(cov_matrix)

    PC1 = Xtrn_s.dot(eigenvectors[:, 0])
    PC2 = Xtrn_s.dot(eigenvectors[:, 1])

    # Ytrn and Xtrn share indices so we can use the Ytrn indices to split the PC's into the 2 classes
    PC1_label0 = PC1[Ytrn_arr == 0]
    PC1_label1 = PC1[Ytrn_arr == 1]
    PC2_label0 = PC2[Ytrn_arr == 0]
    PC2_label1 = PC2[Ytrn_arr == 1]

    plt.scatter(PC1_label0, PC2_label0, color='blue', label='Class 0')
    plt.scatter(PC1_label1, PC2_label1, color='red', label='Class 1')
    plt.suptitle("Instances in Xtrn_s on the space spanned by PC1 and PC2")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()
    plt.legend()
    plt.show()

    # Create a 9x2 matrix M where M[i,j] is the correlation coefficient of attribute i and PC j.
    corr_matrix = np.zeros((Xtrn_s.shape[1], 2))
    for i in range(Xtrn_s.shape[1]):
        corr_matrix[i][0] = pearsonr(Xtrn_s[:, i], PC1)[0]
        corr_matrix[i][1] = pearsonr(Xtrn_s[:, i], PC2)[0]
    print(corr_matrix)


#iaml212cw2_q1_6()   # comment this out when you run the function

# Q1.8
def iaml212cw2_q1_8():
    c_list = np.logspace(np.log10(0.01), np.log10(100), num=13, base=10)
    print(c_list)

    skf = StratifiedKFold(n_splits=5)

    means_train = []
    means_test = []
    stds_train = []
    stds_test = []

    for c in c_list:
        accuracy_train = []
        accuracy_test = []

        for train_index, test_index in skf.split(Xtrn_s, Ytrn_arr):
            X_train, X_test = Xtrn_s[train_index], Xtrn_s[test_index]
            Y_train, Y_test = Ytrn_arr[train_index], Ytrn_arr[test_index]
            svm = SVC(C=c, kernel='rbf')
            svm.fit(X_train, Y_train)
            score_train = svm.score(X_train, Y_train)
            accuracy_train.append(score_train)
            score_test = svm.score(X_test, Y_test)
            accuracy_test.append(score_test)

        means_train.append(np.mean(accuracy_train))
        stds_train.append(np.std(accuracy_train))
        means_test.append(np.mean(accuracy_test))
        stds_test.append(np.std(accuracy_test))

    #sns.set()
    plt.figure(figsize=(8, 5))
    plt.xscale(value='log')
    plt.errorbar(c_list, means_train, yerr=stds_train, fmt='-o', label="Train data")
    plt.errorbar(c_list, means_test, yerr=stds_test, fmt='-o', label="Test data")
    plt.xlabel('Penalty parameter C ')
    plt.ylabel('Mean cross-validation classification accuracy')
    #plt.xticks(c_list)  # Set label locations.
    plt.suptitle("Mean of cross-validation classification accuracy \n "
                 "against penalty parameter C")
    plt.legend()
    plt.grid(True)
    plt.show()

    best_performing_c = c_list[means_test.index(max(means_test))]

    print("The highest mean cross-validation accuracy is " + str(max(means_test)) + " which is for the regularization parameter C = "
          + str(best_performing_c))

    best_svm = SVC(C=best_performing_c, kernel='rbf')
    best_svm.fit(Xtrn_s, Ytrn)
    predictions = best_svm.predict(Xtst_s)
    correct_classifications = 0
    for i in range(len(predictions)):
        if predictions[i] == Ytst[i]:
            correct_classifications+=1
    print("number of instances classified correctly: " + str(correct_classifications))
    print("Classification accuracy: " + str(correct_classifications/len(Ytst)))


#iaml212cw2_q1_8()   # comment this out when you run the function

# Q1.9
def iaml212cw2_q1_9():
    #print(Ytrn_arr.shape)
    #print(Xtrn_arr[Ytrn_arr==0].shape)
    Ztrn_arr = Xtrn_arr[:,[4,7]][Ytrn_arr==0]
    Ztrn_arr = Ztrn_arr[Ztrn_arr[:,0] >=1]
    #print(Ztrn_arr[0:10,:])
    #print(Ztrn_arr[:,0])
    Z_mean = np.mean(Ztrn_arr, axis=0)
    print(Z_mean)
    Z_cov = np.cov(Ztrn_arr.T)
    print(Z_cov)

    X = Ztrn_arr[:, 0]
    Y = Ztrn_arr[:, 1]

    x, y = np.meshgrid(X, Y)
    position = np.dstack((x, y))
    rv = multivariate_normal(Z_mean, Z_cov)
    # levels = 11
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111)

    cnt = ax.contourf(x, y, rv.pdf(position), cmap='magma')
    plt.scatter(X, Y, c='#00CED1')
    plt.xlabel("A4", fontsize=13)
    plt.ylabel("A7", fontsize=13)
    plt.suptitle("Contours of the estimated Gaussian distribution \n"
                 "between attributes A4 and A7 with label 0 and their instances",
                 fontsize=14)
    plt.xlim(0, 60)
    plt.ylim(-1, 60)
    cbar = plt.colorbar(cnt)
    cbar.set_label('Probability density', labelpad=10)
    plt.grid()
    plt.show()

#iaml212cw2_q1_9()   # comment this out when you run the function

# Q1.10
def iaml212cw2_q1_10():
    Ztrn_arr = Xtrn_arr[:, [4, 7]][Ytrn_arr == 0]
    Ztrn_arr = Ztrn_arr[Ztrn_arr[:, 0] >= 1]

    X = Ztrn_arr[:, 0]
    Y = Ztrn_arr[:, 1]
    Z_mean = np.mean(Ztrn_arr, axis=0)
    Z_cov = np.array([[np.var(X, ddof=1), 0], [0, np.var(Y, ddof=1)]])
    print(Z_mean)
    print(Z_cov)

    x, y = np.meshgrid(X, Y)
    position = np.dstack((x, y))
    rv = multivariate_normal(Z_mean, Z_cov)
    # levels = 11
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111)

    cnt = ax.contourf(x, y, rv.pdf(position), cmap='magma')
    plt.scatter(X, Y, c='#00CED1')
    plt.xlabel("A4", fontsize=13)
    plt.ylabel("A7", fontsize=13)
    plt.suptitle("Contours of the estimated Gaussian distribution "
                 "between attributes A4 and A7 \n with label 0 and their instances "
                 "assuming a naive Bayes model", fontsize=14)
    cbar = plt.colorbar(cnt)
    cbar.set_label('Probability density', labelpad=10)
    plt.xlim(0, 60)
    plt.ylim(-1, 60)
    plt.grid()
    plt.show()
#iaml212cw2_q1_10()   # comment this out when you run the function
