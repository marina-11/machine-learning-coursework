
##########################################################
#  Python script template for Question 2 (IAML Level 10)
#  Note that
#  - You should not change the filename of this file, 'iaml212cw2_q2.py', which is the file name you should use when you submit your code for this question.
#  - You should write code for the functions defined below. Do not change their names.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define helper functions, do not define them here, but put them in a separate Python module file, "iaml212cw2_my_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission.
##########################################################

#--- Code for loading modules and the data set and pre-processing --->
# NB: You can edit the following and add code (e.g. code for loading sklearn) if necessary.

import numpy as np
import scipy
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from iaml_cw2_helpers import *
from iaml212cw2_my_helpers import *

Xtrn_org, Ytrn_org, Xtst_org, Ytst_org = load_Q2_dataset()
Xtrn = Xtrn_org / 255.0
Xtst = Xtst_org / 255.0
Ytrn = Ytrn_org - 1
Ytst = Ytst_org - 1
Xmean = np.mean(Xtrn, axis=0)
Xtrn_m = Xtrn - Xmean; Xtst_m = Xtst - Xmean  #Mean−normalised versions
#plt.imshow(Xtrn[0].reshape(28,28), cmap='Greys')
#plt.show()
#print(Xtrn.shape, Xtst.shape)
#print(Ytrn.shape, Ytst.shape)

# Q2.1
def iaml212cw2_q2_1():
    print("maximum pixel value in Xtrn: " + str(max(np.amax(Xtrn, axis=1))))
    print("minimum pixel value in Xtrn: " + str(min(np.amin(Xtrn, axis=1))))
    print("mean pixel value in Xtrn: " + str(np.mean(Xtrn)))
    print("standard deviation of pixel values in Xtrn: " + str(np.std(Xtrn)))

    print("maximum pixel value in Xtst: " + str(max(np.amax(Xtst, axis=1))))
    print("minimum pixel value in Xtst: " + str(min(np.amin(Xtst, axis=1))))
    print("mean pixel value in Xtst: " + str(np.mean(Xtst)))
    print("standard deviation of pixel values in Xtst: " + str(np.std(Xtst)))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax = ax.flatten()
    ax[0].imshow(Xtrn[0].reshape(28,28).T, cmap='Greys')
    #ax[0].set_title(chr(ascii_num(Ytrn[0])))
    ax[0].set_title("Class number: " + str(Ytrn[0]))
    ax[1].imshow(Xtrn[1].reshape(28,28).T, cmap='Greys')
    #ax[1].set_title(chr(ascii_num(Ytrn[1])))
    ax[1].set_title("Class number: " + str(Ytrn[1]))
    plt.show()
#iaml212cw2_q2_1()   # comment this out when you run the function

# Q2.3
def iaml212cw2_q2_3():

    ks = [3,5]
    classes = [0, 5, 8]

    for k in ks:
        fig, ax = plt.subplots(3, k, figsize=(10,5))
        kmeans = KMeans(n_clusters=k, random_state=0)
        fig.suptitle("Images of cluster centres for k = " + str(k))

        for idx, c in enumerate(classes):
            kmeans.fit(Xtrn[Ytrn==c])
            #print(kmeans.n_iter_)
            for i in range(k):
                ax[idx,i].imshow(kmeans.cluster_centers_[i,:].reshape(28,28).T, cmap='Greys')
                ax[0,i].set_title("Cluster " + str(i+1))

        plt.show()

#iaml212cw2_q2_3()   # comment this out when you run the function

# Q2.5
def iaml212cw2_q2_5():
    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(Xtrn_m, Ytrn)
    print('Classification accuracy on training set: {:.3f}'.format(lr.score(Xtrn_m, Ytrn)))
    print('Classification accuracy on test set: {:.3f}'.format(lr.score(Xtst_m, Ytst)))
    predictions = lr.predict(Xtst_m)
    misclassifications = np.zeros(26)

    for i in range(len(Ytst)):
        if predictions[i] != Ytst[i]:
            misclassifications[Ytst[i]] += 1
    print(misclassifications)

    max_values = []
    max_indices = []
    letters = []

    for n in range(5):
        maximum = max(misclassifications)
        max_idx = np.argmax(misclassifications)
        max_values.append(maximum)
        max_indices.append(max_idx)
        misclassifications[max_idx] = -1
        print(misclassifications)

    print(max_values)
    print(max_indices)
    print(misclassifications)

    for v in max_indices:
        letters.append(ascii_num(v))
    print(letters)
#print(Xtrn_m.shape)
#iaml212cw2_q2_5()   # comment this out when you run the function

# Q2.6 
def iaml212cw2_q2_6():
    from sklearn.feature_selection import RFE
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import GridSearchCV
    import time
    #start_time = time.time()

    #lr = LogisticRegression(max_iter=1000, random_state=0)
    #lr.fit(Xtrn_m, Ytrn)

    # min_features_to_select = 1  # Minimum number of features to consider
    # rfecv = RFECV(
    #     estimator=lr,
    #     step=2,
    #     cv=StratifiedKFold(3),
    #     scoring="accuracy",
    #     min_features_to_select=min_features_to_select,
    #     n_jobs=-1
    # )
    # transformed_data = rfecv.fit_transform(Xtrn_m, Ytrn)
    #
    # print("Optimal number of features : %d" % rfecv.n_features_)
    #print("Mean cross validation score : %d" % rfecv.cv_results_["mean_test_score"][rfecv.n_features_])

    # Plot number of features VS. cross-validation scores
    # plt.figure()
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score (accuracy)")
    # plt.plot(
    #     range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
    #     rfecv.grid_scores_,
    # )
    # plt.show()
    # print("--- %s seconds ---" % (time.time() - start_time))
    # trans = feature_selection()
    # param_grid = dict(penalty=['l1', 'l2'], C=[0.0001,0.001,0.01,0.1,1,10,100,1000],
    #                   solver=['lbfgs', 'liblinear'], max_iter=[1000, 2000])
    # grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=3, n_jobs=-1)
    #
    # grid_result = grid.fit(trans[0], Ytrn)
    # # Summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # print("Execution time: " + str((time.time() - start_time)) + ' seconds')
    #
    # trn_fitted_trans = trans[0]
    # tst_trans = trans[1]
    # print(grid_result.score(trn_fitted_trans, Ytrn))
    # print(grid_result.score(tst_trans, Ytst))

    # lr2 = LogisticRegression(penalty='l2', C=0.1, random_state=0, max_iter=1000, solver='lbfgs')
    # lr2.fit(Xtrn_m, Ytrn)
    # print(lr2.score(Xtrn_m,Ytrn))
    # print(lr2.score(Xtst_m, Ytst))

    train = [0.916, 0.845, 0.812, 0.831]
    test = [0.722, 0.747, 0.742, 0.749]
    labels = ['No optimization', 'HT', 'HT + FS\n[st=10,min_f=1]','HT + FS\n[st=1,min=372]' ]
    X = np.arange(4)
    #fig = plt.figure()
    #ax = fig.add_axes([0, 0, 1, 1])
    plt.bar(X + 0.00, train, color='royalblue', width=0.25, label='training set')
    #ax.bar(X + 0.25, data[1], color='g', width=0.25)
    plt.bar(X + 0.25, test, color='orange', width=0.25, label='test set')
    plt.legend()
    plt.suptitle('Classification accuracy on training and test set \n'
                 'for each different optimization method')
    #plt.xlabel('Optimization method')
    plt.ylabel('Classification accuracy')
    plt.ylim([0, 1.1])
    plt.xticks(X+0.125, labels)
    for x, y_trn, y_tst in zip(X, train, test):
        label_trn = "{:.3f}".format(y_trn)
        label_tst = "{:.3f}".format(y_tst)

        plt.annotate(label_trn,  # this is the text
                     (x, y_trn),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 5),  # distance from text to points (x,y)
                     ha='center',
                     size=7)  # horizontal alignment can be left, right or center

        plt.annotate(label_tst,  # this is the text
                     (x+0.26, y_tst),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 5),  # distance from text to points (x,y)
                     ha='center',
                     size=7)  # horizontal alignment can be left, right or center
    plt.show()

#iaml212cw2_q2_6()   # comment this out when you run the function

# Q2.7 
def iaml212cw2_q2_7():
    train_data_0 = Xtrn_m[Ytrn==0]
    mean = np.mean(train_data_0, axis=0)
    cov_matrix = np.cov(train_data_0.T, ddof=1)
    print(train_data_0.shape)
    print(mean.shape)
    print(cov_matrix.shape)
    print("minimum value in cov matrix: " + str(np.amin(cov_matrix)))
    print("maximum value in cov matrix: " + str(np.amax(cov_matrix)))
    print("mean value in cov matrix: " + str(np.mean(cov_matrix)))
    diagonals = np.diagonal(cov_matrix)
    print("minimum value in diagonal of cov matrix: " + str(np.amin(diagonals)))
    print("maximum value in diagonal of cov matrix: " + str(np.amax(diagonals)))
    print("mean value in diagonal of cov matrix: " + str(np.mean(diagonals)))

    plt.hist(diagonals, bins=15)
    plt.ylabel("Frequency")
    plt.xlabel("Values in diagonal of covariance matrix")
    plt.grid()
    plt.show()

    #rv = multivariate_normal(mean, cov_matrix)
    #pdf = rv.pdf(Xtst_m)
    #print(pdf)
#iaml212cw2_q2_7()   # comment this out when you run the function

# Q2.8 
def iaml212cw2_q2_8():
    train_data_0 = Xtrn_m[Ytrn == 0]
    mean = np.mean(train_data_0, axis=0)
    cov_matrix = np.cov(train_data_0.T, ddof=1)

    gmm = GaussianMixture(n_components=1, covariance_type='full')
    gmm.fit(train_data_0)
    log_likelihood_values = gmm.score_samples(Xtst_m)
    print(log_likelihood_values[0])
    print(gmm.covariances_.shape)
    print(train_data_0.shape)

    correct_trn_list = []
    correct_tst_list = []
    accuracies_trn_list = []
    accuracies_tst_list = []
    total_trn = len(Ytrn)
    total_tst = len(Ytst)
    for i in range(26):
        train_data = Xtrn_m[Ytrn==i]
        #test_data
        gm_model = GaussianMixture(n_components=1, covariance_type='full')
        gm_model.fit(train_data)
        predictions_trn = gm_model.predict(Xtrn_m)
        predictions_tst = gm_model.predict(Xtst_m)
        correct_trn = 0
        correct_tst = 0

        for i in range(total_trn):
            if predictions_trn[i] == Ytrn[i]:
                correct_trn += 1

        for i in range(total_tst):
            if predictions_tst[i] == Ytst[i]:
                correct_tst += 1

        correct_trn_list.append(correct_trn)
        correct_tst_list.append(correct_tst)
        accuracies_trn_list.append(correct_trn/total_trn)
        accuracies_tst_list.append(correct_tst/total_tst)
    print(correct_trn_list)
    print(correct_tst_list)
    print(accuracies_trn_list)
    print(accuracies_tst_list)
    print(total_tst)
    print(total_trn)
#iaml212cw2_q2_8()   # comment this out when you run the function
