import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import copy
import seaborn as sn


def GetDefaultParameters():
    fold_1 = range(10)
    fold_2 = range(10, 20)
    data_path = "C:/Users/user/PycharmProjects/untitled/101_ObjectCategories"

    data = {
        'tune': True,
        'data_path': data_path,
        'class_indicies': fold_1,
        'S': 150,
        'split': 20,
        'max_images_per_class': 40,
        'valid_set': 0.2
    }
    prepare = {
        'n_clusters': 500,
        'step_size': 7,
        'scale': [5, 10, 15, 20],
        'img_subset_size': 0.7,
        'sift_subset_size': 0.1
    }
    HP_tune = {
        'kernel': ['linear', 'rbf', 'poly'],
        'c': range(1, 100, 5),
        'gamma': [1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30],
        'degree': [2, 3]
    }
    final_par = {
        'c': 21,
        'kernel': 'linear',
        'degree': 3,
        'gamma': 25
    }

    return {
        'data': data,
        'prepare': prepare,
        'HP_tune': HP_tune,
        'final_par': final_par
    }


# loading the data from the path
def get_data(params_data):
    """
    gets the data
    :param params_data: the relevant parameters.
    :return: A dict:
                dict Data which contains:
                    images (matrix, [NXSXS])
                    labels(vector, [NX1])
                    label_names(vector, [len(class_indicies)X1])
    '''
    """
    data_path = params_data['data_path']
    # The class labels should be sorted by name (ascending)
    label_names = os.listdir(data_path)
    images = []
    labels = []
    label_index = 0

    for class_index in params_data['class_indicies']:
        label_dir_path = f'{data_path}\\{label_names[class_index]}'
        images_filenames_path = [f'{label_dir_path}\\{image_filename}' for image_filename in os.listdir(label_dir_path)]

        # The images should be sorted by name (ascending), then first 20 images should be chosen for training
        sort_images_filenames = sorted(images_filenames_path)
        train_sampled_filenames = sort_images_filenames[0:20]
        remain_sampled_filenames = sort_images_filenames[20:len(images_filenames_path)]

        # get random samples of max 20 images for test
        test_sampled_filenames = random.sample(remain_sampled_filenames,
                                               min(params_data['split'], len(remain_sampled_filenames)))

        sampled_filenames = train_sampled_filenames + test_sampled_filenames

        for image_filename in sampled_filenames:
            images.append(get_image(image_filename, params_data['S']))
            labels.append(label_index)

        label_index = label_index + 1

    cur_label_names = [label_names[i] for i in params_data['class_indicies']]
    data = np.asarray(images)
    labels = np.asarray(labels)

    return {
        'data': data,
        'labels': labels,
        'label_names': cur_label_names
    }


def get_image(image_filename, S):
    raw_image = cv2.imread(image_filename)
    gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    resized_gray_image = cv2.resize(gray_image, (S, S))

    return resized_gray_image


def TrainTestSplit(data, labels, params_data):
    train_data = None
    train_labels = None
    test_data = None
    test_labels = None
    for i in np.unique(labels):  # run on every class
        tmp_data = data[labels == i, :, :]  # all the images of this class
        tmp_labels = labels[labels == i]  # all the labels of the class (equal to label)
        test_size = min(params_data['split'], len(tmp_labels) - params_data['split'])
        tmp_data_train = tmp_data[0:params_data['split'], :, :]
        tmp_labels_train = tmp_labels[0:params_data['split']]
        tmp_data_test = tmp_data[params_data['split']:(test_size + params_data['split']), :, :]
        tmp_labels_test = tmp_labels[params_data['split']:(test_size + params_data['split'])]

        if train_data is None:
            train_data = tmp_data_train
            test_data = tmp_data_test
            train_labels = tmp_labels_train
            test_labels = tmp_labels_test

        else:
            train_data = np.concatenate((train_data, tmp_data_train), axis=0)
            test_data = np.concatenate((test_data, tmp_data_test), axis=0)
            train_labels = np.append(train_labels, tmp_labels_train)
            test_labels = np.append(test_labels, tmp_labels_test)

    return {
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels
    }


def TrainValidSplit(data, labels, params_data):
    train_data = None
    train_labels = None
    valid_data = None
    valid_labels = None
    train_size = int(np.floor(params_data['split']) * (1 - params_data['valid_set']))
    for i in np.unique(labels):  # run on every class
        tmp_data = data[labels == i, :, :]  # all the images of this class
        tmp_labels = labels[labels == i]  # all the labels of the class (equal to label)
        valid_size = min(int(np.floor(params_data['split'] * params_data['valid_set'])),
                         len(tmp_labels) - int(np.floor(params_data['split'] * params_data['valid_set'])))
        tmp_data_train = tmp_data[0:train_size, :, :]
        tmp_labels_train = tmp_labels[0:train_size]
        tmp_data_valid = tmp_data[train_size:(valid_size + train_size), :, :]
        tmp_labels_valid = tmp_labels[train_size:(valid_size + train_size)]

        if train_data is None:
            train_data = tmp_data_train
            valid_data = tmp_data_valid
            train_labels = tmp_labels_train
            valid_labels = tmp_labels_valid

        else:
            train_data = np.concatenate((train_data, tmp_data_train), axis=0)
            valid_data = np.concatenate((valid_data, tmp_data_valid), axis=0)
            train_labels = np.append(train_labels, tmp_labels_train)
            valid_labels = np.append(valid_labels, tmp_labels_valid)

    return {
        'train_data': train_data,
        'train_labels': train_labels,
        'valid_data': valid_data,
        'valid_labels': valid_labels
    }


def Prepare(data, params_prepare, labels, img_dictonary=None):
    '''
    Represent each image I as a codeword histogram H using pre-calculated Kmeans from training data
    :param Data: Array of Images SIFTs
    :param params_prepare: Hyper-parametes for the preparation process
    :param img_dictonary: the img_dictonary-kmean model for the test set. Will be None for the train set
    :return: Codeword Histograms for each image in the given array
    '''

    if img_dictonary is None:
        sub_data = subset_images(data, labels, params_prepare['img_subset_size'])
        array_sifts_sub_data = Data2SIFTs(sub_data, params_prepare)
        sub_array_sifts_train = subset_sifts(array_sifts_sub_data, params_prepare['sift_subset_size'])
        img_dictonary = KmeansDictonary(params_prepare['n_clusters'], sub_array_sifts_train)

    array_sifts = Data2SIFTs(data, params_prepare)

    X = []
    for img in array_sifts:
        img_sifts = np.asarray(img[1].tolist())
        h = img_dictonary.predict(img_sifts)

        xi = np.zeros(img_dictonary.get_params()['n_clusters'])
        for hi in h:
            xi[hi] = xi[hi] + 1

        X.append(xi)
    X = np.asarray(X)
    # normalize by column:
    H = X / (X.sum(axis=1)[0])

    return H, img_dictonary


def subset_images(data, labels, img_sub):
    # training the dictionary from subset of images
    num_of_class = len(np.unique(labels))
    split_classes = np.split(data, num_of_class, axis=0)
    reduced_sub = []
    for c in split_classes:
        reduced_sub.append(c[0:int(np.floor(len(c) * img_sub)), :, :])
    sub_data = np.concatenate(reduced_sub, axis=0)
    return sub_data


def Data2SIFTs(data, params_prepare):
    '''
    Takes subset of images calculate SIFTs for each.
    :param Data: Images
    :param params: Hyper-parametes for the preparation process
    :return: return list of images SIFTs
    '''
    # SIFT - dense extraction
    sift = cv2.xfeatures2d.SIFT_create()
    img_sifts = []

    for gray_img in data:
        kp = []
        step = params_prepare['step_size']
        for scale in params_prepare['scale']:
            kp = np.append(kp, [cv2.KeyPoint(x, y, scale) for y in range(0, gray_img.shape[0], step)
                                for x in range(0, gray_img.shape[1], step)])
        dense_feat = sift.compute(gray_img, kp)
        img_sifts.append(dense_feat)
    return img_sifts


def subset_sifts(img_sifts, sift_sub_size):
    # creating subset of SIFTs and collect them to one array
    num_of_sifts_img = int(np.floor(sift_sub_size * len(img_sifts[1][0][:])))
    idx = np.random.randint(len(img_sifts[1][0][:]), size=num_of_sifts_img)  # choose the subset of img SIFT randomly
    sub_array_sifts = []
    for img in img_sifts:
        img_sift = img[1][idx, :].tolist()
        for s in img_sift:
            sub_array_sifts.append(s)
    sub_sifts = np.asarray(sub_array_sifts)
    return sub_sifts


def KmeansDictonary(n_clusters, array_all_sifts):
    '''
    Compute the dictionary: receives training data as array of SIFTs and compute Kmeans dictionary
    :param n_clusters: number of clusters fo Kmeans
    :param array_all_sifts: training data as array of SIFTs
    :return Kmeans dictionary
    '''
    # apply K means algorithm - building the dictionary
    img_dictonary = KMeans(n_clusters=n_clusters, random_state=123).fit(array_all_sifts)
    return img_dictonary


def TuneHyperParameters(train_data, train_labels, valid_data, valid_labels, params_HP_tune):
    '''
        Tuning hyper-parameters, tune each kernel separately, tune only one parameter at a time.
    :param train_data: data for training  (matrix, [train_size, n_centroids])
    :param train_labels: Labels for training (vector, [train_size, 1])
    :param valid_data: data for validation (matrix, [test_size, n_centroids])
    :param valid_labels: Labels for validation (vector, [test_size, 1)
    :param params_HP_tune: Dict containing all the possible hyper-parameters for the model:
            1. kernel - possible kerenels (list)
            2. c - possible penalty values (list)
            3. gamma - Kernel coefficient for ‘rbf’, ‘poly’ (list)
            4. degree - possible degrees for the poly kernel (list)
    :return: A dict containing a summary of the tuning process:
            1. The min test error
            2. The min train error
            3. A dict containing the chosen hyper-parameters:
            4. A dict containing a summary of the tuning process:
                4.1. The tuned parameter
                4.2. The fixed parameters
                4.3. The resulted error for each value of the tuned parameter
                4.4. The  tuned parameter that had the minimum error
    '''
    Data = {
        'train_data': train_data,
        'valid_data': valid_data,
        'train_labels': train_labels,
        'valid_labels': valid_labels,
    }

    HP = {i: params_HP_tune[i][0] for i in params_HP_tune} # initialize hyper-parameters
    summary_min_error_test = {'linear': None, 'rbf': None, 'poly': None}
    summary_min_error_train = {'linear': None, 'rbf': None, 'poly': None}
    summary_min_hp = {'linear': [], 'rbf': [], 'poly': []}
    tune_summary = []

    for k in params_HP_tune['kernel']:  # run on every kernel separately
        delta_error = 1  # change in error
        min_error = 1
        min_train_error = 1
        HP['kernel'] = k
        if k == 'linear':
            hp_params = 'c'
        elif k == 'rbf':
            hp_params = ['gamma', 'c']
        else:
            hp_params = ['degree', 'gamma', 'c']
        while delta_error > 0:
            delta_error = 0
            min_error, min_train_error, HP, delta_error, tune_summary = tune_param_iter(
                min_error, params_HP_tune, HP, hp_params, delta_error, Data, tune_summary)
            if k == 'linear': delta_error = 0
        summary_min_error_test[k] = min_error
        summary_min_error_train[k] = min_train_error
        summary_min_hp[k] = copy.copy(HP)

    best_kernel = min(summary_min_error_test, key=summary_min_error_test.get)

    print("tune test error:", summary_min_error_test[best_kernel])
    print("tune train error:", summary_min_error_train[best_kernel])

    return {'min_test_error': summary_min_error_test[best_kernel],
            'min_train_error': summary_min_error_train[best_kernel],
            'chosen_HP': summary_min_hp[best_kernel],
            'summary': tune_summary
            }

def tune_param_iter(min_error, params_HP_tune, HP, hp_params, delta_error, Data, tune_summary):
    '''
            Tuning hyper-parameters, tune only one parameter at a time.
        :param min_error: current minimum test error
        :param params_HP_tune: Dict containing all the possible hyper-parameters for the model:
                1. kernel - possible kerenels (list)
                2. c - possible penalty values (list)
                3. gamma - Kernel coefficient for ‘rbf’, ‘poly’ (list)
                4. degree - possible degrees for the poly kernel (list)
        :param HP: A dict containing the current hyper-parameters for the model
        :param hp_params: specifies tuning parameters for the current kernel
        :param delta_error: current change in error
        :param Data: A dict containing the data:
                1. train_data (matrix, [train_size, n_centroids])
                2. test_data (matrix, [test_size, n_centroids])
                3. train_labels (matrix, [train_size, 1])
                4. test_labels (matrix, [test_size, 1])
        :param tune_summary: A dict containing a summary of the tuning process:
                1. The tuned parameter
                2. The fixed parameters
                3. The resulted error for each value of the tuned parameter
                4. The tuned parameter that had the minimum error
        :return: The updated values of the parameters:
                1. The min test error
                2. The min train error
                3. HP
                4. delta_error
                5. tune_summary
        '''

    for p in hp_params:  # run on every hyper-parameter in kernel
        fixed_params = copy.copy(HP)  # copy current values of  all hyper-parameter
        fixed_params.pop(p)  # remove the current tuning hyper-parameter
        chosen = HP[p]  # the current hyper-parameter that we want to tune
        test_error = {}
        train_error = {}
        for v in params_HP_tune[p]:  # check all the possible values of current tuning hyper-parameter
            temp_HP = copy.copy(HP)  # copy current values of  all hyper-parameter
            temp_HP[p] = v  # the current value of tuning hyper-parameter
            train_error[v], test_error[v] = evaluate_model(Data, temp_HP)
            if test_error[v] < min_error or (
                    test_error[v] == min_error and np.random.rand() <= 0.5):  # new best hyper-parameters
                chosen = v
                HP = temp_HP
                delta_error = max(min_error - test_error[v], delta_error)
                min_error = test_error[v]
                min_train_error = train_error[v]

        step_summary = {
            'parameter': p,
            'fixed parameters': fixed_params,
            'errors': test_error,
            'chosen': chosen
        }
        tune_summary.append(step_summary)

    return min_error, min_train_error, HP, delta_error, tune_summary


def evaluate_model(data, params):
    '''
    The function returns the error rate for this data
    :param data: A dict containing the data:
            1. train_data (matrix, [train_size, n_centroids])
            2. test_data (matrix, [test_size, n_centroids])
            3. train_labels (matrix, [train_size, 1])
            4. test_labels (matrix, [test_size, 1])
    :param params: A dict containing the current model's configuration
    :return: Error for this model (float)
    '''
    # train error
    model = Train(data['train_data'], data['train_labels'], params)
    results = Test(model, data['train_data'])
    train_error = CalculateErrorRate(results, data['train_labels'])

    # test error
    results = Test(model, data['valid_data'])
    test_error = CalculateErrorRate(results, data['valid_labels'])

    return train_error, test_error


def Train(data, labels, params):
    '''
    Train a model of multi-class SVM with the given training data and hyper-parameters
    :param data: data for training (matrix, [train_size, n_centroids])
    :param labels: Labels training (matrix, [train_size, 1])
    :param params: A dict containing the current hyper-parameters for the model
    :return: Trained model (object)
    '''
    if params['kernel'] == 'linear':  # train linear multi-class model
        model = LinearSVC(C=params['c'], max_iter=1000)
        model.fit(data, labels)

    else:  # train non-linear multi-class model
        model = {}
        # one vs rest model
        for lbl in np.unique(labels):
            # set one-vs-rest labels for class lbl
            ovr_labels = np.copy(labels)
            ovr_labels[ovr_labels != lbl] = -1
            ovr_labels[ovr_labels == lbl] = 1
            model[lbl] = SVC(C=params['c'], kernel=params['kernel'], gamma=params['gamma'], degree=params['degree'],
                             max_iter=1000)
            model[lbl].fit(data, ovr_labels)

    return model


def Test(model, data):
    '''
    Predict labels to each sample in data using model
    :param model: The model found in the training stage
    :param data: The test data (matrix, [train_size, n_centroids])
    :return: The decision functions of the predictions (matrix, [train_size, n_classes])
    '''
    predictions = None
    # checks if the model is a non-linear model
    if isinstance(model, dict):  # yes -> non-linear model
        for i in model:  # predict for each model - return the confidence for each class
            m = model[i]
            if predictions is None:
                predictions = np.asmatrix(m.decision_function(data))  # vector[test_size, 1]
            else:
                predictions = np.concatenate((predictions, np.asmatrix(m.decision_function(data))), axis=0)
        predictions = np.transpose(predictions)
    else:  # no -> linear model
        predictions = model.decision_function(data)  # matrix[test_size, n_samples]

    return predictions


def CalculateErrorRate(results, labels):
    '''
    Calculate the error rate- ratio of miss classified samples
    :param results: The predictions from the model (matrix, [test_size, n_classes])
    :param labels: The real labels (vector, [test_size, 1])
    :return: The error rate (float)
    '''
    # the prediction for each sample will be the index of the max value
    predictions = np.transpose(np.argmax(results, axis=1))  # -> matrix[test_size, 1]
    num_of_wrong = np.sum(np.not_equal(predictions, labels))

    return float(num_of_wrong / len(labels))


def ReportTuneResults(summary):
    '''
         Draw the tuning process of the parameters onto figures and save the summary file to the tuning result path
        :param summary: A summary of the evaluation
        :param params: Parameters for the report
    '''
    for iter in summary:
        Xs, Ys_test, fixed_params = getXY4plot(iter)
        if iter['parameter'] == 'kernel':
            discrete = True
        else:
            discrete = False
        plotTuning(iter['parameter'], Xs, Ys_test, fixed_params, discrete, chosen=iter['chosen'])


def getXY4plot(iteration):
    '''
        Extract the tuned parameter's X and Y values from a given iteration with dictionary format
        :param iteration: dictionary with tested values as key and validation error as value
        :return: Xs and Ys_test (array-like) represent the tested values of the tuned parameter and the validation error
                respectively, fixed_params (dictionary) the fixed parameter in the optimization process
    '''
    fixed_params = iteration['fixed parameters']
    step = iteration['errors']
    Xs = []
    Ys_test = []
    for key, value in step.items():
        Xs.append(key)
        Ys_test.append(value)

    return Xs, Ys_test, fixed_params


def plotTuning(hp, Xs, Ys_test, fixed_params, discrete, chosen):
    '''
        Draw the tuning process of the parameters onto figures
        :param hp: A summary of the evaluation
        :param Xs: the tested values of the tuned parameter
        :param Ys_test: the validation error of the tuned parameter
        :param fixed_params: the fixed parameter in the optimization process
        :param discrete: boolean indicate if the parameter is categorically or not
        :param chosen: the chosen value of the tuned parameter
    '''
    plt.figure()
    if discrete:
        plt.plot(Xs, Ys_test, 'ro', label='Y Test', color='b')
    else:
        plt.plot(Xs, Ys_test, label='Y Test', color='b')
    plt.axvline(x=chosen, label='chosen value = {}'.format(chosen), c='r')
    plt.xlabel('Value')
    plt.ylabel('Error Rate')
    plt.title('{} Tuning\nfixed parameter: {}'.format(hp, fixed_params))
    plt.legend()
    plt.show()


def CalculateConfusionMatrix(results, labels, labels_names):
    '''
    Calculate the confusion matrix
    :param results: The predictions from the model (matrix, [test_size, n_classes])
    :param labels: The real labels (vector, [test_size, 1])
    :return: Confusion matrix (matrix NxN, N = number of classes)
    '''
    predictions = np.transpose(np.argmax(results, axis=1)).reshape(len(results), 1)  # -> matrix[test_size, 1]
    confusionMatrix = confusion_matrix(labels.reshape(len(labels), 1), predictions)
    return {'CM': confusionMatrix, 'original_labels': labels_names}


def CalculateLargestError(results, labels, label_names):
    '''
    Find the biggest two error of each class, and return their indices
    :param results: The decision matrix of the classifier (matrix [test_size, n_classes]
    :param labels: The real labels (vector [test_size])
    :return: An errors dict containing:
            1. A dict for every class with the label-number as key, containing:
                1.1 The class name (string)
                1.2 The values of the largest errors. if there are no errors, it will contain an empty array (array)
                1.3 The indices of the largest errors. if there are no errors, it will contain an empty array (array)
    '''
    largest_errors_set = {}

    # calculate error for each sample
    errors = results.max(axis=1) - results[np.arange(len(labels)), labels]
    # run on each class
    for l in np.unique(labels):
        l_appearance_counter = np.cumsum(labels == l)  # the index of the last error image of label l
        rel_errors = errors[labels == l]  # take only the errors belong to lable l
        sorted_rel_errors = np.sort(rel_errors)  # sort errors from lowest to highest
        max_errors = sorted_rel_errors[-2:]  # take the two highest error samples
        max_errors = np.delete(max_errors, np.argwhere(max_errors <= 0))  # only take positive errors
        largest_errors_indices = np.argsort(rel_errors)[-len(max_errors):] + np.ones(len(max_errors)).astype(
            int) if len(max_errors) > 0 else []  # return the index in a sub-set that contains only this class
        real_indices = np.searchsorted(l_appearance_counter,
                                       largest_errors_indices)  # transform the indices to the indices of those samples in the full set

        largest_errors_set[l] = {'class name': label_names[l], 'errors': max_errors, 'indices': real_indices}

    return largest_errors_set


def EvaluateAll(results, labels, labels_names):
    '''
    Compute the results statistics and return them as fields of Summary
    :param results: The decision treas from the model (matrix, [test_size, n_classes])
    :param labels: The real labels (vector, [test_size, 1])
    :param encoder: Label encoder to translate the predicted classes from numbers to classes names
            if encoder is None, the predictions will be with the numbers instead of class name
    :param params: An array containing the statistics to calculate. may contain:
            1. 'error rate'
            2. 'confusion matrix'
            3. 'largest error'
    :return: summary
    '''
    summary = {}
    summary['error rate'] = CalculateErrorRate(results, labels)
    summary['confusion matrix'] = CalculateConfusionMatrix(results, labels, labels_names)
    summary['largest error'] = CalculateLargestError(results, labels, labels_names)

    return summary


def ReportTestResults(summary, test_results, test_data):
    '''
        Draw the confusion matrix and print its analysis for the test results and save the results to the results path
        :param summary: A summary of the evaluation
        :param params: Parameters for the report
        :param test_results: dictionary that include the the true label and the predicted label
    '''
    # print the error rate
    print('Error rate = {}'.format(summary['error rate']))
    print('-----------------------------------')
    # confusion matrix
    y_true = test_results['Labels']
    y_pred = test_results['Results']
    predictions = np.transpose(np.argmax(y_pred, axis=1))  # -> matrix[test_size, 1]
    predictions = predictions.flatten()

    # plot the confusion matrix
    plt.figure(figsize=(len(np.unique(test_results['Labels'])), len(np.unique(test_results['Labels']))))
    plt.title('Confusion Matrix')
    ax = sn.heatmap(summary['confusion matrix']['CM'], annot=True,
                    xticklabels=summary['confusion matrix']['original_labels'],
                    yticklabels=summary['confusion matrix']['original_labels'])
    ax.set(xlabel='Predicted Class', ylabel='True Class')
    plt.show()

    print('Largest Errors:')
    print('=========================================================')

    for i in summary['largest error']:
        num_of_largest_error_images = len(summary['largest error'][i]['errors'][:])
        if (num_of_largest_error_images > 0):
            print("there are", num_of_largest_error_images, "errors from class",
                  summary['largest error'][i]['class name'])
            for j in range(0, num_of_largest_error_images):
                print(summary['largest error'][i]['errors'][j])
                plt.imshow(test_data[summary['largest error'][i]['indices'][j]], cmap='gray')
                plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                plt.show()
        else:
            print("there were no errors for class", summary['largest error'][i]['class name'])


#########----------------Main---------------#########


params = GetDefaultParameters()
np.random.seed(249)
random.seed(123)
DandL = get_data(params['data'])
# returns train data, test data, train labels and test labels
SplitData = TrainTestSplit(DandL['data'], DandL['labels'], params['data'])
SplitData_valid = TrainValidSplit(SplitData['train_data'], SplitData['train_labels'], params['data'])

######fold 1-Tuning hyper parameters#####
if params['data']['tune'] == True:
    Train1DataHist, img_dict = Prepare(SplitData_valid['train_data'], params['prepare'], SplitData['train_labels'])
    ValidDataHist, _ = Prepare(SplitData_valid['valid_data'], params['prepare'], SplitData['train_labels'], img_dict)
    Summary = TuneHyperParameters(Train1DataHist, SplitData_valid['train_labels'],
                                  ValidDataHist, SplitData_valid['valid_labels'], params['HP_tune'])
    print(Summary['chosen_HP'])
    ReportTuneResults(Summary['summary'])
######fold 2- results #####
else:
    TrainDataHist, img_dict = Prepare(SplitData['train_data'], params['prepare'], DandL['labels'])
    TestDataHist, _ = Prepare(SplitData['test_data'], params['prepare'], DandL['labels'], img_dict)
    Model = Train(TrainDataHist, SplitData['train_labels'], params['final_par'])
    Results = Test(Model, TestDataHist)
    Summary = EvaluateAll(Results, SplitData['test_labels'], DandL['label_names'])
    ReportTestResults(Summary, {'Results': Results, 'Labels': SplitData['test_labels']}, SplitData['test_data'])
