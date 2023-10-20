#############################################################
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################

from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    tp = 0
    fp = 0
    for p in range(len(y_pred)): # len(pred) == len(true); p = prediction index
        if y_pred[p] == 1: # is it PREDICTED positive
            if y_true[p] == 1: # correct prediction
                tp += 1
            else:
                fp += 1

    return (tp/(tp+fp))
    
## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    tp = 0
    fn = 0

    for p in range(len(y_true)):
        if y_true[p] == 1: # check all positive samples
            if y_pred[p] == 1: # correct prediction
                tp += 1
            else:
                fn += 1
    return (tp/(tp+fn)) # recall == sensitivity

## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    p = get_precision(y_pred, y_true)
    r = get_recall(y_pred, y_true)

    fscore = 2 * ((p * r)/(p + r))

    return fscore, (p,r)

def test_predictions(y_pred, y_true):

    f,_ = get_fscore(y_pred, y_true)
    p = get_precision(y_pred, y_true)
    r = get_recall(y_pred, y_true)

    print(f"Precision = {p}")
    print(f"Recall = {r}")
    print(f"F-Score = {f}")

    return (p,r,f) # returns a tuple of precision, recall, and fscore

### for error analysis
def get_bad_preds(words, y_pred, y_true):
    df = pd.DataFrame({'word' : words, 'y_pred' : y_pred, 'y_true' : y_true})
    wrong = df[df.y_pred != df.y_true]
    return wrong['word']

### for plotting pr curves
def plot_pr_curve(dict, title):
    df = pd.DataFrame(dict).set_index('t')
    plt.title("Precision-Recall Curve for {}".format(title))
    plt.plot(df['r'], df['p'], 'blue', marker = 'o', linewidth = 1.2)
    plt.plot([1,0], [0,1], 'r--', linewidth = 1, alpha = 0.5)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file):
    words = []
    labels = []   
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels

### 2.1: A very simple baseline

## Makes feature matrix for all complex
def all_complex_feature(words):
    return [words,[1] * len(words)]

## Labels every word complex
def all_complex(data_file):
    data = load_file(data_file)
    words = data[0] # data is a tuple where data_file[0] are the words and [1] are the labels
    true_labels = data[1]
    preds = all_complex_feature(words)[1]
    p,r,f = test_predictions(preds, true_labels)

    performance = [p, r, f]
    return performance

### 2.2: Word length thresholding

## Makes feature matrix for word_length_threshold
def length_threshold_feature(words, threshold):
    preds = []
    for w in words:
        if len(w) >= threshold:
            preds.append(1)
        else:
            preds.append(0)
    return [words,preds]

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    thresh = 0
    t_data = load_file(training_file)
    t_words = t_data[0]
    t_labels = t_data[1]

    d_data = load_file(development_file)
    d_words = d_data[0]
    d_labels = d_data[1]

    best_f = 0
    pr_dict = {
        't' : [],
        'p' : [],
        'r' : []
        }
    for t in range(21): # try thresh [0,20]
        temp = length_threshold_feature(t_words, t)
        f,_pr = get_fscore(temp[1], t_labels)
        if f > best_f:
            best_f = f
            thresh = t

        # for plotting pr curve
        pr_dict['t'].append(t)
        pr_dict['p'].append(_pr[0])
        pr_dict['r'].append(_pr[1])
        

    print(f"Best Threshold Found = {thresh}")
    print('-' * 20)
    
    t_M = length_threshold_feature(t_words, thresh)
    d_M = length_threshold_feature(d_words, thresh)

    print("Training Scores:")
    tp, tr, tf = test_predictions(t_M[1], t_labels)

    print("\nDevelopment Scores:")
    dp, dr, df = test_predictions(d_M[1], d_labels)

    training_performance = [tp, tr, tf]
    development_performance = [dp, dr, df]

    plot_pr_curve(pr_dict, title = "Different Word Length Thresholds")

    return training_performance, development_performance

### 2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file): 
   counts = defaultdict(int) 
   with gzip.open(ngram_counts_file, 'rt') as f: 
       for line in f:
           token, count = line.strip().split('\t') 
           if token[0].islower(): 
               counts[token] = int(count) 
   return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set

## Make feature matrix for word_frequency_threshold
def frequency_threshold_feature(words, threshold, counts):
    preds = []
    for w in words:
        try: # catch unseen words
            freq = counts[w]
        except:
            freq = 0 

        if freq <= threshold: # if word is not frequently seen, it is complex
            preds.append(1)
        else:
            preds.append(0)

    return [words,preds]

def word_frequency_threshold(training_file, development_file, counts):
    thresh = 0
    t_data = load_file(training_file)
    t_words = t_data[0]
    t_labels = t_data[1]

    d_data = load_file(development_file)
    d_words = d_data[0]
    d_labels = d_data[1]

    # bounds for threshold optimization
    l_B = min(counts.values())
    u_B = max(counts.values()) // 2 # divison by 2 is optional (a thresh = upper bound would be equal to first baseline)

    print(f"Trying Thresholds between {[l_B, u_B]}")

    best_f = 0
    pr_dict = {
        't' : [],
        'p' : [],
        'r' : []
    }
    for t in np.linspace(l_B, u_B, 50000): # try 10000 thresholds
        temp = frequency_threshold_feature(t_words, threshold = t, counts = counts)
        f,_pr = get_fscore(temp[1], t_labels) 
        if f > best_f:
            best_f = f
            thresh = t
        #else: break # quick stopping, assumes local max = global max

        pr_dict['t'].append(t)
        pr_dict['p'].append(_pr[0])
        pr_dict['r'].append(_pr[1])

    print(f"Best Threshold Found = {thresh}")
    print('-' * 20)
    t_M = frequency_threshold_feature(t_words, threshold = thresh, counts = counts)
    d_M = frequency_threshold_feature(d_words, threshold = thresh, counts = counts)

    print("Training Scores:")
    tp, tr, tf = test_predictions(t_M[1], t_labels)

    print("\nDevelopment Scores:")
    dp, dr, df = test_predictions(d_M[1], d_labels)

    training_performance = [tp, tr, tf]
    development_performance = [dp, dr, df]

    plot_pr_curve(pr_dict, title = "Different Word Frequency Thresholds")

    return training_performance, development_performance

### 2.4: Naive Bayes

def get_clf_features(DataFrame, counts, standardize = True, std_args = {'mu' : -1, 'sd' : 0}):
    df = DataFrame
    df['length'] = [len(w) for w in df['words']] # length of each word
    df['freq'] = [counts[w] for w in df['words']] # frequency according to counts
    
    mu_sd = 0
    if standardize:
        try:
            if std_args['mu'] == -1:
                mu = df[['length','freq']].mean()
                sd = df[['length', 'freq']].std()
        except:
            mu = std_args['mu']
            sd = std_args['sd']

        df[['length', 'freq']] = (df[['length', 'freq']] - mu) / sd

        mu_sd = (mu,sd)

    return df, mu_sd
        
## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    
    ### load data
    t_data = load_file(training_file)
    t_words = t_data[0]
    t_labels = t_data[1]

    d_data = load_file(development_file)
    d_words = d_data[0]
    d_labels = d_data[1]

    ### make dataframes and standardize (if standardize = true)
    train_df = pd.DataFrame({'words' : t_words})
    train_df, t_ms = get_clf_features(train_df, counts)

    dev_df = pd.DataFrame({'words' : d_words})
    dev_df, d_ms = get_clf_features(dev_df, counts, std_args={'mu' : t_ms[0], 'sd' : t_ms[1]})

    ### fit classifier
    x_M = train_df[['length','freq']].to_numpy() #[word length, word frequency] features
    y_M = np.array(t_labels) # 1D array of labels

    smoother = 0
    best_f = 0
    pr_dict = {
        't' : [],
        'p' : [],
        'r' : []
        }
    for e in np.logspace(1,-9,100):
        clf = GaussianNB(var_smoothing = e)
        clf.fit(x_M, y_M)
        t_preds = clf.predict(x_M)
        f,_pr = get_fscore(t_preds, t_labels)

        if f > best_f:
            best_f = f
            smoother = e

        # pr curve
        pr_dict['t'].append(e)
        pr_dict['p'].append(_pr[0])
        pr_dict['r'].append(_pr[1])

    print(f"Best Smoother Found = {smoother}")
    print('-' * 20)

    clf = GaussianNB(var_smoothing = smoother)
    clf.fit(x_M, y_M)

    ### predictions
    test_M = dev_df[['length','freq']].to_numpy()
    t_preds = clf.predict(x_M)
    d_preds = clf.predict(test_M)

    if 0: # report matrices if 1
        print(f"{x_M = }")
        print(f"{y_M = }")
        print(f"{test_M = }")

    print("\nTraining Scores:")
    tp, tr, tf = test_predictions(t_preds, t_labels)

    print("\nDevelopment Scores:")
    dp, dr, df = test_predictions(d_preds, d_labels)

    wrong_train_preds = get_bad_preds(words = t_words, y_pred = t_preds, y_true = t_labels)
    wrong_dev_preds = get_bad_preds(words = d_words, y_pred = d_preds, y_true = d_labels)

    training_performance = (tp, tr, tf)
    development_performance = (dp, dr, df)

    plot_pr_curve(pr_dict, "Naive Bayes across Different Add-Epsilon Smoothing")

    return development_performance, {"train" : wrong_train_preds, "dev" : wrong_dev_preds}, clf

### 2.5: Logistic Regression

def logistic_regression(training_file, development_file, counts):
    ### load data
    t_data = load_file(training_file)
    t_words = t_data[0]
    t_labels = t_data[1]

    d_data = load_file(development_file)
    d_words = d_data[0]
    d_labels = d_data[1]

    ### make dataframes
    train_df = pd.DataFrame({'words' : t_words})
    train_df, t_ms = get_clf_features(train_df, counts)

    dev_df = pd.DataFrame({'words' : d_words})
    dev_df, d_ms = get_clf_features(dev_df, counts, std_args={'mu' : t_ms[0], 'sd' : t_ms[1]})

    ### fit classifier
    # penalty = l2 = ridge regression w/ default strength
    x_M = train_df[['length','freq']].to_numpy() #[word length, word frequency] features
    y_M = np.array(t_labels) # 1D array of labels

    reg_strength = 0
    best_f = 0
    pr_dict = {
        't' : [],
        'p' : [],
        'r' : []
        }
    for c in np.logspace(4,-4, 100):
        clf = LogisticRegression(
            penalty = 'l2',
            solver = 'lbfgs', 
            max_iter = 100, 
            n_jobs = 2,
            C = c
            )
        clf.fit(x_M, y_M)
        t_preds = clf.predict(x_M)
        f,_pr = get_fscore(t_preds, t_labels)

        if f > best_f:
            best_f = f
            reg_strength = c

        # pr curve
        pr_dict['t'].append(c)
        pr_dict['p'].append(_pr[0])
        pr_dict['r'].append(_pr[1])

    print(f"Best C (regularization strength) Found = {reg_strength}")
    print('-' * 20)

    clf = LogisticRegression(
        penalty = 'l2',
        solver = 'lbfgs', 
        max_iter = 100, 
        n_jobs = 2,
        C = reg_strength
        )
    clf.fit(x_M, y_M) # refit on full matrix
    print(f"coefficients for (length, freq) = {clf.coef_}")

    ### predictions
    test_M = dev_df[['length','freq']].to_numpy()
    t_preds = clf.predict(x_M)
    d_preds = clf.predict(test_M)

    if 0: # report matrices if 1
        print(f"{x_M = }")
        print(f"{y_M = }")
        print(f"{test_M = }")

    print("\nTraining Scores:")
    tp, tr, tf = test_predictions(t_preds, t_labels)

    print("\nDevelopment Scores:")
    dp, dr, df = test_predictions(d_preds, d_labels)

    wrong_train_preds = get_bad_preds(words = t_words, y_pred = t_preds, y_true = t_labels)
    wrong_dev_preds = get_bad_preds(words = d_words, y_pred = d_preds, y_true = d_labels)

    plot_pr_curve(pr_dict, "Logistic Regression across Different Regularization Strengths")

    development_performance = (dp, dr, df)
    return development_performance, {"train" : wrong_train_preds, "dev" : wrong_dev_preds}, clf

if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    train_data = load_file(training_file)
    
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)
