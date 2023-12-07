import pandas as pd
import numpy as np
import re # regular expression

from sklearn.svm import LinearSVC # svm with linear kernel
from sklearn.linear_model import SGDClassifier # stochastic gradient descent linmod
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from dep import *

class RTE_classifier():
    def __init__(self, training_answers, weights = [1,1,1,1,1]):
        self.lambdas = weights
        self.answers = training_answers
        self.names = set([name.lower() for name in (names.words('male.txt') + names.words('female.txt'))])

    def create_features(self, text_data):
        cos = []
        jac = []
        #tthmatch = []
        #httmatch = []
        lev = []
        named_ent = []
        qtype = []
        align = []
        neg = []

        weights = np.array(self.lambdas, dtype = np.float64)
        weights /= weights.sum()

        train_answers = self.answers
        story_data = text_data
        
        for story_id in story_data:
            story = story_data[story_id]
            passage = preprocess_text(story['story']) # text
            td_mat, vocab = create_td_mat(passage)
            vtoi = dict(zip(vocab, range(len(vocab))))
            # convert each question 
            for question in ['q1','q2','q3','q4']:
                hypotheses = create_hypothesis(story[question],story[f'{question}_a'])

                # check each hypothesis against the text
                for num, hyp in hypotheses.items():
                    hyp_vocab = set(hyp.split(' '))
                    r_index, _ = rank_sentences(td_mat, convert_question(hyp, vocab))

                    # Cosine Sim Convolution
                    top_index = r_index[0]
                    comb = td_mat[:,[
                        max(0, top_index-2),
                        max(0, top_index-1),
                        top_index,
                        min(td_mat.shape[1]-1, top_index + 1),
                        min(td_mat.shape[1]-1, top_index + 2)
                    ]]
                    comb = (comb * weights).sum(axis = 1)
                    cos.append(cosine_sim(comb, convert_question(hyp, vocab)))

                    # Sets
                    jac.append(jaccard_similarity(vocab, hyp_vocab))
                    #tthmatch.append(unigram_match(vocab, hyp_vocab))
                    #httmatch.append(unigram_match(hyp_vocab, vocab))

                    # Edit
                    lev.append(lev_similarity(passage, hyp))

                    ne_count = 0
                    for word in hyp_vocab:
                        if (word in self.names) and (word in vtoi):
                            ne_count += td_mat.sum(axis = 1)[vtoi[word]]

                    named_ent.append(ne_count)

                    A = []
                    for sentence in passage.split('.'):
                        A.append(get_align(sentence, hyp))
                        
                    align.append(np.mean(A))
            
                    if 'not' in hyp:
                        neg.append(1)
                    else:
                        neg.append(0)

                    if train_answers.T[story_id][int(question[-1])-1] == num:
                        qtype.append(1)
                    else:
                        qtype.append(0)


        # creating dataframe
        D_mat = pd.DataFrame(
            {
                'x1' : cos,
                'x2' : jac,
                'x3' : lev,
                'x4' : neg,
                'x5' : named_ent,
                'x6' : align,
                'y' : qtype
            }
        )

        return standardize(D_mat)
    
    def train(self, dataframe):
        # fits a support vector machine with a linear kernel on the feature set of similarity metrics
        X_mat = dataframe.drop(columns = ['y']).values
        Y_mat = dataframe['y'].values

        param_grid = [
            {
                'loss' : ['hinge', 'squared_hinge'],
                'penalty' : ['elasticnet'],
                'alpha' : np.logspace(-9,0,num =10), # reg param
                'l1_ratio' : np.logspace(-9,0,num=10),
                'max_iter' : [10000],
                'learning_rate' : ['adaptive'],
                'eta0' : np.logspace(-3,0,num = 10)
            }
        ]

        lsvc = GridSearchCV(
            SGDClassifier(),
            param_grid,
            cv = 10,
            scoring = 'accuracy',
            verbose = 0
        ).fit(
            X_mat,
            Y_mat
        )

        print(lsvc.best_params_)
        self.model = lsvc

    def predict(self, X_matrix):
        lsvc = self.model

        scores = pd.DataFrame( 
            lsvc.decision_function(X_matrix.values).reshape(-4,4),
            columns = ['A', 'B', 'C', 'D']
        )
        scores = scores.apply(soft_max, axis = 1)
        scores['best'] = scores.idxmax(axis = 1)
        return scores