import pandas as pd
import numpy as np
import re # regular expression

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from dep import *

class THM_classifier():
    def __init__(self, story_data, weights = [1,1,1,1,1]):
        self.data = story_data
        self.lambdas = weights

    def predict(self):
        story_answers = {}
        story_data = self.data

        # for every story
        for story_id in story_data:

            # extract column
            story = story_data[story_id]

            # convert text into a term-document matrix and vocabulary set
            # each 'document' is a sentence
            passage = preprocess_text(story['story'])
            td_mat, vocab = create_td_mat(passage)

            # len(ans) = 4 where i-th index corresponds to i+1 question
            ans = []
            for question in ['q1','q2','q3','q4']:

                # weights
                lambdas = np.array(self.lambdas, dtype = np.float64)
                lambdas /= lambdas.sum()
                
                query = story[question]
                q = preprocess_text(query)

                # rank index and scores
                r_index, _ = rank_sentences(td_mat, convert_question(q, vocab))

                # create a combination of the window +/- 2 around the top sentence
                top_index = r_index[0]
                comb = td_mat[:,[
                    max(0, top_index-2),
                    max(0, top_index-1),
                    top_index,
                    min(td_mat.shape[1]-1, top_index + 1),
                    min(td_mat.shape[1]-1, top_index + 2)
                ]]
                comb = (comb * lambdas).sum(axis = 1)

                candidate_answers = story[f'{question}_a']
                best = (0,0)
                for num,answer in candidate_answers.items():
                    sim = cosine_sim(comb, convert_question(preprocess_text(answer), vocab))
                    if sim >= best[1]:
                        best = (num, sim)

                # use the answer with the highest similarity to the combined window
                ans.append(best[0])

            story_answers[story_id] = ans

        return pd.DataFrame(story_answers).T