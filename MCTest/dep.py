import pandas as pd
import numpy as np
import re # regular expression
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import names
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn

import editdistance as ed
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def standardize(df):
    stand = df.copy()
    for col in stand:
        if col.lower() == 'y': continue
        
        mu = stand[col].mean()
        std = stand[col].std()
        stand[col] = (stand[col]-mu)/std
    return stand

def answer_to_dict(row, question):
    return {'A': row[f'{question}_a1'], 'B': row[f'{question}_a2'], 'C': row[f'{question}_a3'], 'D': row[f'{question}_a4']}

def load_answers(path):
    return pd.read_csv(path, sep = '\t',  header = None)

def load_stories(path):
    columns = [
        'story_id',
        'author',
        'story',
        'q1',
        'q1_a1',
        'q1_a2',
        'q1_a3',
        'q1_a4',
        'q2',
        'q2_a1',
        'q2_a2',
        'q2_a3',
        'q2_a4',
        'q3',
        'q3_a1',
        'q3_a2',
        'q3_a3',
        'q3_a4',
        'q4',
        'q4_a1',
        'q4_a2',
        'q4_a3',
        'q4_a4',
    ]

    stories = pd.read_csv(path, sep = '\t', names = columns, header=None)
    stories = stories.drop(columns = ['author'])
    for q in ['q1', 'q2', 'q3', 'q4']:
        stories[f'{q}_a'] = stories.apply(answer_to_dict, axis = 1, question = q)
        stories = stories.drop(columns=[f'{q}_a1',f'{q}_a2',f'{q}_a3',f'{q}_a4'])

    return stories.T

def lemmatize(word):
    lemma = wn.morphy(word, pos=wn.VERB)
    if lemma is not None:
        return lemma
    return word

def get_align(sentence1, sentence2):
    A = 0
    s1 = sentence1.split(' ')
    s2 = sentence2.split(' ')
    
    n = len(s1)
    m = len(s2)

    for i in range(n):
        w = s1[i]

        for j in range(m):
            if s2[j] == w:
                A += 1/((abs(j - i) + 1)**2)

    return A
def create_td_mat(story):
    sentences = re.split('[.!?] ', story)
    vocab = set(re.split('[.!?] | ', story.lower())) # different vocab for each story to prevent matrix from getting too sparse (and save memory)
    vtoi = dict(zip(vocab, range(len(vocab)))) # vocab to index
    td_mat = np.zeros(shape = (len(vocab), len(sentences))) # initialize term-document matrix

    for s in range(len(sentences)):
        words = sentences[s].lower().split(' ')
        s_index = s

        for w in words:
            w_index = vtoi[w]
            td_mat[w_index, s_index] += 1

    return td_mat, vocab

def convert_question(question, vocab):
    vtoi = dict(zip(vocab, range(len(vocab)))) # vocab to index
    q = np.zeros(shape = len(vocab))

    for w in question.lower().split(' '):
        try:
            q[vtoi[w]] += 1
        except:
            continue
        
    return q

def cosine_sim(vector1, vector2):
    len1 = np.sqrt(vector1.dot(vector1))
    len2 = np.sqrt(vector2.dot(vector2))

    if len1 == 0 or len2 == 0: # prevent divison by zero; caused by frequent terms using tf-idf or infrequent words in tc matrix
        return 0

    return vector1.dot(vector2)/(len1 * len2)

def unigram_match(bag1, bag2):
    # normalized by bag1
    matches = 0
    for gram in bag1:
        if gram in bag2: matches += 1

    return matches / len(bag1)

def lev_similarity(passage, hypothesis):
    scores = np.array([ed.eval(x, hypothesis)/max(len(x),len(hypothesis)) for x in passage.split(' ')])
    return np.mean(scores)

def jaccard_similarity(set1, set2):
    cap = set1.intersection(set2)
    cup = set1.union(set2)
    return len(cap)/len(cup)

def soft_max(vector):
    e = np.exp(vector)
    return e/np.sum(e)

def sort_by_column(matrix, col_index, ascending = False):
    mat = matrix
    if ascending:
        a = 1
    else:
        a = -1
    # uses quicksort (not stable), non issue with continuous data
    return mat[mat[:,col_index].argsort(kind = "quicksort")[::a]]

def rank_sentences(matrix, question):
    q = question
    mat = matrix
    ncols = mat.shape[1] # sentences


    for r in range(ncols):
        s = cosine_sim(q, mat[:,r]) # cosine sim of question to sentence

        if r == 0: # make array on initial pass
            sims = np.array([[r,s]])

        else:
            sims = np.append(sims, [[r,s]], axis = 0)

    sims = sort_by_column(sims, 1)

    return sims[:,0].astype(np.int32), sims[:,1]

def preprocess_text(text):

    stop_words = set(
        [
            "'s", ",", "?", "who", "what", "why", "when", "did", "where", "but", "however", "..."
        ]
    )

    num_to_word = {
        '1' : 'one',
        '2' : 'two',
        '3' : 'three',
        '4' : 'four',
        '5' : 'five',
        '6' : 'six',
        '7' : 'seven',
        '8' : 'eight',
        '9' : 'nine',
        '0' : 'zero',
    }

    sentence = str(text).split(': ')[-1].lower() # for questions
    #stemmer = PorterStemmer()

    # tokeize
    tokens = word_tokenize(sentence)
    #tokens = [token for token in tokens if token.lower() not in questions]
    tokens = [lemmatize(token) if token not in num_to_word else num_to_word[token] for token in tokens]
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # return string
    return ' '.join(tokens)

def get_accuracy(pred_df, answer_df):

    N_questions = answer_df.shape[0] * answer_df.shape[1]
    pred_df = answer_df == pred_df
    pred_df[pred_df == False] = 0
    pred_df[pred_df == True] = 1
    accuracy = pred_df.values.sum() / N_questions

    return accuracy