import os
import subprocess
import csv
import re
import random
import numpy as np
from scipy.sparse import csr_matrix

def subset_data(linetuples, documentnames, ndocs, random = False):
    if random:
        doc_subset = np.random.choice(documentnames, size = ndocs, replace = False)
    else:
        doc_subset = documentnames[:ndocs]

    lt_subset = []
    vocab_subset = set()

    for line in linetuples:
        if line[0] in doc_subset:
            lt_subset.append(line)
            vocab_subset = vocab_subset.union(set(line[1])) # add new words to vocab

    return lt_subset,doc_subset,list(vocab_subset)

def read_in_shakespeare():
    """Reads in the Shakespeare dataset processesit into a list of tuples.
       Also reads in the vocab and play name lists from files.

    Each tuple consists of
    tuple[0]: The name of the play
    tuple[1] A line from the play as a list of tokenized words.

    Returns:
      tuples: A list of tuples in the above format.
      document_names: A list of the plays present in the corpus.
      vocab: A list of all tokens in the vocabulary.
    """

    tuples = []

    with open("will_play_text.csv") as f:
        csv_reader = csv.reader(f, delimiter=";")
        for row in csv_reader:
            play_name = row[1]
            line = row[5]
            line_tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", line).split()
            line_tokens = [token.lower() for token in line_tokens]

            tuples.append((play_name, line_tokens))

    with open("vocab.txt") as f:
        vocab = [line.strip() for line in f]

    with open("play_names.txt") as f:
        document_names = [line.strip() for line in f]

    return tuples, document_names, vocab

def create_term_document_matrix(line_tuples, document_names, vocab):
    """Returns a numpy array containing the term document matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      document_names: A list of the document names
      vocab: A list of the tokens in the vocabulary

    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

    Let m = len(vocab) and n = len(document_names).

    Returns:
      td_matrix: A mxn numpy array where the number of rows is the number of words
          and each column corresponds to a document. A_ij contains the
          frequency with which word i occurs in document j.
    """
    m = len(vocab)
    n = len(document_names)

    doc_index = dict(zip(document_names, range(len(document_names))))
    word_index = dict(zip(vocab, range(len(vocab))))
    td_matrix = np.zeros(shape = (m,n), dtype=np.int32)

    for line in line_tuples:
        di = doc_index[line[0]]
        words = line[1]
        for w in words:
            wi = word_index[w]
            td_matrix[wi, di] += 1

    return td_matrix


def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
    """Returns a numpy array containing the term context matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      vocab: A list of the tokens in the vocabulary

    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

    Let n = len(vocab).

    Returns:
      tc_matrix: A nxn numpy array where A_ij contains the frequency with which
          word j was found within context_window_size to the left or right of
          word i in any sentence in the tuples.
    """
    n = len(vocab)
    cws = context_window_size

    word_index = dict(zip(vocab, range(len(vocab))))
    tc_matrix = np.zeros(shape = (n,n), dtype = np.int32)

    for line in line_tuples:
        sentence = line[1] # don't care about documents

        for i in range(len(sentence)):
            target_index = word_index[sentence[i]] # target word (row index)
            
            L_win = sentence[max(0, i - cws):(i)] # upper is exclusive
            if i == len(sentence): # if we are at the end of the sentence, no upper window 
              #(i + 1) throws an error
                U_win = []
            else: 
                U_win = sentence[(i+1):(i + cws + 1)] # don't include the target word

            window = L_win + U_win
            for word in window: # add the word instances to the tc_matrix
                wj = word_index[word] #context index
                tc_matrix[target_index,wj] += 1

    return tc_matrix


def create_tf_idf_matrix(term_document_matrix):
    """Given the term document matrix, output a tf-idf weighted version.

    See section 6.5 in the textbook.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
      term_document_matrix: Numpy array where each column represents a document
      and each row, the frequency of a word in that document.

    Returns:
      A numpy array with the same dimension as term_document_matrix, where
      A_ij is weighted by the inverse document frequency of document h.
    """
    N = term_document_matrix.shape[1] # number of documents
    tf = np.log(term_document_matrix + 1) # log counts of term frequencies
    df = np.count_nonzero(term_document_matrix, axis = 1)
    idf = np.log(N/df)
    
    for d in range(N):
       tf[:,d] *= idf 
    
    return tf #tf-idf mat
	
def create_ppmi_matrix(term_context_matrix):
    """Given the term context matrix, output a ppmi weighted version.

    See section 6.6 in the textbook.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
      term_context_matrix: Numpy array where each cell represents whether the 
	  word in the row appears within a window of the word in the column.

    Returns:
      A numpy array with the same dimension as term_context_matrix, where
      A_ij is weighted using PPMI.
    """
    tcm = csr_matrix(term_context_matrix) # saving memory for calculations
    word_counts = tcm.sum(axis = 1).reshape(-1,1) #sum across rows
    context_counts = tcm.sum(axis = 0) #columns
    total = tcm.sum() # total matrix counts

    ppmi_mat = np.log2((tcm * total) / (word_counts * context_counts))
    ppmi_mat = np.maximum(np.nan_to_num(ppmi_mat, copy = False), 0) # replace nan and negative values with 0

    return np.asarray(ppmi_mat)


def compute_cosine_similarity(vector1, vector2):
    """Computes the cosine similarity of the two input vectors.

    Inputs:
      vector1: A nx1 numpy array
      vector2: A nx1 numpy array

    Hint: Use numpy matrix and vector operations to speed up implementation.


    Returns:
      A scalar similarity value.
    """

    len1 = np.sqrt(vector1.dot(vector1))
    len2 = np.sqrt(vector2.dot(vector2))

    if len1 == 0 or len2 == 0: # prevent divison by zero; caused by frequent terms using tf-idf or infrequent words in tc matrix
        return 0

    return vector1.dot(vector2)/(len1 * len2)

def sort_by_column(matrix, col_index, ascending = False):
    mat = matrix
    if ascending:
        a = 1
    else:
        a = -1
    # uses quicksort (not stable), non issue with continuous data
    return mat[mat[:,col_index].argsort(kind = "quicksort")[::a]]

def rank_words(target_word_index, matrix):
    """Ranks the similarity of all of the words to the target word using compute_cosine_similarity.

    Inputs:
      target_word_index: The index of the word we want to compare all others against.
      matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.

    Returns:
      A length-n list of integer word indices, ordered by decreasing similarity to the
      target word indexed by word_index
      A length-n list of similarity scores, ordered by decreasing similarity to the
      target word indexed by word_index
    """
    mat = matrix
    nrows = mat.shape[0] # number of rows  
    woi = mat[target_word_index,:] # word of interest  

    for r in range(nrows):
        if r == target_word_index: # don't compute similarity with itself
            continue
        
        s = compute_cosine_similarity(woi, mat[r,:])

        # create the array on first pass
        if r == 0:
            sims = np.array([[r,s]])

        else:
          sims = np.append(sims, [[r,s]], axis = 0)
    
    sims = sort_by_column(sims, 1)

    return sims[:,0].astype(np.int32), sims[:,1]