'''
CS1671 N-Gram Modeling
@author: jacob
'''

import math, random

################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(c):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * c

def ngrams(c, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    # sliding window
    ng = []
    for x in range(len(text)):
        char = text[x]
        if x < c:
            context = start_pad(c - x) + text[:x] # x = 0, 0 chars, x = 1, 1 char, etc
        else:
            context = text[x-c:x]

        ng.append((context, char))
    return ng

def create_ngram_model(model_class, path, c=2, k=0):
    ''' Creates and returns a new n-gram model '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def get_model_scores(model_class, path, c = 2, k = 0):
    model = create_ngram_model(model_class, 'data/shakespeare_input.txt', c = c, k = k)
    scores = []
    with open(path, encoding='utf-8', errors = 'ignore') as file:
        for f in file:
            p = model.perplexity(f)
            if f == '\n':
                continue
            scores.append(p)
    
    return (scores,sum(scores)/len(scores))

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, c, k):
        self.ngram_max = c
        self.smooth_par = k
        self.vocab = set()
        self.wc = {}

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        unique = set(text)
        self.vocab = self.vocab | unique # | = union
        wc = self.wc
        for g in ngrams(self.ngram_max, text):
            if g[0] in wc:
                if g[1] in wc[g[0]]:
                    wc[g[0]][g[1]] += 1
                else:
                    wc[g[0]][g[1]] = 1
            else:
                wc[g[0]] = {}
                wc[g[0]][g[1]] = 1
        
        self.wc = wc

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        if context not in self.wc: # return 1/len(V) for a novel context
            return 1/len(self.vocab)
        
        k = self.smooth_par
        v = len(self.vocab)
        p = 0
        counts = self.wc[context]
        if char not in counts:
            c = 0
        else:
            c = counts[char]
            
        # p(char | context) = counts(context, char)/counts(context)
        p = (c + k) / (sum(counts.values()) + (k * v))

        return p

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        r = random.random()
        if context not in self.wc:
            x = r / (1/len(self.vocab)) # equal chance with no context
            x = math.floor(x)
            return list(self.vocab)[x]
        
        counts = self.wc[context]
        total =  sum(counts.values())
        curr = 0

        for k,v in counts.items():
            curr += (v / total)
            if r <= curr:
                return k

        return '-'

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        c = self.ngram_max
        text = ""

        for i in range(length):
            if i >= c:
                context = text[(i - c):i]
            else:
                context = start_pad(c - i) + text[:i]

            text += self.random_char(context)

        return text

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        perp = 0
        c = self.ngram_max

        for w in range(len(text)): # take the geometric mean of all chars
            if w >= c:
                context = text[(w - c):w]
            else:
                context = start_pad(c - w) + text[:w]
            char = text[w]
            prob = self.prob(context, char)
            if not prob: # catch log(0) <- undefined
                return (float('inf'))

            perp -= math.log(prob) #
        
        perp = perp / (len(text))
        return math.e**(perp)

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        self.ngram_max = c
        self.smooth_par = k
        self.weights = [1/(c+1)] * (c+1)
        self.vocab = set()
        self.wc = {}

    def set_lambdas(self, lambdas):
        if len(lambdas) != (self.ngram_max + 1):
            print("Error: len(lambdas) does not match max ngram")
        s = sum(lambdas)
        if s < 1:
            print("Error: Weights do not add up to 1.")
        elif s > 1:
           print("sum(lambdas) > 1, normalizing by sum(lambdas)...\n")
           t = [x / s for x in lambdas]
        else:
            t = lambdas
        self.weights = t
        print(f"lambdas = {t}")

    def get_vocab(self):
        return self.vocab

    def update(self, text):
        unique = set(text)
        self.vocab = self.vocab | unique # | = union
        wc = self.wc
        for c in range(self.ngram_max,-1,-1):
            for g in ngrams(c, text):
                if g[0] in wc:
                    if g[1] in wc[g[0]]:
                        wc[g[0]][g[1]] += 1
                    else:
                        wc[g[0]][g[1]] = 1
                else:
                    wc[g[0]] = {}
                    wc[g[0]][g[1]] = 1
            
        self.wc = wc

    def prob(self, context, char):
        p_interp = 0
        lambdas = self.weights

        for l in range(len(lambdas)):
            t_context = context[l:]
            prob = super().prob(t_context, char) # use parent prob 
            p_interp += lambdas[l] * prob

        return p_interp
        
def __main__():
    try:
        print("This script trains a 4-gram model with add-k smoothing where k = 1.")
        print("For demonstration, this code will generate an n-length amount of random text when trained on the provided data.\n")
        training_set = input("Path to Training Data (.txt): ")
        n = input("n = ")
        m = create_ngram_model(NgramModelWithInterpolation, training_set ,c = 3, k = 1)
        m.set_lambdas([10,5,2,1])
        print(m.random_text(int(n)))
    except:
        print("Something went wrong, check file path name.")

__main__()