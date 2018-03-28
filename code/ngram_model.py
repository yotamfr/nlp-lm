#!/usr/local/bin/python

from data_utils import utils as du
import numpy as np
import pandas as pd
import csv

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                     index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)


def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """
    trigram_counts = dict()
    bigram_counts = dict()
    unigram_counts = dict()
    token_count = 0
    ### YOUR CODE HERE
    # raise NotImplementedError
    for sentence in dataset:
        for w in range(len(sentence)):
            token_count += 1

            unigram = sentence[w]
            if not unigram in unigram_counts:
                unigram_counts[unigram] = 1
            else:
                unigram_counts[unigram] += 1

            if w >= 1:
                bigram = (sentence[w-1], sentence[w])
                if not bigram in bigram_counts:
                    bigram_counts[bigram] = 1
                else:
                    bigram_counts[bigram] += 1

            if w >= 2:
                trigram = (sentence[w-2], sentence[w-1], sentence[w])
                if not trigram in trigram_counts:
                    trigram_counts[trigram] = 1
                else:
                    trigram_counts[trigram] += 1

    ### END YOUR CODE
    return trigram_counts, bigram_counts, unigram_counts, token_count


def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    def log_prob(sentence, v_lambda):
        total = np.float128(0)
        for w in range(len(sentence)):

            ptri, pbi, puni = np.float128(0), np.float128(0), np.float128(0)
            unigram = sentence[w]
            puni = np.float128(unigram_counts[unigram])/train_token_count

            if w >= 1:
                bigram = (sentence[w - 1], sentence[w])
                unigram = sentence[w-1]
                pbi = np.float128(0) if bigram not in bigram_counts \
                    else np.float128(bigram_counts[bigram])/unigram_counts[unigram]

            if w >= 2:
                trigram = (sentence[w - 2], sentence[w - 1], sentence[w])
                bigram = (sentence[w - 2], sentence[w - 1])
                ptri = np.float128(0) if trigram not in trigram_counts \
                    else np.float128(trigram_counts[trigram])/bigram_counts[bigram]

                total += np.log2(np.dot([ptri, pbi, puni], v_lambda))

        return total

    perplexity = 0
    ### YOUR CODE HERE
    # raise NotImplementedError
    v_lambda = [lambda1, lambda2, 1 - lambda1 - lambda2]
    l = np.sum([log_prob(sentence, v_lambda) for sentence in eval_dataset])
    l /= np.sum(np.sum([len(sentence) for sentence in eval_dataset]))
    perplexity = np.exp2(-l)

    ### END YOUR CODE
    return perplexity


def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    #Some examples of functions usage
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    print "#perplexity: " + str(perplexity)
    ### YOUR CODE HERE
    opt, arg_opt = perplexity, [0.5, 0.4]
    grid = [[x, y] for x in np.linspace(0, 1, 11) for y in np.linspace(0, 1-x, (1-x)/.1 + 1)]
    for config in grid:
        l1, l2 = config[0], config[1]
        perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, l1, l2)
        print "#lambda1={0}, lambda2={1}, lambda3={2} perplexity={3}".format(l1, l2, 1-l1-l2, perplexity)
        if perplexity < opt: opt, arg_opt = perplexity, [l1, l2]
    print "==== BEST RESULT ===="
    l1, l2 = arg_opt[0], arg_opt[1]
    print "#lambda1={0}, lambda2={1}, lambda3={2} perplexity={3}".format(l1, l2, 1-l1-l2, opt)
    ### END YOUR CODE


if __name__ == "__main__":
    test_ngram()
