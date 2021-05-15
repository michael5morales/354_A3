# Michael Morales 111320457
# CSE354, Spring 2021

import sys
import re  # regular expressions
import numpy as np
import torch
import torch.nn as nn  # pytorch
import pandas as pd


# sys.stdout = open('a3_morales_111320457_OUTPUT.txt', 'w')


def read(filename):
    file = open(filename)
    return_data = {}  # contains
    count_dict = {}
    context_strings = []

    for line in file:
        items = re.split(r'\t+', line)
        context = items[2]

        context = context.lower()

        context = f'<s> {context} </s>'

        head_match = re.compile(r'<head>([^<]+)</head>')  # matches contents of head
        tokens = context.split()  # get the tokens
        headindex = -1  # will be set to the index of the target word
        for i in range(len(tokens)):
            m = head_match.match(tokens[i])
            if m:  # a match: we are at the target token
                tokens[i] = m.groups()[0]
                headindex = i

            if tokens[i] != '</s>':
                context_items = re.split(r'\/', tokens[i])
                tokens[i] = context_items[0]

            if context_items[0] in count_dict.keys():
                count_dict[context_items[0]] = count_dict[(context_items[0])] + 1
            else:
                count_dict[context_items[0]] = 1

        context = ' '.join(tokens)  # turn context back into string

        # context_final = f'<s> {context} </s>'

        context_strings.append(context)

    temp = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

    # vocabulary = []
    vocab_count = {}

    for i in range(5000):
        # vocabulary.append((temp[i])[0])
        vocab_count[(temp[i])[0]] = (temp[i])[1]

    vocab_count["<OOV>"] = 0

    # return_data["vocab"] = vocabulary
    return_data["context"] = context_strings
    return_data["count"] = vocab_count

    file.close()
    return return_data


def bigram_count(context, vocabulary):
    copy_dict = {'<OOV>': 0}
    bigrams = vocabulary.copy()

    for key in vocabulary.keys():
        bigrams[key] = copy_dict.copy()

    for line in context:
        # line = line[3:(len(line) - 4)]  # REMOVES <s> <\s> FOR BIGRAM COUNTING

        tokens = line.split()

        for x in range(len(tokens) - 1):
            if tokens[x] in vocabulary.keys():
                left_temp = tokens[x]
            else:
                left_temp = "<OOV>"

            if tokens[x + 1] in vocabulary.keys():
                right_temp = tokens[x + 1]
            else:
                right_temp = "<OOV>"

            try:
                bigrams[left_temp][right_temp] += 1
            except:
                temp_dict = {right_temp: 1}
                bigrams[left_temp].update(temp_dict)

    return bigrams


def trigram_count(context, vocabulary):
    copy_dict = {'<OOV>': 0}
    trigrams = {}

    for k, v in vocabulary.items():
        for i in v.keys():
            trigrams[(k, i)] = copy_dict.copy()

    for line in context:
        # line = line[3:(len(line) - 4)]  # REMOVES <s> <\s> FOR BIGRAM COUNTING

        tokens = line.split()

        for x in range(len(tokens) - 2):
            temp_tuple = (tokens[x], tokens[x+1])

            if temp_tuple in trigrams.keys():
                left_temp = temp_tuple
            else:
                left_temp = ("<OOV>", "<OOV>")

            if tokens[x + 2] in vocabulary.keys():
                right_temp = tokens[x + 2]
            else:
                right_temp = "<OOV>"

            try:
                trigrams[left_temp][right_temp] += 1
            except:
                temp_dict = {right_temp: 1}
                trigrams[left_temp].update(temp_dict)

    return trigrams


def get_trigram(trigrams, left, right):
    try:
        return left[0], left[1], right, trigrams[left][right]
    except:
        return left[0], left[1], right, 0


def bigram_probs(bigram_dict, unigram_copy, prev_word):
    bi_probs = {}

    temp_bi_dict = bigram_dict[prev_word]

    for i, j in temp_bi_dict.items():
        bi_probs[(prev_word, i)] = (j + 1) / (unigram_copy[prev_word] + len(unigram_copy))

    return bi_probs


def get_bi_prob(bigram_prob_copy, prev_word1, x):
    try:
       return bigram_prob_copy[(prev_word1, x)]
    except:
        return 0


def trigram_probs(trigram_dict, bigram_dict, bigram_prob_copy, prev_word1, prev_word2):
    tri_probs = {}
    for x, y in trigram_dict[(prev_word2, prev_word1)].items():
        temp = (y + 1) / (bigram_dict[prev_word2][prev_word1] + len(bigram_dict))
        to_add = get_bi_prob(bigram_prob_copy, prev_word1, x)
        tri_probs[(prev_word2, prev_word1, x)] = float((temp + to_add)/2)

    return tri_probs


def lang_model_probs(bigram_copy, unigram_copy, word_minus1, word_minus2=None, trigram_copy=None):
    if word_minus2 is None:  # Single previous word
        return bigram_probs(bigram_copy, unigram_copy, word_minus1)
    else:  # Two previous words
        temp_bi_prob = bigram_probs(bigram_copy, unigram_copy, word_minus1)
        return trigram_probs(trigram_copy, bigram_copy, temp_bi_prob, word_minus1, word_minus2)


def get_trigram_probs(trigram_prob, words1):
    try:
        return words1, trigram_prob[words1]
    except:
        return words1, 'INVALID W_i'


def generate_language(words2, bi_probs, bigrams, tri_probs, trigrams, unig):
    if len(words2) == 1:
        p = bigram_probs(bigrams, unig, words2[0])

        temp = list(p.items())# = sorted(p.items(), key=lambda x: x[1], reverse=True)

        keys = []
        vals = []
        sum = 0

        for i in range(32):
            keyT = ((temp[i])[0])[1]
            valT = (temp[i])[1]
            keys.append(keyT)
            vals.append(valT)
            sum += valT

        for i in range(32):
            vals[i] = (vals[i] / sum)

        temp = np.random.choice(a=keys, p=vals)
        words2.append(temp)

        return words2
    else:
        end_idx = len(words2) - 1

        word_minus1 = words2[end_idx]
        word_minus2 = words2[end_idx - 1]

        p = bigram_probs(bigrams, unig, word_minus1)

        t = trigram_probs(trigrams, bigrams, p, word_minus1, word_minus2)

        keys = []
        vals = []
        sum = 0

        if len(t) > 32:
            # temp_sort = sorted(t.items(), key=lambda x: x[1], reverse=True)
            temp_sort = list(t.items())

            for i in range(32):
                keyT = ((temp_sort[i])[0])[2]
                valT = (temp_sort[i])[1]
                keys.append(keyT)
                vals.append(valT)
                sum += valT
        else:
            for x, y in t.items():
                vals.append(y)
                sum += y
                keys.append(x[len(x) - 1])

        for i in range(len(vals)):
            vals[i] = (vals[i] / sum)

        temp = np.random.choice(a=keys, p=vals)
        words2.append(temp)
        return words2


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # READ TRAINING DATA
    training_data = read(sys.argv[1])
    # vocab, context_data, unigram = training_data.values()

    context_data, unigram = training_data.values()

    print("CHECKPOINT 2.2 - counts")
    print("1-grams:")
    print("\t(\'language\',)", unigram["language"])
    print("\t(\'the\',)", unigram["the"])
    print("\t(\'formal\',)", unigram["formal"])

    # BIGRAM COUNT
    bigram = bigram_count(context_data, unigram)

    print("2-grams:")
    print("\t(\'the\', \'language\')", bigram["the"]["language"])
    print("\t(\'<OOV>\', \'language\')", bigram["<OOV>"]["language"])
    print("\t(\'to\', \'process\')", bigram["to"]["process"])

    # TRIGRAM COUNT
    trigram = trigram_count(context_data, bigram)

    print("3-grams:")
    print('\t', get_trigram(trigram, ("specific", "formal"), "languages"))
    print('\t', get_trigram(trigram, ("to", "process"), "<OOV>"))
    print('\t', get_trigram(trigram, ("specific", "formal"), "event"))

    print("\nCHECKPOINT 2.3 - Probs with add-one")
    # BI GRAM PROBS
    print("2-grams:")

    probs = {}

    for i, j in bigram.items():
        probs.update(lang_model_probs(bigram, unigram, i))

    print('\t', ('the', 'language'), probs[("the", "language")])
    print('\t', ('<OOV>', 'language'), probs[("<OOV>", "language")])
    print('\t', ('to', 'process'), probs[("to", "process")])

    # TRI GRAM PROBS
    print("3-grams:")

    # probs = lang_model_probs(bigram, unigram, "formal", "specific", trigram)

    tri_prob = {}
    for i, j in trigram.items():
        tri_prob.update(lang_model_probs(bigram, unigram, i[1], i[0], trigram))

    print('\t', get_trigram_probs(tri_prob, ('specific', 'formal', 'languages')))
    print('\t', get_trigram_probs(tri_prob, ('to', 'process', '<OOV>')))
    print('\t', get_trigram_probs(tri_prob, ('specific', 'formal', 'event')))

    print("\nFINAL CHECKPOINT - Generated Language")
    print("PROMPT: <s>")

    word_list = ['<s>']

    words = generate_language(word_list.copy(), probs, bigram, tri_prob, trigram, unigram)
    while words[len(words) - 1] != '</s>' and len(words) < 32:
        words = generate_language(words, probs, bigram, tri_prob, trigram, unigram)
    print('\t', words)

    words = generate_language(word_list.copy(), probs, bigram, tri_prob, trigram, unigram)
    while words[len(words) - 1] != '</s>' and len(words) < 32:
        words = generate_language(words, probs, bigram, tri_prob, trigram, unigram)
    print('\t', words)

    words = generate_language(word_list.copy(), probs, bigram, tri_prob, trigram, unigram)
    while words[len(words) - 1] != '</s>' and len(words) < 32:
        words = generate_language(words, probs, bigram, tri_prob, trigram, unigram)
    print('\t', words)

    print("\nPROMPT: <s> language is")
    word_list = ['<s>', 'language', 'is']

    words = generate_language(word_list.copy(), probs, bigram, tri_prob, trigram, unigram)
    while words[len(words) - 1] != '</s>' and len(words) < 32:
        words = generate_language(words, probs, bigram, tri_prob, trigram, unigram)
    print('\t', words)

    words = generate_language(word_list.copy(), probs, bigram, tri_prob, trigram, unigram)
    while words[len(words) - 1] != '</s>' and len(words) < 32:
        words = generate_language(words, probs, bigram, tri_prob, trigram, unigram)
    print('\t', words)

    words = generate_language(word_list.copy(), probs, bigram, tri_prob, trigram, unigram)
    while words[len(words) - 1] != '</s>' and len(words) < 32:
        words = generate_language(words, probs, bigram, tri_prob, trigram, unigram)
    print('\t', words)

    print("\nPROMPT: <s> machines")
    word_list = ['<s>', 'machines']

    words = generate_language(word_list.copy(), probs, bigram, tri_prob, trigram, unigram)
    while words[len(words) - 1] != '</s>' and len(words) < 32:
        words = generate_language(words, probs, bigram, tri_prob, trigram, unigram)
    print('\t', words)

    words = generate_language(word_list.copy(), probs, bigram, tri_prob, trigram, unigram)
    while words[len(words) - 1] != '</s>' and len(words) < 32:
        words = generate_language(words, probs, bigram, tri_prob, trigram, unigram)
    print('\t', words)

    words = generate_language(word_list.copy(), probs, bigram, tri_prob, trigram, unigram)
    while words[len(words) - 1] != '</s>' and len(words) < 32:
        words = generate_language(words, probs, bigram, tri_prob, trigram, unigram)
    print('\t', words)

    print("\nPROMPT: <s> they want to process")
    word_list = ['<s>', 'they', 'want', 'to', 'process']

    words = generate_language(word_list.copy(), probs, bigram, tri_prob, trigram, unigram)
    while words[len(words) - 1] != '</s>' and len(words) < 32:
        words = generate_language(words, probs, bigram, tri_prob, trigram, unigram)
    print('\t', words)

    words = generate_language(word_list.copy(), probs, bigram, tri_prob, trigram, unigram)
    while words[len(words) - 1] != '</s>' and len(words) < 32:
        words = generate_language(words, probs, bigram, tri_prob, trigram, unigram)
    print('\t', words)

    words = generate_language(word_list.copy(), probs, bigram, tri_prob, trigram, unigram)
    while words[len(words) - 1] != '</s>' and len(words) < 32:
        words = generate_language(words, probs, bigram, tri_prob, trigram, unigram)
    print('\t', words)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
