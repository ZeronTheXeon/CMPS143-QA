#!/usr/bin/env python
'''
Created on May 14, 2014
@author: reid

Modified on May 21, 2015
'''

import sys, nltk, operator
from nltk.stem import WordNetLemmatizer
from qa_engine.base import QABase
import csv
from collections import defaultdict
from nltk.corpus import wordnet as wn

DATA_DIR = "./wordnet"







# The standard NLTK pipeline for POS tagging a document
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    return sentences


def get_bow(tagged_tokens, stopwords):

    noun_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_nouns.csv"))
    verb_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_verbs.csv"))
    wordnet_lemmatizer = WordNetLemmatizer()
    words = [set([t[0].lower() for t in tagged_tokens if t[0].lower() not in stopwords])]
    words = set()
    for x in tagged_tokens:
        if x[0].lower() not in stopwords:
            if "VB" in x[1]:
                words.add(wordnet_lemmatizer.lemmatize(x[0].lower(), pos='v'))
            else:
                words.add(wordnet_lemmatizer.lemmatize(x[0].lower(), pos='n'))

    return words


def find_phrase(tagged_tokens, qbow):
    for i in range(len(tagged_tokens) - 1, 0, -1):
        word = (tagged_tokens[i])[0]
        if word in qbow:
            return tagged_tokens[i + 1:]


# qtokens: is a list of pos tagged question tokens with SW removed
# sentences: is a list of pos tagged story sentences
# stopwords is a set of stopwords
def baseline(qtokens, sentences, stopwords):
    # Collect all the candidate answers
    answers = []
    for i in range(len(sentences)):
        sent = sentences[i]
        # A list of all the word tokens in the sentence
        sbow = get_bow(sent, stopwords)
        qbow = get_bow(qtokens, stopwords)

        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)
        # if overlap > 0:
        #     print(sbow)

        answers.append((overlap, sent, i))  # this may be i + 1

    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    # Return the best answer
    best_answer = answers[0][1]
    best_sent_number = answers[0][2]
    return best_answer, best_sent_number

def load_wordnet_ids(filename):
    file = open(filename, 'r')
    if "noun" in filename: type = "noun"
    else: type = "verb"
    csvreader = csv.DictReader(file, delimiter=",", quotechar='"')
    word_ids = defaultdict()
    for line in csvreader:
        word_ids[line['synset_id']] = {'synset_offset': line['synset_offset'], 'story_'+type: line['story_'+type], 'stories': line['stories']}
    return word_ids

def get_bow_wordnet(tagged_tokens, stopwords):

    noun_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_nouns.csv"))
    verb_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_verbs.csv"))

    wordnet_lemmatizer = WordNetLemmatizer()
    words = [set([t[0].lower() for t in tagged_tokens if t[0].lower() not in stopwords])]
    words = set()
    for x in tagged_tokens:
        if x[0].lower() not in stopwords:
            if "VB" in x[1]:
                new_words = []
                temp = wordnet_lemmatizer.lemmatize(x[0].lower(), pos='v')
                verb_synset = wn.synsets(temp)
                for syn in verb_synset:
                    new_words.append(syn.name()[0:syn.name().index(".")])
                    verb_hypernym = syn.hypernyms()
                    for hyp in verb_hypernym:
                        new_words.append(hyp.name()[0:hyp.name().index(".")])
                for x in new_words:
                    words.add(x)
                #word.add(new_words)

            else:
                new_words = []
                temp = wordnet_lemmatizer.lemmatize(x[0].lower(), pos='n')
                noun_synset = wn.synsets(temp)
                for syn in noun_synset:
                    new_words.append(syn.name()[0:syn.name().index(".")])
                    noun_hypernym = syn.hypernyms()
                    for hyp in noun_hypernym:
                        new_words.append(hyp.name()[0:hyp.name().index(".")])
                for x in new_words:
                    words.add(x)
                #word.add(new_words)

    return words


# qtokens: is a list of pos tagged question tokens with SW removed
# sentences: is a list of pos tagged story sentences
# stopwords is a set of stopwords
def baseline_wordnet(qtokens, sentences, stopwords):
    # Collect all the candidate answers
    answers = []
    for i in range(len(sentences)):
        sent = sentences[i]
        # A list of all the word tokens in the sentence
        sbow = get_bow_wordnet(sent, stopwords)
        qbow = get_bow_wordnet(qtokens, stopwords)

        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)
        # if overlap > 0:
        #     print(sbow)

        answers.append((overlap, sent, i))  # this may be i + 1

    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    # Return the best answer
    best_answer = answers[0][1]
    best_sent_number = answers[0][2]
    return best_answer, best_sent_number





if __name__ == '__main__':
    question_id = "fables-01-1"

    driver = QABase()
    q = driver.get_question(question_id)
    story = driver.get_story(q["sid"])
    text = story["text"]
    question = q["text"]
    print("question:", question)

    # These stopwords are used for the better sentene recall. Overall it drops our F stop so when 
    # we are using chunking, set the baseline call to use stopwords not stopwrods1
    stopwords1 = set(nltk.corpus.stopwords.words("english"))
    stopwords1.union({"who", "what", "when", "where", "why", "'s"})
    stopwords1 = stopwords1 - {"had", "have", "from"}

    qbow = get_bow_wordnet(get_sentences(question)[0], stopwords1)
    sentences = get_sentences(text)
    answer, number = baseline_wordnet(qbow, sentences, stopwords1)
    answerText = ""
    for (x, y) in answer:
        answerText += (" " if x[0].isalnum() else "") + x
    print("answer:", answerText)
