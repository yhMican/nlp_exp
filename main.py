from wikipedia2vec import Wikipedia2Vec
import pickle   # For decompressing .pkl file

# For NER
import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint

# NER: pick up candidate words/phrases
# class NER:
#     def __init__(self):
#         self.


# NED & NEL
#   Disambiguate the semantics of the word
#   Link the node with other nodes


def preprocess(sentence):
    sentence = nltk.word_tokenize(sentence)
    sentence = nltk.pos_tag(sentence)
    return sentence


def nltkDemo():
    ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'
    sentence = preprocess(ex)
    print(sentence)

    pattern = 'NP: {<DT>?<JJ>*<NN>}'
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(sentence)
    print(cs)

    iob_tagged = tree2conlltags(cs)
    pprint(iob_tagged)


def wikipedia2VecDemo():
    with open('enwiki_20180420_100d.pkl.bz2', 'rb') as MODEL_FILE:
        model = Wikipedia2Vec.load(MODEL_FILE)
        print(model.get_entity_vector('Scarlett Johansson'))


if __name__ == '__main__':
    # wikipedia2VecDemo()
    nltkDemo()