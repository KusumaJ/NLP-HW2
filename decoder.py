import sys
import copy

import numpy as np
import torch

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
from train_model import DependencyModel

dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.extractor = extractor

        # Create a new model and load the parameters
        self.model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):

        state = State(range(1,len(words)))
        state.stack.append(0)

        # TODO: Write the body of this loop for part 5
        while state.buffer:
          features = self.extractor.get_input_representation(words,pos,state)
          predictions = self.model(torch.tensor(features, dtype=torch.long)).detach().numpy()[0]
          flag = False
          for tr in np.sort(predictions)[::-1]:
              tr = np.where(predictions==tr)[0]
              for t in tr: # in case turns out more than 1 transition has the same prob
                if(t==0 and len(state.buffer)!=0): #buffer not empty 
                    state.shift()
                    flag = True
                    break
                if(len(state.stack)!=0 and len(state.buffer)!=0):
                    if(t<=45): 
                        # print("left")
                        state.left_arc(dep_relations[t-1])
                    else: 
                        # print("right")
                        state.right_arc(dep_relations[t-46])
                    flag = True
                    break
              if(flag): break  

        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))

        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {}".format(WORD_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print(deps.print_tree())

# >> python evaluate.py data/model.pt data/dev.conll

# Micro Avg. Labeled Attachment Score: 0.7316421230886723
# Micro Avg. Unlabeled Attachment Score: 0.7798881012069796

# Macro Avg. Labeled Attachment Score: 0.7431484382034693
# Macro Avg. Unlabeled Attachment Score: 0.7911402127390919
