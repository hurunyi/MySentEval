# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
SICK Relatedness and Entailment
'''
from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

from senteval.utils import cosine
from senteval.tools.relatedness import RelatednessPytorch
from senteval.tools.validation import SplitClassifier


class SICKRelatednessEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SICK-Relatedness*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def do_prepare(self, params, prepare):
        samples = self.sick_data['train']['X_A'] + \
                  self.sick_data['train']['X_B'] + \
                  self.sick_data['dev']['X_A'] + \
                  self.sick_data['dev']['X_B'] + \
                  self.sick_data['test']['X_A'] + self.sick_data['test']['X_B']
        return prepare(params, samples)

    def loadFile(self, fpath):
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[3])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        return sick_data

    def run(self, params, batcher):
        sick_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.sick_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_corpus = sorted(zip(self.sick_data[key]['X_A'],
                                       self.sick_data[key]['X_B'],
                                       self.sick_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            self.sick_data[key]['X_A'] = [x for (x, y, z) in sorted_corpus]
            self.sick_data[key]['X_B'] = [y for (x, y, z) in sorted_corpus]
            self.sick_data[key]['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['X_A', 'X_B']:
                sick_embed[key][txt_type] = []
                for ii in range(0, len(self.sick_data[key]['y']), bsize):
                    batch = self.sick_data[key][txt_type][ii:ii + bsize]
                    embeddings = batcher(params, batch)
                    sick_embed[key][txt_type].append(embeddings)
                sick_embed[key][txt_type] = np.vstack(sick_embed[key][txt_type])
            sick_embed[key]['y'] = np.array(self.sick_data[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = sick_embed['train']['X_A']
        trainB = sick_embed['train']['X_B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = self.encode_labels(self.sick_data['train']['y'])

        # Dev
        devA = sick_embed['dev']['X_A']
        devB = sick_embed['dev']['X_B']
        devF = np.c_[np.abs(devA - devB), devA * devB]
        devY = self.encode_labels(self.sick_data['dev']['y'])

        # Test
        testA = sick_embed['test']['X_A']
        testB = sick_embed['test']['X_B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = self.encode_labels(self.sick_data['test']['y'])

        config = {'seed': self.seed, 'nclasses': 5}
        clf = RelatednessPytorch(train={'X': trainF, 'y': trainY},
                                 valid={'X': devF, 'y': devY},
                                 test={'X': testF, 'y': testY},
                                 devscores=self.sick_data['dev']['y'],
                                 config=config)

        devpr, yhat = clf.run()

        pr = pearsonr(yhat, self.sick_data['test']['y'])[0]
        sr = spearmanr(yhat, self.sick_data['test']['y'])[0]
        pr = 0 if pr != pr else pr
        sr = 0 if sr != sr else sr
        se = mean_squared_error(yhat, self.sick_data['test']['y'])
        logging.debug('Dev : Pearson {0}'.format(devpr))
        logging.debug('Test : Pearson {0} Spearman {1} MSE {2} \
                       for SICK Relatedness\n'.format(pr, sr, se))

        return {'devpearson': devpr, 'pearson': pr, 'spearman': sr, 'mse': se,
                'yhat': yhat, 'ndev': len(devA), 'ntest': len(testA)}

    def encode_labels(self, labels, nclass=5):
        """
        Label encoding from Tree LSTM paper (Tai, Socher, Manning)
        """
        Y = np.zeros((len(labels), nclass)).astype('float32')
        for j, y in enumerate(labels):
            for i in range(nclass):
                if i+1 == np.floor(y) + 1:
                    Y[j, i] = y - np.floor(y)
                if i+1 == np.floor(y):
                    Y[j, i] = np.floor(y) - y + 1
        return Y


class SICKEntailmentEval(SICKRelatednessEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SICK-Entailment*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        label2id = {'CONTRADICTION': 0, 'NEUTRAL': 1, 'ENTAILMENT': 2}
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[4])
        sick_data['y'] = [label2id[s] for s in sick_data['y']]
        return sick_data

    def run(self, params, batcher):
        sick_embed = {'train': {}, 'dev': {}, 'test': {}}
        bsize = params.batch_size

        for key in self.sick_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_corpus = sorted(zip(self.sick_data[key]['X_A'],
                                       self.sick_data[key]['X_B'],
                                       self.sick_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            self.sick_data[key]['X_A'] = [x for (x, y, z) in sorted_corpus]
            self.sick_data[key]['X_B'] = [y for (x, y, z) in sorted_corpus]
            self.sick_data[key]['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['X_A', 'X_B']:
                sick_embed[key][txt_type] = []
                for ii in range(0, len(self.sick_data[key]['y']), bsize):
                    batch = self.sick_data[key][txt_type][ii:ii + bsize]
                    embeddings = batcher(params, batch)
                    sick_embed[key][txt_type].append(embeddings)
                sick_embed[key][txt_type] = np.vstack(sick_embed[key][txt_type])
            logging.info('Computed {0} embeddings'.format(key))

        # Train
        trainA = sick_embed['train']['X_A']
        trainB = sick_embed['train']['X_B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = np.array(self.sick_data['train']['y'])

        # Dev
        devA = sick_embed['dev']['X_A']
        devB = sick_embed['dev']['X_B']
        devF = np.c_[np.abs(devA - devB), devA * devB]
        devY = np.array(self.sick_data['dev']['y'])

        # Test
        testA = sick_embed['test']['X_A']
        testB = sick_embed['test']['X_B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = np.array(self.sick_data['test']['y'])

        config = {'nclasses': 3, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': params.classifier,
                  'nhid': params.nhid}
        clf = SplitClassifier(X={'train': trainF, 'valid': devF, 'test': testF},
                              y={'train': trainY, 'valid': devY, 'test': testY},
                              config=config)

        devacc, testacc = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} for \
                       SICK entailment\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(devA), 'ntest': len(testA)}


class MySICKRelatednessEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : MySICK-Relatedness*****\n\n')
        self.seed = seed
        # self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        # self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))

    def loadFile(self, fpath):
        sent1 = []
        sent2 = []
        raw_scores = []
        skipFirstLine = True
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sent1.append(text[1].split())
                    sent2.append(text[2].split())
                    raw_scores.append(text[3])

        raw_scores = np.array(raw_scores)
        not_empty_idx = raw_scores != ''
        gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
        sent1 = np.array(sent1, dtype=object)[not_empty_idx]
        sent2 = np.array(sent2, dtype=object)[not_empty_idx]
        # sort data by length to minimize padding in batcher
        sorted_data = sorted(zip(sent1, sent2, gs_scores), key=lambda z: (len(z[0]), len(z[1]), z[2]))
        sent1, sent2, gs_scores = map(list, zip(*sorted_data))

        self.sick_data = (sent1, sent2, gs_scores)
        self.sample = sent1 + sent2

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

        return prepare(params, self.sample)

    def run(self, params, batcher):
        sys_scores = []
        input1, input2, gs_scores = self.sick_data
        for ii in range(0, len(gs_scores), params.batch_size):
            batch1 = input1[ii:ii + params.batch_size]
            batch2 = input2[ii:ii + params.batch_size]

            # we assume get_batch already throws out the faulty ones
            if len(batch1) == len(batch2) and len(batch1) > 0:
                enc1 = batcher(params, batch1)
                enc2 = batcher(params, batch2)

                for kk in range(enc2.shape[0]):
                    sys_score = self.similarity(enc1[kk], enc2[kk])
                    sys_scores.append(sys_score)

        result = {'pearson': pearsonr(sys_scores, gs_scores), 'spearman': spearmanr(sys_scores, gs_scores),
                  'nsamples': len(sys_scores)}
        logging.debug('pearson = %.4f, spearman = %.4f, nsamples = %d' %
                      (result['pearson'][0], result['spearman'][0], result['nsamples']))

        return result