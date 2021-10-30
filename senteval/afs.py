from __future__ import absolute_import, division, unicode_literals

import os
import io
import numpy as np
import logging
import csv

from scipy.stats import spearmanr, pearsonr

from senteval.utils import cosine


class AFSEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : AFS*****\n\n')
        self.seed = seed
        self.datasets = ['ArgPairs_DP', 'ArgPairs_GC', 'ArgPairs_GM']
        self.loadFile(task_path)

    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        for dataset in self.datasets:
            sent1 = []
            sent2 = []
            raw_scores = []
            skipFirstLine = True
            with io.open(fpath + '/%s.csv' % dataset, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                for text in reader:
                    if skipFirstLine:
                        skipFirstLine = False
                    else:
                        sent1.append(text[9].split())
                        sent2.append(text[10].split())
                        raw_scores.append(text[0])

            raw_scores = np.array(raw_scores)
            not_empty_idx = raw_scores != ''
            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
            sent1 = np.array(sent1, dtype=object)[not_empty_idx]
            sent2 = np.array(sent2, dtype=object)[not_empty_idx]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores), key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

        return prepare(params, self.samples)

    def run(self, params, batcher):
        results = {}
        for dataset in self.datasets:
            sys_scores = []
            input1, input2, gs_scores = self.data[dataset]
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

            results[dataset] = {'pearson': pearsonr(sys_scores, gs_scores),
                                'spearman': spearmanr(sys_scores, gs_scores),
                                'nsamples': len(sys_scores)}
            logging.debug('%s : pearson = %.4f, spearman = %.4f' %
                          (dataset, results[dataset]['pearson'][0],
                           results[dataset]['spearman'][0]))

        weights = [results[dset]['nsamples'] for dset in results.keys()]
        list_prs = np.array([results[dset]['pearson'][0] for
                            dset in results.keys()])
        list_spr = np.array([results[dset]['spearman'][0] for
                            dset in results.keys()])

        avg_pearson = np.average(list_prs)
        avg_spearman = np.average(list_spr)
        wavg_pearson = np.average(list_prs, weights=weights)
        wavg_spearman = np.average(list_spr, weights=weights)

        results['all'] = {'pearson': {'mean': avg_pearson,
                                      'wmean': wavg_pearson},
                          'spearman': {'mean': avg_spearman,
                                       'wmean': wavg_spearman}}
        logging.debug('ALL (weighted average) : Pearson = %.4f, \
            Spearman = %.4f' % (wavg_pearson, wavg_spearman))
        logging.debug('ALL (average) : Pearson = %.4f, \
            Spearman = %.4f\n' % (avg_pearson, avg_spearman))

        return results