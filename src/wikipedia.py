import csv
import os
import sys
from tqdm import tqdm
import logging
import torch
from models import InferSent
import numpy as np


PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_W2V = '/data/hurunyi/glove.840B.300d.txt'  # or crawl-300d-2M.vec for V2
MODEL_PATH = 'infersent1.pkl'
V = 1  # version of InferSent
wikipedia_path = "/data/hurunyi/wikipedia"


sys.path.insert(0, PATH_SENTEVAL)
import senteval
from senteval import utils
from senteval.utils import cosine


def read_data(fpath):
	sent1 = []
	sent2 = []
	sent3 = []

	with open(os.path.join(fpath, "test.csv"), "r") as f:
		skipFirstLine = True
		reader = csv.reader(f, delimiter=',', quotechar='"')
		print("Loading data...")
		for text in tqdm(reader):
			if skipFirstLine:
				skipFirstLine = False
			else:
				sent1.append(text[1])
				sent2.append(text[2])
				sent3.append(text[3])
		print("Successfully loaded!")
	return sent1, sent2, sent3


def prepare(params, samples):
	params.infersent.build_vocab(samples, tokenize=False)


def batcher(params, batch):
	sentences = [' '.join(s) for s in batch]
	embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
	return embeddings


def main():
	assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), \
		'Set MODEL and GloVe PATHs'

	logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

	params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
	params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
									 'tenacity': 3, 'epoch_size': 2}

	params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
					'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
	model = InferSent(params_model)
	model.load_state_dict(torch.load(MODEL_PATH))
	model.set_w2v_path(PATH_TO_W2V)

	params_senteval['infersent'] = model.cuda()

	params = utils.dotdict(params_senteval)
	params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
	params.seed = 1111 if 'seed' not in params else params.seed

	params.batch_size = 128 if 'batch_size' not in params else params.batch_size
	params.nhid = 0 if 'nhid' not in params else params.nhid
	params.kfold = 5 if 'kfold' not in params else params.kfold
	if 'classifier' not in params or not params['classifier']:
		params.classifier = {'nhid': 0}

	assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

	sent1, sent2, sent3 = read_data(wikipedia_path)
	samples = sent1 + sent2 + sent3
	prepare(params, samples)
	similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
	sent_num = len(sent1)
	right_cot = 0

	print("Begin to predict!")
	for i in tqdm(range(0, len(sent1), params.batch_size)):
		batch1 = sent1[i:i + params.batch_size]
		batch2 = sent2[i:i + params.batch_size]
		batch3 = sent3[i:i + params.batch_size]

		# we assume get_batch already throws out the faulty ones
		if len(batch1) == len(batch2) and len(batch1) == len(batch3) and len(batch1) > 0:
			enc1 = batcher(params, batch1)
			enc2 = batcher(params, batch2)
			enc3 = batcher(params, batch3)

			for j in range(enc1.shape[0]):
				score2 = similarity(enc1[j], enc2[j])
				score3 = similarity(enc1[j], enc3[j])
				if score2 > score3:
					right_cot += 1

	print(f"Acc: {right_cot} / {sent_num} = {right_cot/sent_num}")


if __name__ == "__main__":
	main()
