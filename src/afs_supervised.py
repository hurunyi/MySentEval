import os
import logging
import sys
import torch
from torch import nn
import torch.optim as optim
import csv
import numpy as np
from models import InferSent
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold


PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_W2V = '/data/hurunyi/glove.840B.300d.txt'  # or crawl-300d-2M.vec for V2
MODEL_PATH = 'infersent1.pkl'
V = 1  # version of InferSent
afs_fpath = '/home/hurunyi/SentEval/data/downstream/AFS'


sys.path.insert(0, PATH_SENTEVAL)
import senteval
from senteval import utils


def read_afs_data(fpath):
	datasets = ['ArgPairs_DP', 'ArgPairs_GC', 'ArgPairs_GM']
	sent1 = []
	sent2 = []
	raw_scores = []

	for dataset in datasets:
		with open(fpath + '/%s.csv' % dataset, 'r', encoding='utf-8', errors='ignore') as f:
			skipFirstLine = True
			reader = csv.reader(f)
			for text in reader:
				if skipFirstLine:
					skipFirstLine = False
				else:
					sent1.append(text[9])
					sent2.append(text[10])
					raw_scores.append(text[0])

	sent1 = np.array(sent1)
	sent2 = np.array(sent2)
	gs_scores = np.array([float(x) for x in raw_scores])

	return sent1, sent2, gs_scores


def encode_labels(labels, nclass=5):
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


def prepare(params, samples):
	params.infersent.build_vocab(samples, tokenize=False)


def batcher(params, batch):
	sentences = [' '.join(s) for s in batch]
	embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
	return embeddings


class afs_trainer(object):
	def __init__(self, train, valid, devscores, config):
		# fix seed
		np.random.seed(config['seed'])
		torch.manual_seed(config['seed'])
		assert torch.cuda.is_available(), 'torch.cuda required'
		torch.cuda.manual_seed(config['seed'])

		self.train = train
		self.valid = valid
		self.devscores = devscores

		self.inputdim = train['X'].shape[1]
		self.nclasses = config['nclasses']
		self.seed = config['seed']
		self.l2reg = 0.
		self.batch_size = 64
		self.maxepoch = 1000
		self.early_stop = True

		self.model = nn.Sequential(
			nn.Linear(self.inputdim, self.nclasses),
			nn.Softmax(dim=-1),
		)
		self.loss_fn = nn.MSELoss()

		if torch.cuda.is_available():
			self.model = self.model.cuda()
			self.loss_fn = self.loss_fn.cuda()

		self.loss_fn.size_average = False
		self.optimizer = optim.Adam(self.model.parameters(), weight_decay=self.l2reg)

	def prepare_data(self, trainX, trainy, devX, devy):
		# Transform probs to log-probs for KL-divergence
		trainX = torch.from_numpy(trainX).float().cuda()
		trainy = torch.from_numpy(trainy).float().cuda()
		devX = torch.from_numpy(devX).float().cuda()
		devy = torch.from_numpy(devy).float().cuda()

		return trainX, trainy, devX, devy

	def run(self):
		self.nepoch = 0
		bestpr = -1
		bestsp = -1
		early_stop_count = 0
		r = np.arange(1, 6)
		stop_train = False

		# Preparing data
		trainX, trainy, devX, devy = \
			self.prepare_data(self.train['X'], self.train['y'], self.valid['X'], self.valid['y'])

		# Training
		while not stop_train and self.nepoch <= self.maxepoch:
			self.trainepoch(trainX, trainy, nepoches=50)
			yhat = np.dot(self.predict_proba(devX), r)
			pr = pearsonr(yhat, self.devscores)[0]
			sp = spearmanr(yhat, self.devscores)[0]
			pr = 0 if pr != pr else pr  # if NaN bc std=0
			sp = 0 if sp != sp else sp
			# early stop on Pearson
			if pr + sp > bestpr + bestsp:
				bestpr = pr
				bestsp = sp
			elif self.early_stop:
				if early_stop_count >= 3:
					stop_train = True
				early_stop_count += 1
		print(f"Pearson: {bestpr}, Spearman: {bestsp}")
		return bestpr, bestsp

	def trainepoch(self, X, y, nepoches=1):
		self.model.train()
		for _ in range(self.nepoch, self.nepoch + nepoches):
			permutation = np.random.permutation(len(X))
			all_costs = []
			for i in range(0, len(X), self.batch_size):
				# forward
				idx = torch.from_numpy(permutation[i:i + self.batch_size]).long().cuda()
				Xbatch = X[idx]
				ybatch = y[idx]
				output = self.model(Xbatch)
				# loss
				loss = self.loss_fn(output, ybatch)
				all_costs.append(loss.item())
				# backward
				self.optimizer.zero_grad()
				loss.backward()
				# Update parameters
				self.optimizer.step()
		self.nepoch += nepoches

	def predict_proba(self, devX):
		self.model.eval()
		probas = []
		with torch.no_grad():
			for i in range(0, len(devX), self.batch_size):
				Xbatch = devX[i:i + self.batch_size]
				if len(probas) == 0:
					probas = self.model(Xbatch).data.cpu().numpy()
				else:
					probas = np.concatenate((probas, self.model(Xbatch).data.cpu().numpy()), axis=0)
		return probas


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

	config = {'seed': 1111, 'nclasses': 5}
	params = utils.dotdict(params_senteval)
	params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
	params.seed = 1111 if 'seed' not in params else params.seed

	params.batch_size = 128 if 'batch_size' not in params else params.batch_size
	params.nhid = 0 if 'nhid' not in params else params.nhid
	params.kfold = 5 if 'kfold' not in params else params.kfold
	if 'classifier' not in params or not params['classifier']:
		params.classifier = {'nhid': 0}

	assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

	k = 10
	splits = KFold(n_splits=k, shuffle=True, random_state=42)
	sent1, sent2, scores = read_afs_data(afs_fpath)
	s1 = sent1.tolist()
	s2 = sent2.tolist()
	samples = s1 + s2
	prepare(params, samples)

	pr_avg = 0
	sp_avg = 0

	for fold, (train_idx, dev_idx) in enumerate(splits.split(np.arange(len(sent1)))):
		print(f"Fold {fold+1}:")
		sent1_embed, sent2_embed = [], []
		bsize = params.batch_size
		for ii in range(0, len(sent1), bsize):
			batch1 = sent1[ii:ii + bsize]
			batch2 = sent2[ii:ii + bsize]
			embeddings1 = batcher(params, batch1)
			embeddings2 = batcher(params, batch2)
			sent1_embed.append(embeddings1)
			sent2_embed.append(embeddings2)
		sent1_embed = np.vstack(sent1_embed)
		sent2_embed = np.vstack(sent2_embed)

		# Train
		trainA = sent1_embed[train_idx]
		trainB = sent2_embed[train_idx]
		trainScores = scores[train_idx]
		trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
		trainY = encode_labels(trainScores)

		# Dev
		devA = sent1_embed[dev_idx]
		devB = sent2_embed[dev_idx]
		devScores = scores[dev_idx]
		devF = np.c_[np.abs(devA - devB), devA * devB]
		devY = encode_labels(devScores)

		trainer = afs_trainer(train={'X': trainF, 'y': trainY}, valid={'X': devF, 'y': devY}, devscores=devScores,
							  config=config)
		pr, sp = trainer.run()
		pr_avg += pr
		sp_avg += sp

	pr_avg /= k
	sp_avg /= k
	print(f"\nPearson_avg: {pr_avg}, Spearman_avg: {sp_avg}")


if __name__ == "__main__":
	main()
