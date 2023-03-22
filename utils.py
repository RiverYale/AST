import os, shutil, tqdm
import numpy as np
import torch

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix="", logger=None):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix
		self.logger = logger

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		if self.logger is None:
			print('\t'.join(entries))
		else:
			self.logger.info('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ProgressBar(ProgressMeter):
	def __init__(self, num_batches, meters, prefix="", logger=None):
		super().__init__(num_batches, meters, prefix, logger)
		self.pbar = tqdm.tqdm(total=num_batches, )

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		if self.logger is None:
			print('\t'.join(entries))
		else:
			self.logger.info('\t'.join(entries))


def save_checkpoint(state, path, filename, is_best=False):
	torch.save(state, os.path.join(path, filename))
	if is_best:
		shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'model_best.pth.tar'))


def accuracy(query_features, query_labels, gallery_features, gallery_labels, topk=[1,5,10,0.01]):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	N = query_features.shape[0]
	M = gallery_features.shape[0]

	query_features_norm = np.sqrt(np.sum(query_features**2, axis=1, keepdims=True)) * np.sqrt(query_features.shape[2])
	gallery_features_norm = np.sqrt(np.sum(gallery_features**2, axis=1, keepdims=True)) * np.sqrt(gallery_features.shape[2])
	query_features_normed = (query_features / query_features_norm).reshape(N, -1)
	gallery_features_normed = (gallery_features / gallery_features_norm).reshape(M, -1)
	similarity = np.matmul(query_features_normed, gallery_features_normed.transpose())

	cmc = torch.IntTensor(M).zero_()
	ap = 0
	for i in range(N):
		rank_gallery_idx = np.argsort(similarity[i])[::-1]
		right_gallery_idx = np.argwhere(gallery_labels==query_labels[i])
		right_rank_idx = np.argwhere(np.in1d(rank_gallery_idx, right_gallery_idx)==True)

		cmc[right_rank_idx[0][0]:] += 1

		# Follow the evaluation in University-1652
		for i in range(len(right_rank_idx)):
			d_recall = 1/len(right_rank_idx)
			precision = (i+1) / (right_rank_idx[i][0]+1)

			if right_rank_idx[i][0] != 0:
				old_precision = i / right_rank_idx[i][0]
			else:
				old_precision = 1.0
				
			ap += d_recall*((old_precision + precision)/2)

	results = np.zeros([len(topk)])
	mAP = ap / N * 100
	for i, k in enumerate(topk):
		if k < 1:
			results[i] = cmc[int(N*k)-1] / N * 100
		else:
			results[i] = cmc[k-1] / N * 100

	return results, mAP, similarity