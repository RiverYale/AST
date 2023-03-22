import argparse
import logging
import yaml
import random
import time
import os
from tqdm import tqdm as tqdm_
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.cuda import amp
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torch.optim import lr_scheduler

from utils import *
from datasets.University1652 import University1652_train_dataset, University1652_test_dataset
from models.AST import AST
from criterion.soft_triplet import SoftTripletBiLoss
from criterion.multi_similarity_loss import MultiSimilarityLoss
from criterion.triplet_loss import TripletLoss, Tripletloss
from pytorch_metric_learning import losses, miners  # pip install pytorch-metric-learning
from criterion.circle_loss import CircleLoss, convert_label_to_similarity


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='code/config.yaml', type=str, metavar='FILE', help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--seed', default=None, type=int, metavar='NUM',
					help='seed for initializing training')
parser.add_argument('--root', default='', type=str, metavar='PATH',
					help='path to load dataset (default: none)')
parser.add_argument('--evaluate', action='store_true',
					help='only run in evaluate mode')

parser.add_argument('--drone-h', default=224, type=int, metavar='NUM',
					help='drone image input height')
parser.add_argument('--drone-w', default=224, type=int, metavar='NUM',
					help='drone image input width')
parser.add_argument('--sate-h', default=224, type=int, metavar='NUM',
					help='satellite image input height')
parser.add_argument('--sate-w', default=224, type=int, metavar='NUM',
					help='satellite image input width')
parser.add_argument('--pad', default=5, type=int, metavar='NUM',
					help='input padding')
parser.add_argument('--erasing-p', default=0, type=float,
					help='random erasing probability, in [0,1]')
parser.add_argument('--color-jitter', action='store_true',
					help='use color jitter in training')
parser.add_argument('--DA', action='store_true',
					help='use Color Data Augmentation')
parser.add_argument('--num-worker', default=4, type=int, metavar='NUM',
					help='num worker for dataloader')

parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
					help='path to load checkpoint (default: none)')
parser.add_argument('--only-state', action='store_true',
					help='only load the model params except training progress from checkpoint')
parser.add_argument('--checkpoint-freq', default=20, type=int, metavar='N',
					help='save checkpoint frequency (default: 20)， if 0, only save in the end')
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
					help='print frequency (default: 10)')

parser.add_argument('--save-path', default='', type=str, metavar='PATH',
					help='path to save checkpoint (default: none)')
parser.add_argument('--save-log', action='store_true',
					help='whether to save output log')

parser.add_argument('--share-weight', action='store_true',
					help='two branches share the same weight')
parser.add_argument('--sample-num', default=8, type=int, metavar='N',
					help='sample n drone images from each class')
parser.add_argument('--test-batch-size', default=128, type=int, metavar='N',
					help='mini-batch size for test dataloader')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
					help='mini-batch size (default: 32)')
parser.add_argument('--lr', default=0.01, type=float, metavar='LR',
					help='initial learning rate')
parser.add_argument('--lr-tscale', default=1, type=float,
					help='initial learning rate scale of transformer')
parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam', 'adamw'],
					help='optimizer')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum of SGD solver')
parser.add_argument('--weight-decay', default=0, type=float, metavar='W',
					help='weight decay (default: 1e-4)')
parser.add_argument('--steps', default=[7, 20], nargs='*', type=int,
					help='learning rate schedule (when to drop lr)')
parser.add_argument('--ft-dim', default=512, type=int,
					help='the feature vector dimension')
parser.add_argument('--alpha', default=0, type=int,
					help='alpha for weight soft margin triplet loss')

parser.add_argument('--num-epochs', default=50, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--enable-amp', action='store_true',
					help='enable torch amp mode')
parser.add_argument('--warm-epoch', default=5, type=int, metavar='N',
					help='warm epoch num')

parser.add_argument('--part-num', default=1, type=int,
					help='the number of split part')


def _parse_args():
	# Do we have a config file to parse?
	args_config, remaining = config_parser.parse_known_args()
	if args_config.config:
		with open(args_config.config, 'r') as f:
			cfg = yaml.safe_load(f)
			parser.set_defaults(**cfg)

	# The main arg parser parses the rest of the args, the usual
	# defaults will have been overridden if config file specified.
	args = parser.parse_args(remaining)

	# Cache the args as a text string to save them in the output dir later
	args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
	return args, args_text


def get_logger(args):
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)

	if args.save_log:
		log_path = args.save_path + f'/train_{time.strftime("%Y%m%d%H%M")}.log'
		file_hander = logging.FileHandler(log_path, encoding='utf-8')
		file_fmt = logging.Formatter(fmt='%(message)s')
		file_hander.setLevel(logging.INFO)
		file_hander.setFormatter(file_fmt)
		logger.addHandler(file_hander)

	console_hander = logging.StreamHandler()
	consoler_fmt = logging.Formatter(fmt='%(message)s')
	console_hander.setLevel(logging.DEBUG)
	console_hander.setFormatter(consoler_fmt)
	logger.addHandler(console_hander)

	return logger


logger = None
def main():
	global logger

	args, args_text = _parse_args()
	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)	
	logger = get_logger(args)

	for k in args.__dict__:
		logger.info(f'{k:>15s} | {args.__dict__[k]}')

	if args.seed is not None:
		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		logger.info('You have chosen to seed training. '
					  'This will turn on the CUDNN deterministic setting, '
					  'which can slow down your training considerably! '
					  'You may see unexpected behavior when restarting '
					  'from checkpoints.')

	logger.info('Preparing for training...')

	# ====> load dataset <====
	train_dataset = University1652_train_dataset(args)
	query_drone_dataset = University1652_test_dataset(args, ['query_drone'], (args.drone_h, args.drone_w))
	query_satellite_dataset = University1652_test_dataset(args, ['query_satellite'], (args.sate_h, args.sate_w))
	gallery_drone_dataset = University1652_test_dataset(args, ['gallery_drone'], (args.drone_h, args.drone_w))
	gallery_satellite_dataset = University1652_test_dataset(args, ['gallery_satellite'], (args.sate_h, args.sate_w))

	args.num_classes = train_dataset.get_class_num()

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
						num_workers=args.num_worker, pin_memory=True, sampler=None, drop_last=True)
	query_drone_loader = DataLoader(query_drone_dataset, batch_size=args.test_batch_size, shuffle=False,
						num_workers=args.num_worker, pin_memory=True, sampler=None, drop_last=False)
	query_satellite_loader = DataLoader(query_satellite_dataset, batch_size=args.test_batch_size, shuffle=False,
						num_workers=args.num_worker, pin_memory=True, sampler=None, drop_last=False)
	gallery_drone_loader = DataLoader(gallery_drone_dataset, batch_size=args.test_batch_size, shuffle=False,
						num_workers=args.num_worker, pin_memory=True, sampler=None, drop_last=False)
	gallery_satellite_loader = DataLoader(gallery_satellite_dataset, batch_size=args.test_batch_size, shuffle=False,
						num_workers=args.num_worker, pin_memory=True, sampler=None, drop_last=False)

	# ====> load model <====
	cudnn.benchmark = True
	model = AST(args)
	model.cuda()

	# ====> load criterion <====
	criterion = SoftTripletBiLoss(alpha=args.alpha).cuda()

	# ====> load optimizer <====
	parameters_filt_list = []
	for key in model.drone_module._modules.keys():
		if '_head' in key:
			module = getattr(model.drone_module, key)
			parameters_filt_list += list(map(id, module.parameters()))
	for key in model.sate_module._modules.keys():
		if '_head' in key:
			module = getattr(model.sate_module, key)
			parameters_filt_list += list(map(id, module.parameters()))
	parameters_filt_list = list(set(parameters_filt_list))
	class_parameters = list(filter(lambda p: p.requires_grad and id(p) in parameters_filt_list, model.parameters()))
	trans_parameters = list(filter(lambda p: p.requires_grad and id(p) not in parameters_filt_list, model.parameters()))

	parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
	if args.optim == 'sgd':
		optimizer = torch.optim.SGD([
			{'params': trans_parameters, 'lr': args.lr_tscale * args.lr},
			{'params': class_parameters, 'lr': args.lr}
		], momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
	elif args.optim == 'adam':
		optimizer = torch.optim.Adam(parameters, args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
	elif args.optim == 'adamw':
		optimizer = torch.optim.AdamW(parameters, args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
	
	# ====> load scheduler <====
	# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=0.1)
	# scheduler = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
	# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)

	# ====> load checkpoint <====
	args.start_epoch = 0
	if args.checkpoint:
		logger.info(f'Loading checkpoint{"[only state dict]" if args.only_state else ""} from {args.checkpoint} ...')
		checkpoint = torch.load(args.checkpoint)
		model.load_checkpoint(checkpoint['state_dict'])
		if not args.only_state:
			args.start_epoch = checkpoint['epoch']

	# ====> optionally, evaluate only <====
	if args.evaluate:
		logger.info('Start evaluating ...')
		validate(query_drone_loader, gallery_satellite_loader, model, 'drone->satellite', args)
		validate(query_satellite_loader, gallery_drone_loader, model, 'satellite->drone', args)
		return

	# ====> start training <====
	logger.info(f'Start training ...')
	scaler = amp.GradScaler(enabled=args.enable_amp)
	end, spend = datetime.now(), 0
	for epoch in range(args.start_epoch, args.num_epochs):
		predict_end = '?' if spend==0 else (end+spend*(args.num_epochs-epoch)).strftime('%H:%M:%S')
		logger.info(f'Start traning epoch: {epoch}/{args.num_epochs}, finish at {predict_end}')
		train_one_epoch(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, args)
		spend = datetime.now() - end
		end = datetime.now()
		
		if epoch == args.num_epochs - 1 or args.checkpoint_freq != 0 and (epoch + 1) % args.checkpoint_freq == 0:
			ds_topk, ds_mAP = validate(query_drone_loader, gallery_satellite_loader, model, 'drone->satellite', args)
			sd_topk, sd_mAP = validate(query_satellite_loader, gallery_drone_loader, model, 'satellite->drone', args)

			logger.info(f'Save checkpoint_{(epoch+1):03d} ...')
			save_checkpoint({
				'epoch': epoch+1,
				'state_dict': model.state_dict(),
				'drone->satellite': [ds_topk, ds_mAP],
				'satellite->drone': [sd_topk, sd_mAP],
				'args': args,
			}, path=args.save_path, filename=f'checkpoint_{(epoch+1):03d}.pth.tar')
			logger.info(f'Successfully saved to {args.save_path}/checkpoint_{(epoch+1):03d}.pth.tar')
	

def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, args):
	ce_criterion = nn.CrossEntropyLoss().cuda()

	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	loss_meter = AverageMeter('Loss', ':.4e')
	progress = ProgressMeter(
		len(train_loader),
		# [batch_time, data_time, loss_meter, mean_ps, mean_ns],
		[batch_time, data_time, loss_meter],
		prefix="Epoch: [{}]".format(epoch),
		logger=logger)

	warm_up = 0.1  # We start from the 0.1*lrRate
	warm_iteration = len(train_loader) * args.warm_epoch

	# Set model to training mode
	model.train(True)

	end = time.time()
	for i, (labels, imgs, paths) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		labels = labels.cuda(non_blocking=True)
		drone_img = imgs[0].cuda(non_blocking=True)
		sate_img = imgs[1].cuda(non_blocking=True)

		# zero the parameter gradients	
		optimizer.zero_grad()

		with amp.autocast(enabled=args.enable_amp):
			drone_outputs, sate_outputs = model(drone_img, sate_img, paths=paths)
		drone_class, drone_feat = drone_outputs
		sate_class, sate_feat = sate_outputs

		triplet_loss, drone_loss, sate_loss = 0, 0, 0
		for p in range(args.part_num + 1):
			loss_p, _, _ = criterion(drone_feat[:,:,p], sate_feat[:,:,p], labels)
			triplet_loss += loss_p
			drone_loss += ce_criterion(drone_class[:,:,p], labels)
			sate_loss += ce_criterion(sate_class[:,:,p], labels)
		triplet_loss /= args.part_num + 1
		drone_loss /= args.part_num + 1
		sate_loss /= args.part_num + 1
		
		loss = drone_loss + sate_loss + triplet_loss

		loss_meter.update(loss.item(), labels.size(0))

		# backward + optimize only if in training phase
		if epoch < args.warm_epoch:
			warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
			loss *= warm_up

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			progress.display(i)

	if epoch >= args.warm_epoch:
		# scheduler.step()
		scheduler.step(loss)


def validate(query_loader, gallery_loader, model, mode, args):
	logger.info(f'Start validating: {mode}')

	batch_time = AverageMeter('Time', ':6.3f')
	progress_q = ProgressMeter(
		len(query_loader),
		[batch_time],
		prefix='Test_query: ',
		logger=logger)
	progress_k = ProgressMeter(
		len(gallery_loader),
		[batch_time],
		prefix='Test_gallery: ',
		logger=logger)
	
	if mode == 'drone->satellite':
		query_module = model.drone_module
		gallery_module = model.sate_module
	elif mode == 'satellite->drone':
		query_module = model.sate_module
		gallery_module = model.drone_module

	query_module.cuda()
	gallery_module.cuda()
	query_module.eval()
	gallery_module.eval()

	query_features = np.zeros([len(query_loader.dataset), args.ft_dim, args.part_num+1])
	query_labels = np.zeros([len(query_loader.dataset)])
	gallery_features = np.zeros([len(gallery_loader.dataset), args.ft_dim, args.part_num+1])
	gallery_labels = np.zeros([len(gallery_loader.dataset)])

	gallery_paths = []
	query_paths = []

	with torch.no_grad():
		end = time.time()
		# gallery features
		idx = 0
		for i, (labels, images, paths) in enumerate(tqdm_(gallery_loader, ncols=80)):
			labels = labels.cuda(non_blocking=True)
			images = images.cuda(non_blocking=True)
			now_batch_size = images.size()[0]

			# compute output
			gallery_embed = gallery_module(images)
			gallery_features[idx:idx+now_batch_size, :] = gallery_embed.detach().cpu().numpy()
			gallery_labels[idx:idx+now_batch_size] = labels.cpu().numpy()

			idx += now_batch_size

			gallery_paths += paths

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

		progress_k.display(i+1)

		end = time.time()
		# query features
		idx = 0
		for i, (labels, images, paths) in enumerate(tqdm_(query_loader, ncols=80)):
			labels = labels.cuda(non_blocking=True)
			images = images.cuda(non_blocking=True)
			now_batch_size = images.size()[0]

			# compute output
			query_embed = query_module(images)  # delta
			query_features[idx:idx+now_batch_size, :] = query_embed.detach().cpu().numpy()
			query_labels[idx:idx+now_batch_size] = labels.cpu().numpy()

			idx += now_batch_size

			query_paths += paths

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

		progress_q.display(i+1)

		topk, mAP, similarity = accuracy(query_features, query_labels, gallery_features, gallery_labels, topk=[1,5,10,0.01])

		logger.info(f'top1:{topk[0]:.4f}, top5:{topk[1]:.4f}, top10:{topk[2]:.4f}, top1%:{topk[3]:.4f}, mAP:{mAP:.4f}')

	if args.evaluate:
		if mode == 'drone->satellite':
			query_file = os.path.join(args.save_path, 'query_drone.txt')
			gallery_file = os.path.join(args.save_path, 'gallery_satellite.txt')
			similarity_file = os.path.join(args.save_path, 'drone_satellite_similarity.npy')
		elif mode == 'satellite->drone':
			query_file = os.path.join(args.save_path, 'query_satellite.txt')
			gallery_file = os.path.join(args.save_path, 'gallery_drone.txt')
			similarity_file = os.path.join(args.save_path, 'satellite_drone_similarity.npy')
		with open(query_file, 'w') as f:
			for path in query_paths:
				f.write(path + '\n')
		with open(gallery_file, 'w') as f:
			for path in gallery_paths:
				f.write(path + '\n')
		np.save(similarity_file, similarity)

	return topk, mAP


if __name__ == '__main__':
	end = datetime.now()
	main()
	logger.info(f'Program finished after {datetime.now()-end}')
