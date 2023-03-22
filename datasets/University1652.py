import os
import glob
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from .autoaugment import ImageNetPolicy
from .random_erasing import RandomErasing


class University1652(Dataset):

	def __init__(self, root, mode, sample_num, view_types, transform_dict):
		super(University1652, self).__init__()
		
		self.root = root
		self.mode = mode
		self.view_types = view_types
		self.transform_dict = transform_dict
		self.to_tensor = transforms.ToTensor()
		
		if self.mode == 'train':
			# {view1:{class1:[class1_1.jpg],class2:[class2_1.jpg]}}
			self.view_dict = {}
			for view in view_types:
				class_dict = {}
				for class_name in os.listdir(os.path.join(root, mode, view)):
					class_dict[class_name] = glob.glob(os.path.join(root, mode, view, class_name, '*'))
				self.view_dict[view] = class_dict

			class_names = os.listdir(os.path.join(root, mode, view_types[0]))
			self.idx_to_class = {i:class_name for i, class_name in enumerate(class_names)}
			self.sample_num = sample_num
			self.class_num = len(class_names)
			self.len = self.class_num * self.sample_num
		elif self.mode == 'test' or 'scan' in self.mode:
			if 'test' in self.mode:
				mode = 'test'
			elif 'train' in self.mode:
				mode = 'train'
			test_data = []
			class_names = []
			for view in view_types:
				test_data += glob.glob(os.path.join(root, mode, view, '*', '*'))
				class_names += os.listdir(os.path.join(root, mode, view))
			test_data.sort()
			self.test_data = test_data
			self.class_num = len(list(set(class_names)))
			self.len = len(self.test_data)

	def __getitem__(self, index):
		if self.mode == 'train':
			class_idx = index // self.sample_num
			class_name = self.idx_to_class[class_idx]
			items = []
			img_paths = []
			for view in self.view_types:
				imgs = self.view_dict[view][class_name]
				img_path = np.random.choice(imgs, 1)[0]
				img = Image.open(img_path)
				img = self.transform_dict[view](img)
				items.append(img)
				img_paths.append(img_path[len(self.root)+1:])
			return class_idx, items, img_paths
		elif self.mode == 'test' or 'scan' in self.mode:
			view, class_name = self.test_data[index].split(os.path.sep)[-3:-1]
			img = Image.open(self.test_data[index])
			img = self.transform_dict[view](img)
			return int(class_name), img, self.test_data[index][len(self.root)+1:]

	def __len__(self):
		return self.len

	def get_class_num(self):
		return self.class_num


# Load data for training
def University1652_train_dataset(args):
	transform_drone_list = [
		# transforms.RandomResizedCrop(size=(args.drone_h, args.drone_w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
		transforms.Resize((args.drone_h, args.drone_w), interpolation=InterpolationMode.BICUBIC),
		transforms.Pad(args.pad, padding_mode='edge'),
		transforms.RandomCrop((args.drone_h, args.drone_w)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]
	transform_satellite_list = [
		transforms.Resize((args.sate_h, args.sate_w), interpolation=InterpolationMode.BICUBIC),
		transforms.Pad(args.pad, padding_mode='edge'),
		transforms.RandomAffine(90),
		transforms.RandomCrop((args.sate_h, args.sate_w)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]

	if args.erasing_p > 0:
		transform_drone_list = transform_drone_list + [RandomErasing(probability=args.erasing_p, mean=[0.0, 0.0, 0.0])]

	if args.color_jitter:
		transform_drone_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_drone_list
		transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_satellite_list

	if args.DA:
		transform_drone_list = [ImageNetPolicy()] + transform_drone_list
	
	root = args.root
	mode = 'train'
	sample_num = args.sample_num
	view_types = ['drone', 'satellite']
	transform_dict = {
		'drone': transforms.Compose(transform_drone_list),
		'satellite': transforms.Compose(transform_satellite_list)
	}
	return University1652(root, mode, sample_num, view_types, transform_dict)


# Load data for evaluation
def University1652_test_dataset(args, view_types, img_size, mode='test'):
	data_transforms = transforms.Compose([
			transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	
	root = args.root
	sample_num = 0
	transform_dict = {}
	for view in view_types:
		transform_dict[view] = data_transforms

	return University1652(root, mode, sample_num, view_types, transform_dict)