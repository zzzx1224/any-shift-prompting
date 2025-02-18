import numpy as np 
import os
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
# from utils import get_transform
import pdb
import random
import torch
import time
import cv2

# data_path = '../kfold/'
data_path = '../224kfold/'

class PACS(Dataset):
	def __init__(self, test_domain, num_domains=3, transform=None):
		# assert phase in ['train', 'val', 'test']
		self.domain_list = ['art_painting', 'photo', 'cartoon', 'sketch']
		self.domain_list.remove(test_domain)
		self.num_domains = num_domains
		assert self.num_domains <= len(self.domain_list)

		self.train_img_list = []
		self.train_label_list = []

		# self.num_imgs = []
		for i in range(len(self.domain_list)):
			f = open('../files/' + self.domain_list[i] + '_train_kfold.txt', 'r')
			lines = f.readlines()
			train_domain_imgs = []
			train_domain_labels = []
			# domain_imgs = {}
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				train_domain_imgs.append(data_path + img)
				train_domain_labels.append(int(label)-1)
			self.train_img_list.append(train_domain_imgs)
			self.train_label_list.append(train_domain_labels)
			# self.num_imgs.append(len(train_domain_imgs))
		# pdb.set_trace()

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []
		# self.transform = transform
		# self.meta_test_domain = np.random.randint(len(self.domain_list))

		seed = 777

		# elif phase == 'val':
		self.domain_list.append(test_domain)
		# pdb.set_trace()
		for i in range(len(self.domain_list)):
			f = open('../files/' + self.domain_list[i] + '_crossval_kfold.txt', 'r')
			lines = f.readlines()

			val_domain_imgs = []
			val_domain_labels = []

			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				# self.val_img_list.append(data_path + img)
				# self.val_label_list.append(int(label)-1)
				val_domain_imgs.append(data_path + img)
				val_domain_labels.append(int(label)-1)
			np.random.seed(seed)
			np.random.shuffle(val_domain_imgs)
			np.random.seed(seed)
			np.random.shuffle(val_domain_labels)
			self.val_img_list.append(val_domain_imgs)
			self.val_label_list.append(val_domain_labels)
		self.domain_list.remove(test_domain)


		# else:
		f = open('../files/' + test_domain + '_test_kfold.txt', 'r')
		lines = f.readlines()
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			self.test_img_list.append(data_path + img)
			self.test_label_list.append(int(label)-1)

		# seed = 777
		# pdb.set_trace()
		np.random.seed(seed)
		np.random.shuffle(self.test_img_list)
		np.random.seed(seed)
		np.random.shuffle(self.test_label_list)

		# pdb.set_trace()

	def reset(self, phase, domain_id, transform=None):
		# pdb.set_trace()
		self.phase = phase
		if phase == 'train':
			self.transform = transform
			self.img_list = self.train_img_list[domain_id]
			self.label_list = self.train_label_list[domain_id]
			# pdb.set_trace()

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list[domain_id]
			self.label_list = self.val_label_list[domain_id]

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list
			# pdb.set_trace()

		# pdb.set_trace()
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		# image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		if self.transform is not None:
			image = self.transform(image)

		label = self.label_list[item]
		# return image and label
		# return image, self.label_list[item]
		return image, label

	def __len__(self):
		return len(self.img_list)

class randPACS(Dataset):
	def __init__(self, test_domain, num_domains=3, transform=None):
		# assert phase in ['train', 'val', 'test']
		self.domain_list = ['art_painting', 'photo', 'cartoon', 'sketch']
		self.domain_list.remove(test_domain)
		self.num_domains = num_domains
		assert self.num_domains <= len(self.domain_list)

		self.train_img_list = []
		self.train_label_list = []

		# self.num_imgs = []
		for i in range(len(self.domain_list)):
			# f = open('../files/random_group_' + test_domain + '_' + str(i) +'.txt', 'r')
			f = open('../files/cluster_z2_' + test_domain + '_' + str(i) +'.txt', 'r')
			lines = f.readlines()
			train_domain_imgs = []
			train_domain_labels = []
			# domain_imgs = {}
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				train_domain_imgs.append(data_path + img)
				# train_domain_labels.append(int(label)-1)
				train_domain_labels.append(int(label))
			self.train_img_list.append(train_domain_imgs)
			self.train_label_list.append(train_domain_labels)
			# self.num_imgs.append(len(train_domain_imgs))
		# pdb.set_trace()

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []
		# self.transform = transform
		# self.meta_test_domain = np.random.randint(len(self.domain_list))

		seed = 777

		# elif phase == 'val':
		self.domain_list.append(test_domain)
		# pdb.set_trace()
		for i in range(len(self.domain_list)):
			f = open('../files/' + self.domain_list[i] + '_crossval_kfold.txt', 'r')
			lines = f.readlines()

			val_domain_imgs = []
			val_domain_labels = []

			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				# self.val_img_list.append(data_path + img)
				# self.val_label_list.append(int(label)-1)
				val_domain_imgs.append(data_path + img)
				val_domain_labels.append(int(label)-1)
			np.random.seed(seed)
			np.random.shuffle(val_domain_imgs)
			np.random.seed(seed)
			np.random.shuffle(val_domain_labels)
			self.val_img_list.append(val_domain_imgs)
			self.val_label_list.append(val_domain_labels)
		self.domain_list.remove(test_domain)


		# else:
		f = open('../files/' + test_domain + '_test_kfold.txt', 'r')
		lines = f.readlines()
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			self.test_img_list.append(data_path + img)
			self.test_label_list.append(int(label)-1)

		# seed = 777
		# pdb.set_trace()
		np.random.seed(seed)
		np.random.shuffle(self.test_img_list)
		np.random.seed(seed)
		np.random.shuffle(self.test_label_list)

		# pdb.set_trace()

	def reset(self, phase, domain_id, transform=None):
		# pdb.set_trace()
		self.phase = phase
		if phase == 'train':
			self.transform = transform
			self.img_list = self.train_img_list[domain_id]
			self.label_list = self.train_label_list[domain_id]
			# pdb.set_trace()

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list[domain_id]
			self.label_list = self.val_label_list[domain_id]

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list

		# pdb.set_trace()
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		# image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		if self.transform is not None:
			image = self.transform(image)

		label = self.label_list[item]
		# return image and label
		# return image, self.label_list[item]
		return image, label

	def __len__(self):
		return len(self.img_list)


class OfficeHome(Dataset):
	def __init__(self, test_domain, num_domains=3, transform=None):
		# assert phase in ['train', 'val', 'test']
		self.domain_list = ['art', 'clipart', 'product', 'real_World']
		self.domain_list.remove(test_domain)
		self.num_domains = num_domains
		assert self.num_domains <= len(self.domain_list)

		self.train_img_list = []
		self.train_label_list = []

		# self.num_imgs = []
		for i in range(len(self.domain_list)):
			f = open('../office-home_files/' + self.domain_list[i] + '_train.txt', 'r')
			lines = f.readlines()
			train_domain_imgs = []
			train_domain_labels = []
			# domain_imgs = {}
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				train_domain_imgs.append(img)
				train_domain_labels.append(int(label))
			self.train_img_list.append(train_domain_imgs)
			self.train_label_list.append(train_domain_labels)
			# self.num_imgs.append(len(train_domain_imgs))
		# pdb.set_trace()

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []
		# self.transform = transform
		# self.meta_test_domain = np.random.randint(len(self.domain_list))

		seed = 777

		# else:
		f = open('../office-home_files/' + test_domain + '_train.txt', 'r')
		lines = f.readlines()
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			self.test_img_list.append(img)
			self.test_label_list.append(int(label))

		# seed = 777
		# pdb.set_trace()
		np.random.seed(seed)
		np.random.shuffle(self.test_img_list)
		np.random.seed(seed)
		np.random.shuffle(self.test_label_list)

		# pdb.set_trace()

	def reset(self, phase, domain_id, transform=None):
		# pdb.set_trace()
		self.phase = phase
		if phase == 'train':
			self.transform = transform
			self.img_list = self.train_img_list[domain_id]
			self.label_list = self.train_label_list[domain_id]
			# pdb.set_trace()

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list[domain_id]
			self.label_list = self.val_label_list[domain_id]

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list

		# pdb.set_trace()
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		if self.transform is not None:
			image = self.transform(image)

		label = self.label_list[item]
		# return image and label
		# return image, self.label_list[item]
		return image, label

	def __len__(self):
		return len(self.img_list)


class VLCS(Dataset):
	def __init__(self, test_domain, num_domains=3, transform=None):
		# assert phase in ['train', 'val', 'test']
		self.domain_list = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
		self.domain_list.remove(test_domain)
		self.num_domains = num_domains
		assert self.num_domains <= len(self.domain_list)

		self.train_img_list = []
		self.train_label_list = []

		# self.num_imgs = []
		for i in range(len(self.domain_list)):
			f = open('../vlcs_files/' + self.domain_list[i] + '_train.txt', 'r')
			lines = f.readlines()
			train_domain_imgs = []
			train_domain_labels = []
			# domain_imgs = {}
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				train_domain_imgs.append(data_path + img)
				train_domain_labels.append(int(label)-1)
			self.train_img_list.append(train_domain_imgs)
			self.train_label_list.append(train_domain_labels)
			# self.num_imgs.append(len(train_domain_imgs))
		# pdb.set_trace()

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []
		# self.transform = transform
		# self.meta_test_domain = np.random.randint(len(self.domain_list))

		seed = 777

		# elif phase == 'val':
		self.domain_list.append(test_domain)
		# pdb.set_trace()
		for i in range(len(self.domain_list)):
			f = open('../vlcs_files/' + self.domain_list[i] + '_crossval.txt', 'r')
			lines = f.readlines()

			val_domain_imgs = []
			val_domain_labels = []

			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				# self.val_img_list.append(data_path + img)
				# self.val_label_list.append(int(label)-1)
				val_domain_imgs.append(data_path + img)
				val_domain_labels.append(int(label)-1)
			np.random.seed(seed)
			np.random.shuffle(val_domain_imgs)
			np.random.seed(seed)
			np.random.shuffle(val_domain_labels)
			self.val_img_list.append(val_domain_imgs)
			self.val_label_list.append(val_domain_labels)
		self.domain_list.remove(test_domain)


		# else:
		f = open('../vlcs_files/' + test_domain + '_test.txt', 'r')
		lines = f.readlines()
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			self.test_img_list.append(data_path + img)
			self.test_label_list.append(int(label)-1)

		# seed = 777
		# pdb.set_trace()
		np.random.seed(seed)
		np.random.shuffle(self.test_img_list)
		np.random.seed(seed)
		np.random.shuffle(self.test_label_list)

		# pdb.set_trace()

	def reset(self, phase, domain_id, transform=None):
		# pdb.set_trace()
		self.phase = phase
		if phase == 'train':
			self.transform = transform
			self.img_list = self.train_img_list[domain_id]
			self.label_list = self.train_label_list[domain_id]
			# pdb.set_trace()

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list[domain_id]
			self.label_list = self.val_label_list[domain_id]

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list

		# pdb.set_trace()
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		# image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		if self.transform is not None:
			image = self.transform(image)

		label = self.label_list[item]
		# return image and label
		# return image, self.label_list[item]
		return image, label

	def __len__(self):
		return len(self.img_list)


class DomainNet(Dataset):
	def __init__(self, test_domain, num_domains=5, transform=None):
		# assert phase in ['train', 'val', 'test']
		self.domain_list = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
		self.domain_list.remove(test_domain)
		self.num_domains = num_domains
		assert self.num_domains <= len(self.domain_list)

		data_path = '../domainnet/'
		# data_path = '../domain224/'

		self.train_img_list = []
		self.train_label_list = []

		# self.num_imgs = []
		for i in range(len(self.domain_list)):
			f = open('../domainnet/txtfiles/' + self.domain_list[i] + '_train.txt', 'r')
			lines = f.readlines()
			train_domain_imgs = []
			train_domain_labels = []
			# domain_imgs = {}
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				train_domain_imgs.append(data_path + img)
				train_domain_labels.append(int(label))
			self.train_img_list.append(train_domain_imgs)
			self.train_label_list.append(train_domain_labels)
			# self.num_imgs.append(len(train_domain_imgs))
		# pdb.set_trace()

		# for i in range(len(self.domain_list)):
		# 	f = open('../domainnet/txtfiles/' + self.domain_list[i] + '_train.txt', 'r')
		# 	lines = f.readlines()
		# 	train_domain_imgs = {}
		# 	# print(self.domain_list[i])
		# 	for i in range(345):
		# 		train_domain_imgs[i] = []
		# 	# train_domain_labels = []
		# 	# domain_imgs = {}
		# 	for line in lines:
		# 		[img, label] = line.strip('\n').split(' ')
		# 		train_domain_imgs[int(label)].append(data_path + img)
		# 		# train_domain_imgs.append(data_path + img)
		# 		# train_domain_labels.append(int(label)-1)
		# 	self.train_img_list.append(train_domain_imgs)
		# 	# self.train_label_list.append(train_domain_labels)
		# 	# self.num_imgs.append(len(train_domain_imgs))
		# # pdb.set_trace()

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []
		# self.transform = transform
		# self.meta_test_domain = np.random.randint(len(self.domain_list))

		seed = 777

		# elif phase == 'val':
		self.domain_list.append(test_domain)
		# pdb.set_trace()
		for i in range(len(self.domain_list)):
			f = open('../domainnet/txtfiles/' + self.domain_list[i] + '_test.txt', 'r')
			lines = f.readlines()

			val_domain_imgs = []
			val_domain_labels = []

			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				# self.val_img_list.append(data_path + img)
				# self.val_label_list.append(int(label)-1)
				val_domain_imgs.append(data_path + img)
				val_domain_labels.append(int(label))
			np.random.seed(seed)
			np.random.shuffle(val_domain_imgs)
			np.random.seed(seed)
			np.random.shuffle(val_domain_labels)
			self.val_img_list.append(val_domain_imgs)
			self.val_label_list.append(val_domain_labels)
		self.domain_list.remove(test_domain)


		# else:
		f = open('../domainnet/txtfiles/' + test_domain + '_test.txt', 'r')
		lines = f.readlines()
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			self.test_img_list.append(data_path + img)
			self.test_label_list.append(int(label))

		# seed = 777
		# pdb.set_trace()
		np.random.seed(seed)
		np.random.shuffle(self.test_img_list)
		np.random.seed(seed)
		np.random.shuffle(self.test_label_list)

		# pdb.set_trace()

	# def reset(self, phase, domain_id, categories, transform=None):
	def reset(self, phase, domain_id, transform=None):
		# pdb.set_trace()
		self.phase = phase
		if phase == 'train':
			self.transform = transform
			self.img_list = []
			self.label_list = []
			# for cate in categories:
			# 	self.img_list += self.train_img_list[domain_id][cate]
			# 	self.label_list += [cate] * len(self.train_img_list[domain_id][cate])

			self.img_list = self.train_img_list[domain_id]
			self.label_list = self.label_list = self.train_label_list[domain_id]
			# self.label_list = self.train_label_list[domain_id]
			# pdb.set_trace()

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list[domain_id]
			self.label_list = self.val_label_list[domain_id]

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list

		# pdb.set_trace()
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		# image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		if self.transform is not None:
			image = self.transform(image)

		label = self.label_list[item]
		# return image and label
		# return image, self.label_list[item]
		return image, label

	def __len__(self):
		return len(self.img_list)


class FashionMNIST(Dataset):
	def __init__(self, train_domain, test_domain, transform=None):
		# assert phase in ['train', 'val', 'test']
		# self.domain_list = ['0', '15', '30', '45', '60', '75', '90']
		# for te_dom in test_domain:
		# 	self.domain_list.remove(te_dom)
		self.domain_list = train_domain

		self.train_domain_imgs = []
		self.train_domain_labels = []

		data_path = '../fashion_MNIST/'

		# for i in range(len(self.domain_list)):
		f = open('../fashion_MNIST/' + self.domain_list[-1] + '_train_imgs.txt', 'r')
		lines = f.readlines()
		all_training_imgs = []
		all_training_labels = []
		
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			all_training_imgs.append(img[2:])
			all_training_labels.append(int(label))

		seed = 777
		np.random.seed(seed)
		np.random.shuffle(all_training_imgs)
		np.random.seed(seed)
		np.random.shuffle(all_training_labels)

		training_imgs = all_training_imgs[:10000]
		training_labels = all_training_labels[:10000]

		# pdb.set_trace()

		for i in range(len(self.domain_list)):
			domain_imgs = []
			domain_labels = []
			for j in range(10000):
				domain_imgs.append(data_path + self.domain_list[i] + training_imgs[j][2:])
				domain_labels.append(training_labels[j])

			self.train_domain_imgs.append(domain_imgs)
			self.train_domain_labels.append(domain_labels)

		# pdb.set_trace()

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []
		# self.transform = transform
		# self.meta_test_domain = np.random.randint(len(self.domain_list))

		# elif phase == 'val':
		for i in range(len(self.domain_list)):
			self.domain_imgs = []
			self.domain_labels = []
			f = open('../fashion_MNIST/' + self.domain_list[i] + '_test_imgs.txt', 'r')
			lines = f.readlines()
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				self.domain_imgs.append(data_path + img)
				self.domain_labels.append(int(label))
			self.val_img_list.append(self.domain_imgs)
			self.val_label_list.append(self.domain_labels)


		# else:
		for i in range(len(test_domain)):
			f = open('../fashion_MNIST/' + test_domain[i] + '_test_imgs.txt', 'r')
			lines = f.readlines()
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				self.test_img_list.append(data_path + img)
				self.test_label_list.append(int(label))
		# pdb.set_trace()

	def reset(self, phase, domain_id, transform=None):
		# pdb.set_trace()
		if phase == 'train':
			self.transform = transform
			self.img_list = self.train_domain_imgs[domain_id]
			self.label_list = self.train_domain_labels[domain_id]

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list[domain_id]
			self.label_list = self.val_label_list[domain_id]

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list

		# pdb.set_trace()

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item])#.convert('RGB')  # (C, H, W)
		image = image.resize((28, 28))
		# image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		# image = image.resize((224, 224))
		# pdb.set_trace()
		if self.transform is not None:
			image = self.transform(image)

		# return image and label
		return image, self.label_list[item]#, self.meta_train_imgs, self.meta_train_labels

	def __len__(self):
		return len(self.img_list)

class MNIST(Dataset):
	def __init__(self, train_domain, test_domain, transform=None):
		# assert phase in ['train', 'val', 'test']
		# self.domain_list = ['0', '15', '30', '45', '60', '75', '90']
		# for te_dom in test_domain:
		# 	self.domain_list.remove(te_dom)
		self.domain_list = train_domain

		self.train_domain_imgs = []
		self.train_domain_labels = []

		data_path = '../mnist/'

		# for i in range(len(self.domain_list)):
		f = open('../mnist/' + self.domain_list[-1] + '_train_imgs.txt', 'r')
		lines = f.readlines()
		all_training_imgs = []
		all_training_labels = []
		
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			all_training_imgs.append(img[2:])
			all_training_labels.append(int(label))

		seed = 777
		np.random.seed(seed)
		np.random.shuffle(all_training_imgs)
		np.random.seed(seed)
		np.random.shuffle(all_training_labels)

		training_imgs = all_training_imgs[:10000]
		training_labels = all_training_labels[:10000]

		# pdb.set_trace()

		for i in range(len(self.domain_list)):
			domain_imgs = []
			domain_labels = []
			for j in range(10000):
				domain_imgs.append(data_path + self.domain_list[i] + training_imgs[j][2:])
				domain_labels.append(training_labels[j])

			self.train_domain_imgs.append(domain_imgs)
			self.train_domain_labels.append(domain_labels)

		# pdb.set_trace()

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []
		# self.transform = transform
		# self.meta_test_domain = np.random.randint(len(self.domain_list))

		# elif phase == 'val':
		for i in range(len(self.domain_list)):
			self.domain_imgs = []
			self.domain_labels = []
			f = open('../mnist/' + self.domain_list[i] + '_test_imgs.txt', 'r')
			lines = f.readlines()
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				self.domain_imgs.append(data_path + img)
				self.domain_labels.append(int(label))
			self.val_img_list.append(self.domain_imgs)
			self.val_label_list.append(self.domain_labels)


		# else:
		for i in range(len(test_domain)):
			f = open('../mnist/' + test_domain[i] + '_test_imgs.txt', 'r')
			lines = f.readlines()
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				self.test_img_list.append(data_path + img)
				self.test_label_list.append(int(label))
		# pdb.set_trace()

	def reset(self, phase, domain_id, transform=None):
		# pdb.set_trace()
		if phase == 'train':
			self.transform = transform
			self.img_list = self.train_domain_imgs[domain_id]
			self.label_list = self.train_domain_labels[domain_id]

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list[domain_id]
			self.label_list = self.val_label_list[domain_id]

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list

		# pdb.set_trace()

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item])#.convert('RGB')  # (C, H, W)
		image = image.resize((28, 28))
		# image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		# image = image.resize((224, 224))
		# pdb.set_trace()
		if self.transform is not None:
			image = self.transform(image)

		# return image and label
		return image, self.label_list[item]#, self.meta_train_imgs, self.meta_train_labels

	def __len__(self):
		return len(self.img_list)


class SVHN(Dataset):
	def __init__(self, train_domain, test_domain, transform=None):
		# assert phase in ['train', 'val', 'test']
		if train_domain == 'rand':
			self.domain_list = ['0', '1', '2']
			filename = 'random_group__'
		elif train_domain == 'chan':
			self.domain_list = ['0', '1', '2']
			filename = 'chan_group__'
		elif train_domain == 'dchan':
			self.domain_list = ['0', '1', '2']
			filename = 'dchan_group__'
		elif train_domain == '3chan':
			self.domain_list = ['0', '1', '2']
			filename = '3chan_group__'
		elif train_domain == 'new':
			self.domain_list = ['0', '1', '2']
			filename = 'new_group_'
		elif train_domain == '3clus':
			self.domain_list = ['0', '1', '2']
			filename = '3clus_group_'
		elif train_domain == '4clus':
			self.domain_list = ['0', '1', '2', '3']
			filename = '4clus_group_'
		elif train_domain == 'rota':
			self.domain_list = ['0', '30', '60']
			filename = 'rota_group__'

		self.train_domain_imgs = []
		self.train_domain_labels = []

		data_path = '../mnist/'

		# pdb.set_trace()

		for i in range(len(self.domain_list)):
			f = open('../svhn/' + filename + self.domain_list[i] + '.txt', 'r')
			lines = f.readlines()
			domain_imgs = []
			domain_labels = []
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				domain_imgs.append('../svhn/' + img)
				domain_labels.append(int(label))

			self.train_domain_imgs.append(domain_imgs)
			self.train_domain_labels.append(domain_labels)

		# pdb.set_trace()

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []
		# self.transform = transform
		# self.meta_test_domain = np.random.randint(len(self.domain_list))

		# elif phase == 'val':
		# for i in range(len(self.domain_list)):
		# 	self.domain_imgs = []
		# 	self.domain_labels = []
		# 	f = open('../mnist/' + '0_test_imgs.txt', 'r')
		# 	lines = f.readlines()
		# 	for line in lines:
		# 		[img, label] = line.strip('\n').split(' ')
		# 		self.domain_imgs.append(data_path + img)
		# 		self.domain_labels.append(int(label))
		# 	self.val_img_list.append(self.domain_imgs)
		# 	self.val_label_list.append(self.domain_labels)


		# else:
		# for i in range(len(test_domain)):
		f = open('../mnist/' + '0_test_imgs.txt', 'r')
		lines = f.readlines()
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			self.test_img_list.append(data_path + img)
			self.test_label_list.append(int(label))
		# pdb.set_trace()
		self.val_img_list = self.test_img_list
		self.val_label_list = self.test_label_list

	def reset(self, phase, domain_id, transform=None):
		self.phase = phase
		# pdb.set_trace()
		if phase == 'train':
			self.transform = transform
			self.img_list = self.train_domain_imgs[domain_id]
			self.label_list = self.train_domain_labels[domain_id]

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list[domain_id]
			self.label_list = self.val_label_list[domain_id]

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list

		# pdb.set_trace()

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		if self.phase=='train':
			image = image.resize((32, 32))
		else:
			image = image.resize((28, 28))
		# image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		# image = image.resize((224, 224))
		# pdb.set_trace()
		if self.transform is not None:
			image = self.transform(image)

		# return image and label
		return image, self.label_list[item]#, self.meta_train_imgs, self.meta_train_labels

	def __len__(self):
		return len(self.img_list)


from robustness.tools import folder
from robustness.tools.breeds_helpers import (
    make_entity13,
    make_entity30,
    make_living17,
    make_nonliving26,
)
from robustness.tools.helpers import get_label_mapping
from torchvision.datasets import ImageFolder as torch_ImageFolder
from data_utils import *
class Subset(Dataset):
	"""
    Subset of a dataset at specified indices.

    Arguments:
            dataset (Dataset): The whole Dataset
            indices (sequence): Indices in the whole set selected for subset
    """

	def __init__(self, dataset, indices, transform=None):
		self.dataset = dataset
		self.indices = indices
		self.transform = transform

	def __getitem__(self, idx):
		# logger.debug(f"IDx recieved {idx}")
		# logger.debug(f"Indices type {type(self.indices[idx])} value {self.indices[idx]}")
		x = self.dataset[self.indices[idx]]

		if self.transform is not None:
			transformed_img = self.transform(x[0])

			return transformed_img, x[1], x[2:]

		else:
			return x

	@property
	def y_array(self):
		return self.dataset.y_array[self.indices]

	def __len__(self):
		return len(self.indices)

def get_breeds(
    dataset=None,
    source=True,
    target=True,
    root_dir=None,
    target_split=1,
    transforms=None,
    num_classes=None,
    split_fraction=0.2,
    seed=42,
):
	root_dir = f"{root_dir}/imagenet/"

	if dataset == "living17":
		ret = make_living17(f"{root_dir}/imagenet_class_hierarchy/modified/", split="good")
	elif dataset == "entity13":
		ret = make_entity13(f"{root_dir}/imagenet_class_hierarchy/modified/", split="good")
	elif dataset == "entity30":
		ret = make_entity30(f"{root_dir}/imagenet_class_hierarchy/modified/", split="good")
	elif dataset == "nonliving26":
		ret = make_nonliving26(f"{root_dir}/imagenet_class_hierarchy/modified/", split="good")

	ImageFolder = dataset_with_targets(folder.ImageFolder)

	source_label_mapping = get_label_mapping("custom_imagenet", ret[1][0])
	target_label_mapping = get_label_mapping("custom_imagenet", ret[1][1])
	# pdb.set_trace()
	if source or (target and target_split == 0):
		sourceset = ImageFolder(
            f"{root_dir}/images/train/", label_mapping=source_label_mapping
        )

		source_idx, target_idx = split_idx(
            sourceset.y_array, num_classes, source_frac=0.8, seed=seed
        )

		source_trainset = Subset(
            sourceset, source_idx, transform=transforms
        )
		source_testset = ImageFolder(
            f"{root_dir}/images/val/",
            label_mapping=source_label_mapping,
            transform=transforms,
        )
		# logger.debug(
        #     f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}"
        # )

	if target:
		if target_split == 0:
			target_train_idx, target_test_idx = split_idx(
                sourceset.y_array[target_idx],
                num_classes,
                source_frac=split_fraction,
                seed=seed,
            )

			target_train_idx, target_test_idx = (
                target_idx[target_train_idx],
                target_idx[target_test_idx],
            )

			target_trainset = Subset(
                sourceset, target_train_idx, transform=transforms
            )
			target_testset = Subset(
                sourceset, target_test_idx, transform=transforms
            )

		elif target_split == 1:
			targetset = ImageFolder(
				f"{root_dir}/images/train/", label_mapping=target_label_mapping
			)

			target_train_idx, target_test_idx = split_idx(
				targetset.y_array, num_classes, source_frac=split_fraction, seed=seed
			)

			target_trainset = Subset(
				targetset, target_train_idx, transform=transforms
			)
			target_testset = Subset(
				targetset, target_test_idx, transform=transforms
			)

		elif target_split == 2:
			targetset = ImageFolder(
				f"{root_dir}/imagenetv2/imagenetv2-matched-frequency-format-val",
				label_mapping=source_label_mapping,
			)

			target_train_idx, target_test_idx = split_idx(
				targetset.y_array, num_classes, source_frac=split_fraction, seed=seed
			)

			target_trainset = Subset(
				targetset, target_train_idx, transform=transforms
			)
			target_testset = Subset(
				targetset, target_test_idx, transform=transforms
			)

		elif target_split == 3:
			targetset = ImageFolder(
				f"{root_dir}/imagenetv2/imagenetv2-matched-frequency-format-val",
				label_mapping=target_label_mapping,
			)

			target_train_idx, target_test_idx = split_idx(
				targetset.y_array, num_classes, source_frac=split_fraction, seed=seed
			)

			target_trainset = Subset(
				targetset, target_train_idx, transform=transforms
			)
			target_testset = Subset(
				targetset, target_test_idx, transform=transforms
			)

		else:
			raise ValueError("target_split must be between 0 and 3 for BREEDs dataset")
	dataset = {}

	if source and target:
		dataset["source_train"] = source_trainset
		dataset["source_test"] = source_testset
		dataset["target_train"] = target_trainset
		dataset["target_test"] = target_testset

	elif source:
		dataset["source_train"] = source_trainset
		dataset["source_test"] = source_testset

	elif target:
		dataset["target_train"] = target_trainset
		dataset["target_test"] = target_testset

	return dataset


def dataset_with_targets(cls):
	"""
    Modifies the dataset class to return target
    """

	def y_array(self):
		return np.array(self.targets).astype(int)

	return type(cls.__name__, (cls,), {"y_array": property(y_array)})