import os
import glob
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import torch
from utils.mask_util import mask_image
from utils.image_extract import ImageExtraction,get_default_angles

class Sun360PreExtractDataset(data.Dataset):
	def __init__(self, image_dir,train=True,rotation_rate=3,extract=False):
		super(Sun360PreExtractDataset, self).__init__()
		path = os.path.join(image_dir,"train" if train else "test")
		self.image_filenames = glob.glob(path+"/*.jpg")
		self.angles = get_default_angles()
		self.extract = ImageExtraction(view_angle=60) if extract else None
		self.rotation_rate = rotation_rate

	def __getitem__(self, index):
		file_path = self.image_filenames[index//self.rotation_rate]
		img = np.array(Image.open(file_path)).astype(np.float32) / 127.5 - 1
		roll_degree = int((index%self.rotation_rate)*(img.shape[1]/self.rotation_rate))

		img = np.roll(img,roll_degree,1)
		if self.extract:
			img = self.extract(torch.from_numpy(img).permute(2,0,1),)
		else:
			img = torch.from_numpy(cv2.resize(img,(512,256))).permute(2,0,1)
		return img,os.path.basename(file_path),roll_degree

	def __len__(self):
		return len(self.image_filenames)*self.rotation_rate

class FirstStageDataset(data.Dataset):
	def __init__(self, npz_paths, image_dir, train=True):
		super(FirstStageDataset, self).__init__()
		self.pickle_filenames = sorted([x for x in glob.glob(npz_paths+"/*.npz")])
		self.image_dir = image_dir
		self.train = train
		print("dataset size:",len(self.pickle_filenames))

	def __getitem__(self, index):
		data = np.load(self.pickle_filenames[index])
		path = os.path.join(self.image_dir,"train",os.path.basename(data["path"][0]))
		if not os.path.exists(path):
			path = os.path.join(self.image_dir,"test",os.path.basename(data["path"][0]))
		r_img = Image.open(path).resize((512,256))
		if self.train:
			mask = mask_image(r_img)
		else:
			mask = np.array(Image.open("mask_img.png").convert("RGB")) / 255.
		r_img = np.array(r_img)/255.
		r_img = np.roll(r_img, data["degree"][0]//2, 1)
		img_masked = mask * r_img
		img_data = np.concatenate([img_masked,mask.mean(axis=-1,keepdims=True)],axis=-1).transpose(2,0,1).astype(np.float32)
		return data["data"],img_data

	def __len__(self):
		return len(self.pickle_filenames)

class SecondStageDataset(data.Dataset):
	def __init__(self,npz_paths, image_dir, real_image_path,train=True):
		super(SecondStageDataset, self).__init__()
		self.pickle_filenames = sorted([x for x in glob.glob(npz_paths+"/*.npz")])
		self.image_dir = image_dir
		self.real_image_dir = real_image_path
		self.train = train
		print("dataset size:",len(self.pickle_filenames))


	def __getitem__(self, index):
		data = np.load(self.pickle_filenames[index])
		datas = data["data"]

		path = os.path.join(self.image_dir,data["path"][0])
		img = np.array(Image.open(path)).astype(np.float32) / 255
		img = cv2.resize(img,(512,256))
		img = np.roll(img, data["degree"][0]//2, 1)
		img = img.transpose(2,0,1)

		path = os.path.join(self.real_image_dir,"train" if self.train else "test",data["path"][0])
		r_img = np.array(Image.open(path)).astype(np.float32) / 255
		r_img = cv2.resize(r_img,(512,256))
		r_img = np.roll(r_img, data["degree"][0]//2, 1)

		if self.train:
			mask = mask_image(height=256,width=512)
		else:
			mask = np.array(Image.open("mask_img.png").convert("RGB")) / 255.

		img_masked = mask * r_img
		img_data = np.concatenate([img_masked,mask.mean(axis=-1,keepdims=True)],axis=-1).transpose(2,0,1).astype(np.float32)
		return datas,img,img_data

	def __len__(self):
		return len(self.pickle_filenames)


if __name__ == '__main__':
	# set =Sun360OutdoorExtractedImageDataset("./datas","../sun360", train=True)
	# set =Sun360OutdoorQuantizeVQLowresImageDataset("./datas_all", train=True)
	set = SecondStageDataset("./datas","./vqvae_recon", train=True)
	print(len(set))
	for i in range(len(set)):
		a = set[i]
		# plt.imshow(a[0].permute(1,2,0).numpy()/2+0.5)
		# plt.show()
		# plt.imshow(aug(a[0]).squeeze(0).permute(1,2,0).numpy()/2+0.5)
		# plt.show()
		print(a[0].shape,a[1].shape,)
		# plt.imshow(a[3])
		# plt.show()
		# plt.imshow(a[1][10].permute(1,2,0))
		# plt.show()
		# plt.imshow(a[2].transpose(1,2,0))
		# plt.show()
		break