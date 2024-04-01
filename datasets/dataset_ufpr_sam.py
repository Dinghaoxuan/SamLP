import os
from typing import Any
import numpy as np
import cv2
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import random
from einops import repeat
from icecream import ic
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
from torchvision.transforms.functional import resize, to_pil_image, rotate, hflip, vflip  # type: ignore
from torch.nn import functional as F


def random_rot_flip_torch(image, label):
    k = np.random.randint(0, 4)
    image = rotate(image, k*90)
    label = rotate(label, k*90)
    axis = np.random.randint(0, 2)
    if axis == 0:
        image = hflip(image)
        label = hflip(label)
    elif axis == 1:
        image = vflip(image)
        label = vflip(label)
    return image, label

def random_rotate_torch(image, label):
    angle = np.random.randint(-20, 20)
    image = rotate(image, angle)
    label = rotate(label, angle)
    return image, label


class SamTransformTest:
    def __init__(self, target_length):
        self.target_length = target_length
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def get_low_res_mask(self, x):
        pass

    def pad_image(self, x):
        h, w = x.shape[-2:]
        padh = self.target_length - h
        padw = self.target_length - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def normalize_image(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        return x

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    
    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape BxHxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        # print(image_batch.shape)
        return np.array(resize(to_pil_image(image), target_size))

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = self.apply_image(image)
        label = self.apply_image(label)

        image_torch = torch.as_tensor(image).permute(2,0,1).contiguous()#[None, :, :, :]
        label_torch = torch.as_tensor(label).contiguous()

        image_torch = self.pad_image(self.normalize_image(image_torch))
        label_torch = self.pad_image(label_torch)[None, :, :]

        # if random.random() > 0.5:
        #     image_torch, label_torch = random_rot_flip_torch(image_torch, label_torch)
        # elif random.random() > 0.5:
        #     image_torch, label_torch = random_rotate_torch(image_torch, label_torch)

        low_res_label = resize(label_torch, self.target_length//4)#.squeeze()
        sample = {'image': image_torch, 'label': label_torch.float(), 'low_res_label': low_res_label.float()}
        return sample


    
class SamTransform:
    def __init__(self, target_length):
        self.target_length = target_length
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def get_low_res_mask(self, x):
        pass

    def pad_image(self, x):
        h, w = x.shape[-2:]
        padh = self.target_length - h
        padw = self.target_length - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def normalize_image(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        return x

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    
    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape BxHxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        # print(image_batch.shape)
        return np.array(resize(to_pil_image(image), target_size))

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = self.apply_image(image)
        label = self.apply_image(label)

        image_torch = torch.as_tensor(image).permute(2,0,1).contiguous()#[None, :, :, :]
        label_torch = torch.as_tensor(label).contiguous()

        image_torch = self.pad_image(self.normalize_image(image_torch))
        label_torch = self.pad_image(label_torch)[None, :, :]

        if random.random() > 0.5:
            image_torch, label_torch = random_rot_flip_torch(image_torch, label_torch)
        elif random.random() > 0.5:
            image_torch, label_torch = random_rotate_torch(image_torch, label_torch)

        low_res_label = resize(label_torch, self.target_length//4)#.squeeze()
        sample = {'image': image_torch, 'label': label_torch.float(), 'low_res_label': low_res_label.float()}
        return sample
    
def collater(data):
    images = [s['image'] for s in data]
    labels = [s['label'] for s in data]
    low_res_labels = [s['low_res_label'] for s in data]
    
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0).squeeze()
    low_res_labels = torch.stack(low_res_labels, dim=0).squeeze()

    return {'image': images, 'label': labels, 'low_res_label': low_res_labels}
    # pass


class UFPR_ALPR_Dataset(data.Dataset):
    def __init__(self, root, split='training', transform=None):
        self.data_dir = os.path.join(root, split)
        self.image_list = self.build_image_list()
        self.transform = transform

    def build_image_list(self):
        image_list = []
        for i in range(len(os.listdir(self.data_dir))):
            path = os.path.join(self.data_dir, os.listdir(self.data_dir)[i])
            files = os.listdir(path)
            for j in range(len(files)):
                if os.path.splitext(files[j])[-1] == '.png':
                    image_list.append(os.path.join(path, files[j]))
        # image_list = image_list[490:]
        return image_list

    def load_image(self, path):
        img = cv2.imread(path)
        img = img.astype(np.uint8)
        return img

    def load_annotations(self, path):
        file = path.replace('png', 'txt')
        with open(file, 'r') as f:
            data = f.read()

        lines = data.replace('\t', '').replace('-', '').split('\n')
        for line in lines:
            line_split = line.split(':')
            prop = line_split[0].strip()
            data = line_split[1].strip()
            if prop == "position_plate":
                data = data.split(" ")
                data = np.array(data, dtype=np.float32)
                label = data.reshape((1,4))

        return label

    def plate_mask(self, img, annot):
        h, w = img.shape[0], img.shape[1]
        mask = np.zeros((h, w))
        mask[int(annot[:,1]):int(annot[:,1]+annot[:,3]),int(annot[:,0]):int(annot[:,0]+annot[:,2])] = 1
        mask = mask.astype(np.uint8)
        return mask

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path = self.image_list[idx]
        img = self.load_image(path)
        plate_annot = self.load_annotations(path)
        mask = self.plate_mask(img, plate_annot)
        sample = {'image': img, 'label': mask}
        if self.transform:
            sample = self.transform(sample)
        
        return sample


if __name__=='__main__':
    # db_train = UFPR_ALPR_Dataset(root='/media/disk1/yxding/dhx/Dataset/UFPR-ALPR/', split="training",
    #                            transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[512, 512], low_res=[128, 128])]))
    
    db_train = UFPR_ALPR_Dataset(root='/media/disk1/yxding/dhx/Dataset/UFPR-ALPR/', split="training",
                               transform=SamTransform(1024))
    
    trainloader = data.DataLoader(db_train, batch_size=2, shuffle=True, collate_fn=collater, drop_last=True, num_workers=2)

    for v in trainloader:
        images = v['image']
        labels = v['label']
        low_res_labels = v['low_res_label']

        print(images.shape)
        print(labels.shape)
        print(low_res_labels.shape)
        raise
    
    # sample = db_train[10]
    # label = sample['label']
    # image = sample['image']
    # low_res_label = sample['low_res_label']

    # # print(label.shape)
    # # print(image.shape)

    # image = sample['image'].permute(1,2,0).numpy()
    # cv2.imwrite('test_image.png', image*100)

    # label = sample['label'].permute(1,2,0).numpy().astype(np.uint8)
    # label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite('test_label.png', label*100)

    # low_res_label = sample['low_res_label'].permute(1,2,0).numpy().astype(np.uint8)
    # low_res_label = cv2.cvtColor(low_res_label, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite('test_low_res_label.png', low_res_label*100)


    # label = sample['label'].numpy().astype(np.uint8)
    # image = sample['image'].permute(1,2,0).numpy()

    # cv2.imwrite('test_image.png', image*255)
    # label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite('test_label.png', label*255)

    # print(image.shape)
    # print(label.shape)



