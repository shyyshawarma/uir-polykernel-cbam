import os

import albumentations as A
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class NonRefDataReader(Dataset):
    def __init__(self, img_dir, inp='input', mode='train', ori=False, img_options=None):
        super(NonRefDataReader, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(img_dir, inp)))

        self.inp_filenames = [os.path.join(img_dir, inp, x) for x in inp_files if is_image_file(x)]

        self.mode = mode

        self.img_options = img_options

        self.sizex = len(self.inp_filenames)  # get the size of input

        if self.mode == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.Rotate(p=0.3),
                A.Transpose(p=0.3),
                A.RandomResizedCrop(
                size=(img_options['h'], img_options['w']),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
                interpolation=1,
                p=1.0
)

            ])
            self.degrade = A.Compose([
                A.NoOp()
            ])
        else:
            if ori:
                self.transform = A.Compose([
                    A.NoOp(),
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(height=img_options['h'], width=img_options['w']),
                ])
            self.degrade = A.Compose([
                A.NoOp(),
            ])

    def mixup(self, inp_img, mode='mixup'):
        mixup_index_ = random.randint(0, self.sizex - 1)

        _, transformed = self.load(mixup_index_)

        alpha = 0.2
        lam = np.random.beta(alpha, alpha)

        mixup_inp_img = F.to_tensor(self.degrade(image=transformed['image'])['image'])

        if mode == 'mixup':
            inp_img = lam * inp_img + (1 - lam) * mixup_inp_img
        else:
            img_h, img_w = self.img_options['h'], self.img_options['w']

            cx = np.random.uniform(0, img_w)
            cy = np.random.uniform(0, img_h)

            w = img_w * np.sqrt(1 - lam)
            h = img_h * np.sqrt(1 - lam)

            x0 = int(np.round(max(cx - w / 2, 0)))
            x1 = int(np.round(min(cx + w / 2, img_w)))
            y0 = int(np.round(max(cy - h / 2, 0)))
            y1 = int(np.round(min(cy + h / 2, img_h)))

            inp_img[:, y0:y1, x0:x1] = mixup_inp_img[:, y0:y1, x0:x1]

        return inp_img

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path, transformed = self.load(index_)

        if self.mode == 'train':
            inp_img = F.to_tensor(self.degrade(image=transformed['image'])['image'])
        else:
            inp_img = F.to_tensor(transformed['image'])

        if self.mode == 'train':
            if index_ > 0 and index_ % 3 == 0:
                if random.random() > 0.5:
                    inp_img = self.mixup(inp_img, mode='mixup')
                else:
                    inp_img = self.mixup(inp_img, mode='cutmix')

        filename = os.path.basename(inp_path)

        return inp_img, filename

    def load(self, index_):
        inp_path = self.inp_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')

        inp_img = np.array(inp_img)

        transformed = self.transform(image=inp_img)

        return inp_path, transformed


class DataReader(Dataset):
    def __init__(self, img_dir, inp='input', tar='target', mode='train', ori=False, img_options=None):
        super(DataReader, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(img_dir, inp)))
        tar_files = sorted(os.listdir(os.path.join(img_dir, tar)))

        self.inp_filenames = [os.path.join(img_dir, inp, x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(img_dir, tar, x) for x in tar_files if is_image_file(x)]

        self.mode = mode

        self.img_options = img_options

        self.sizex = len(self.tar_filenames)  # get the size of target

        if self.mode == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.Rotate(p=0.3),
                A.Transpose(p=0.3),
                A.RandomResizedCrop(
                size=(img_options['h'], img_options['w']),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
                interpolation=1,
                p=1.0
)

            ],
                additional_targets={
                    'target': 'image',
                }
            )
            self.degrade = A.Compose([
                A.NoOp()
            ])
        else:
            if ori:
                self.transform = A.Compose([
                    A.NoOp(),
                ],
                    additional_targets={
                        'target': 'image',
                    }
                )
            else:
                self.transform = A.Compose([
                    A.Resize(height=img_options['h'], width=img_options['w']),
                ],
                    additional_targets={
                        'target': 'image',
                    }
                )
            self.degrade = A.Compose([
                A.NoOp(),
            ])

    def mixup(self, inp_img, tar_img, mode='mixup'):
        mixup_index_ = random.randint(0, self.sizex - 1)

        _, transformed = self.load(mixup_index_)

        alpha = 0.2
        lam = np.random.beta(alpha, alpha)

        mixup_inp_img = F.to_tensor(self.degrade(image=transformed['image'])['image'])
        mixup_tar_img = F.to_tensor(transformed['target'])

        if mode == 'mixup':
            inp_img = lam * inp_img + (1 - lam) * mixup_inp_img
            tar_img = lam * tar_img + (1 - lam) * mixup_tar_img
        else:
            img_h, img_w = self.img_options['h'], self.img_options['w']

            cx = np.random.uniform(0, img_w)
            cy = np.random.uniform(0, img_h)

            w = img_w * np.sqrt(1 - lam)
            h = img_h * np.sqrt(1 - lam)

            x0 = int(np.round(max(cx - w / 2, 0)))
            x1 = int(np.round(min(cx + w / 2, img_w)))
            y0 = int(np.round(max(cy - h / 2, 0)))
            y1 = int(np.round(min(cy + h / 2, img_h)))

            inp_img[:, y0:y1, x0:x1] = mixup_inp_img[:, y0:y1, x0:x1]
            tar_img[:, y0:y1, x0:x1] = mixup_tar_img[:, y0:y1, x0:x1]

        return inp_img, tar_img

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        tar_path, transformed = self.load(index_)
        
        if self.mode == 'train':
            inp_img = F.to_tensor(self.degrade(image=transformed['image'])['image'])
        else:
            inp_img = F.to_tensor(transformed['image'])
        tar_img = F.to_tensor(transformed['target'])

        filename = os.path.basename(tar_path)

        return inp_img, tar_img, filename

    def get_labels(self):
        labels = []
        for filename in self.inp_filenames:
            label = None
            if 'uieb' in filename:
                label = 0
            elif 'euvp' in filename:
                label = 1
            elif 'lsui' in filename:
                label = 2

            labels.append(label)

        return labels

    def load(self, index_):
        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        inp_img = np.array(inp_img)
        tar_img = np.array(tar_img)

        transformed = self.transform(image=inp_img, target=tar_img)

        return tar_path, transformed