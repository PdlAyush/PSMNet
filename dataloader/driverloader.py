import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

class StereoDataset(data.Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.left_image_dir = os.path.join(data_root, 'left')
        self.right_image_dir = os.path.join(data_root, 'right')
        self.disparity_dir = os.path.join(data_root, 'disparity')

        
        self.image_list = []
        for name in os.listdir(self.left_image_dir):
            image_name = name.split('.')[0]
            left_image_path = os.path.join(self.left_image_dir, image_name + '.jpg')
            right_image_path = os.path.join(self.right_image_dir, image_name + '.jpg')
            disparity_path = os.path.join(self.disparity_dir, image_name + '.png')

            if os.path.exists(left_image_path) and os.path.exists(right_image_path) and os.path.exists(disparity_path):
                self.image_list.append(image_name)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        left_image_path = os.path.join(self.left_image_dir, image_name + '.jpg')
        right_image_path = os.path.join(self.right_image_dir, image_name + '.jpg')
        disparity_path = os.path.join(self.disparity_dir, image_name + '.png')

        try:
            left_image = Image.open(left_image_path).convert('RGB')
            right_image = Image.open(right_image_path).convert('RGB')
            disparity = Image.open(disparity_path).convert('L')

            if self.transform:
                left_image = self.transform(left_image)
                right_image = self.transform(right_image)
                disparity = self.transform(disparity)

            return left_image, right_image, disparity

        except FileNotFoundError:
            print(f"Data not found for image: {image_name}, skipping...")
            return None, None, None
