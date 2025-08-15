from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

# # 定义自定义数据集类
# class ImageDataset(Dataset):
#     def __init__(self, image_folder, transform=None):
#         self.image_folder = image_folder
#         self.transform = transform
#         self.image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         image_path = self.image_files[idx]
#         image = Image.open(image_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         return image, image

class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = []
        # 递归遍历文件夹，获取所有图片路径
        for root, _, files in os.walk(image_folder):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image


def get_dataloader(image_folder, batch_size=128, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageDataset(image_folder=image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader