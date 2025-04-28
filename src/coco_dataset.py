import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import numpy as np


class COCODataset(Dataset):
    def __init__(self,
                 root_dir=r'C:\Users\pjtka\Documents\COCO',
                 split='train',
                 transform=None):
        """
        Args:
            root_dir (string): Directory with all the COCO dataset images and annotations
            split (string): Dataset split - 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        # Set paths
        self.root_dir = root_dir
        self.split = split

        # Paths for images and annotations
        self.img_dir = os.path.join(root_dir, f'{split}2017/{split}2017')
        self.ann_file = os.path.join(root_dir, f'annotations_trainval2017/annotations/instances_{split}2017.json')

        # Initialize COCO API
        self.coco = COCO(self.ann_file)

        # Get all image IDs
        self.img_ids = list(self.coco.imgs.keys())

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Get category IDs and create mapping
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_class_id = {cat['id']: i for i, cat in enumerate(self.categories)}


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Returns a dictionary with:
        - 'image': transformed image tensor
        - 'class_labels': list of class labels for the image
        - 'label_mask': list of binary masks for each object in the image
        """
        # Get image info
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)

        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Process class labels and masks
        class_labels = []
        label_masks = []

        for ann in anns:
            # Convert COCO category ID to class index
            class_id = self.cat_id_to_class_id.get(ann['category_id'], -1)

            if class_id != -1:
                class_labels.append(class_id)

                # Create segmentation mask
                mask = self.coco.annToMask(ann)
                mask_tensor = torch.from_numpy(mask).float()

                # Resize mask to match image tensor
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=(image_tensor.shape[1], image_tensor.shape[2]),
                    mode='nearest'
                ).squeeze()

                label_masks.append(mask_tensor)

        return {
            'image': image_tensor,
            'class_labels': class_labels,
            'label_mask': label_masks,
            'img_path': img_path
        }

    def get_category_names(self):
        """
        Returns a list of category names in order of their class indices
        """
        return [cat['name'] for cat in sorted(self.categories, key=lambda x: self.cat_id_to_class_id[x['id']])]


# Example usage
if __name__ == '__main__':
    # Create dataset
    dataset = COCODataset(split='train')

    # Print category names
    print("Categories:", dataset.get_category_names())

    # Test data loading
    sample = dataset[0]
    print("Image shape:", sample['image'].shape)
    print("Class labels:", sample['class_labels'])
    print("Number of label masks:", len(sample['label_mask']))