import os
import random
import json
import cv2
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm


BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
CATEGORY_ID_TO_NAME = {1: 'hole'}

class DataProcessor:
    def __init__(self, data_path, image_dir, output_dir):
        self.data = self.load_data(data_path)
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.augmentation_counts = defaultdict(int)
        
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.CLAHE(p=0.1),
            A.RandomSunFlare(p=0.1),
            A.RandomShadow(p=0.1),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    @staticmethod
    def load_data(data_path):
        with open(data_path) as f:
            return json.load(f)

    def create_directory_structure(self):
        dirs = ['train/images', 'train/labels', 
                'val/images', 'val/labels', 
                'test/images', 'test/labels']
        for dir_path in dirs:
            os.makedirs(os.path.join(self.output_dir, dir_path), exist_ok=True)

    def get_annotations(self, image_ids):
        if not isinstance(image_ids, list):
            image_ids = [image_ids]
            
        image_id_to_filename = {img['id']: img['file_name'] for img in self.data['images']}
        result = {img_id: {'file_name': image_id_to_filename.get(img_id), 'bboxes': []} 
                 for img_id in image_ids}
        
        for ann in self.data['annotations']:  
            if ann['image_id'] in image_ids:
                result[ann['image_id']]['bboxes'].append(ann['bbox'])
                
        return result

    def visualize_bbox(self, img, bbox, class_name, color=BOX_COLOR, thickness=2):
        x_min, y_min, w, h = bbox
        x_min, x_max = int(x_min), int(x_min + w)
        y_min, y_max = int(y_min), int(y_min + h)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

        (text_width, text_height), _ = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        
        cv2.rectangle(
            img, 
            (x_min, y_min - int(1.3 * text_height)), 
            (x_min + text_width, y_min), 
            color, -1
        )

        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )
        return img

    def visualize(self, image, bboxes, category_ids):
        img = image.copy()
        for bbox, category_id in zip(bboxes, category_ids):
            class_name = CATEGORY_ID_TO_NAME[category_id]
            img = self.visualize_bbox(img, bbox, class_name)
        
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    def save_yolo_annotations(self, image_id, bboxes, image_path, output_dir):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return

        h, w = img.shape[:2]
        yolo_labels = []
        
        for bbox in bboxes:
            x_min, y_min, w_box, h_box = bbox
            x_center = (x_min + w_box / 2) / w
            y_center = (y_min + h_box / 2) / h
            width = w_box / w
            height = h_box / h
            yolo_labels.append(f"0 {x_center} {y_center} {width} {height}")
        
        label_file = os.path.join(output_dir, 'labels', f"{image_id}.txt")
        with open(label_file, 'w') as f:
            f.write("\n".join(yolo_labels))

    def process_images(self, image_ids, output_subdir, augmentations_per_image=0):

        annotations = self.get_annotations(image_ids)
        
        for image_id, info in tqdm(annotations.items(), desc=f"Processing {output_subdir}"):
            if info['file_name'] is None:
                print(f"\nError: Image with ID {image_id} not found")
                continue

            image_path = os.path.join(self.image_dir, info['file_name'])
            original_image = cv2.imread(image_path)
            
            if original_image is None:
                print(f"\nError loading image: {image_path}")
                continue

            original_bboxes = info['bboxes']
            category_ids = [1] * len(original_bboxes)

            output_image_dir = os.path.join(self.output_dir, output_subdir, 'images')
            os.makedirs(output_image_dir, exist_ok=True)
            
            original_output_path = os.path.join(output_image_dir, f"{image_id}_0.jpg")
            cv2.imwrite(original_output_path, original_image)
            
            output_label_dir = os.path.join(self.output_dir, output_subdir)
            self.save_yolo_annotations(f"{image_id}_0", original_bboxes, image_path, output_label_dir)

            for aug_num in range(1, augmentations_per_image + 1):
                try:
                    transformed = self.transform(
                        image=original_image, 
                        bboxes=original_bboxes, 
                        category_ids=category_ids
                    )
                    aug_image = transformed['image']
                    aug_bboxes = transformed['bboxes']

                    aug_output_path = os.path.join(output_image_dir, f"{image_id}_{aug_num}.jpg")
                    cv2.imwrite(aug_output_path, aug_image)
                    
                    self.save_yolo_annotations(
                        f"{image_id}_{aug_num}", 
                        aug_bboxes, 
                        image_path, 
                        output_label_dir
                    )
                    
                    self.augmentation_counts[image_id] += 1
                    
                except Exception as e:
                    print(f"\nError augmenting image {image_id}: {str(e)}")

    def split_data(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):

        image_ids = [img['id'] for img in self.data['images']]
        total_images = len(image_ids)

        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        test_count = total_images - train_count - val_count
        
        remaining_ids = image_ids.copy()
        random.shuffle(remaining_ids)
        
        train_ids = remaining_ids[:train_count]
        val_ids = remaining_ids[train_count:train_count+val_count]
        test_ids = remaining_ids[train_count+val_count:]
        
        return train_ids, val_ids, test_ids

    def run_pipeline(self, train_augment=5, val_augment=2):

        print("Starting data processing pipeline...")
        print(f"Total images available: {len(self.data['images'])}")
        
        self.create_directory_structure()
        train_ids, val_ids, test_ids = self.split_data()
        
        print("\nDataset split:")
        print(f"Training set: {len(train_ids)} images")
        print(f"Validation set: {len(val_ids)} images")
        print(f"Test set: {len(test_ids)} images")
        
        print("\nProcessing datasets:")
        self.process_images(train_ids, 'train', augmentations_per_image=train_augment)
        self.process_images(val_ids, 'val', augmentations_per_image=val_augment)
        self.process_images(test_ids, 'test', augmentations_per_image=0)
        
        final_train = len(train_ids) * (1 + train_augment)
        final_val = len(val_ids) * (1 + val_augment)
        final_test = len(test_ids)
        
        print("\nFinal dataset sizes:")
        print(f"Training set: {final_train} images ({len(train_ids)} originals + {len(train_ids)*train_augment} augmented)")
        print(f"Validation set: {final_val} images ({len(val_ids)} originals + {len(val_ids)*val_augment} augmented)")
        print(f"Test set: {final_test} images")

if __name__ == "__main__":
    processor = DataProcessor(
        data_path='result.json',
        image_dir='.',
        output_dir='./data'
    )
    
    processor.run_pipeline(
        train_augment=5,  
        val_augment=2     
    )
