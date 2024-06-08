import os
import shutil
import random

def create_dataset_dirs(base_dir):
    for split in ['train', 'test']:
        for category in ['Lesion', 'Normal']:
            os.makedirs(os.path.join(base_dir, split, category), exist_ok=True)

def split_and_copy_images(src_dir, dst_dir, split_ratio=0.7):
    for category in ['Lesion', 'Normal']:
        src_path = os.path.join(src_dir, category)
        images = os.listdir(src_path)
        random.shuffle(images)
        split_index = int(len(images) * split_ratio)
        
        train_images = images[:split_index]
        test_images = images[split_index:]
        
        for image in train_images:
            shutil.copy(os.path.join(src_path, image), os.path.join(dst_dir, 'train', category, image))
        
        for image in test_images:
            shutil.copy(os.path.join(src_path, image), os.path.join(dst_dir, 'test', category, image))
    
    print(f"Data copied from {src_dir} to {dst_dir}")

if __name__ == "__main__":
    src_dir = "notebook/data"
    dst_dir = "artifacts/dataset"
    create_dataset_dirs(dst_dir)
    split_and_copy_images(src_dir, dst_dir)
