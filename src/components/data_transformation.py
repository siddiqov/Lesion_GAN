import os
import numpy as np
import cv2

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(folder.split('/')[-1])
    return images, labels

def prepare_data(base_dir='artifacts/dataset'):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for category in ['Lesion', 'Normal']:
        train_images, train_category_labels = load_images_from_folder(os.path.join(base_dir, 'train', category))
        test_images, test_category_labels = load_images_from_folder(os.path.join(base_dir, 'test', category))

        train_data.extend(train_images)
        train_labels.extend(train_category_labels)
        test_data.extend(test_images)
        test_labels.extend(test_category_labels)

    return (np.array(train_data), np.array(train_labels)), (np.array(test_data), np.array(test_labels))
