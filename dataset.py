import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import random

class_idx = {
    'corona': 0,
    'no_pd': 1,
    'surface': 2
}

class PDScalogram:
    def __init__(self, data_path):
        self.data_path = data_path
        # Normalize path: handle relative and absolute paths
        if not os.path.isabs(data_path):
            self.data_path = os.path.abspath(data_path)
        
        self.classes = sorted(list(class_idx.keys()), key=lambda c: class_idx[c])
        self.nclasses = len(self.classes)
        
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        
        print(f'Using dataset path: {self.data_path}')
        self._load_data()
        self._shuffle()
        
    def _load_data(self):
        train_ratio = 0.8
        
        for class_name in self.classes:
            class_path = os.path.join(self.data_path, class_name)
            print(f'Checking class path: {class_path}, exists: {os.path.isdir(class_path)}')
            if not os.path.isdir(class_path):
                print(f'Warning: Class path not found: {class_path}')
                continue
                
            class_label = class_idx[class_name]
            image_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
            
            # filter labeled files
            image_files = [f for f in image_files if 'labeled' not in f]
            image_files = sorted(image_files)
            
            # split train/test
            split_idx = int(len(image_files) * train_ratio)
            train_files = image_files[:split_idx]
            test_files = image_files[split_idx:]
            
            # load train images
            for fname in train_files:
                fpath = os.path.join(class_path, fname)
                img = Image.open(fpath).convert('RGB')
                img_array = np.array(img) / 255.0
                self.X_train.append(img_array)
                self.y_train.append(class_label)
            
            # load test images
            for fname in test_files:
                fpath = os.path.join(class_path, fname)
                img = Image.open(fpath).convert('RGB')
                img_array = np.array(img) / 255.0
                self.X_test.append(img_array)
                self.y_test.append(class_label)
        
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
    
    def _shuffle(self):
        # shuffle training samples
        index = list(range(self.X_train.shape[0]))
        random.Random(0).shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = self.y_train[index]
        
        # shuffle test samples
        index = list(range(self.X_test.shape[0]))
        random.Random(0).shuffle(index)
        self.X_test = self.X_test[index]
        self.y_test = self.y_test[index]
