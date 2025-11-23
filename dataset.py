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
    def __init__(self, data_path, total_training_samples=None):
        """
        Args:
            data_path: Path to dataset
            total_training_samples: If set (e.g. 30, 60), this is the TOTAL number of training samples 
                                    across all classes. They will be distributed evenly among classes.
                                    If None, use all available training data.
        """
        self.data_path = data_path
        self.total_training_samples = total_training_samples
        
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
        
    def _load_data(self):
        # Fixed test set size: 75 samples per class
        TEST_SAMPLES_PER_CLASS = 75
        
        # Determine samples per class for training if a total limit is set
        samples_per_class_train = None
        if self.total_training_samples is not None:
            samples_per_class_train = self.total_training_samples // self.nclasses
            print(f"Limiting training data: {self.total_training_samples} total -> ~{samples_per_class_train} per class")
        
        for class_name in self.classes:
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.isdir(class_path):
                print(f'Warning: Class path not found: {class_path}')
                continue
                
            class_label = class_idx[class_name]
            image_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
            
            # filter labeled files
            image_files = [f for f in image_files if 'labeled' not in f]
            
            # Shuffle before split
            random.Random(42).shuffle(image_files)
            
            if len(image_files) < TEST_SAMPLES_PER_CLASS:
                print(f"Warning: Class {class_name} has fewer than {TEST_SAMPLES_PER_CLASS} images.")
                test_files = image_files
                train_files = []
            else:
                test_files = image_files[:TEST_SAMPLES_PER_CLASS]
                train_files = image_files[TEST_SAMPLES_PER_CLASS:]
            
            # Limit training samples based on calculated per-class limit
            if samples_per_class_train is not None:
                if len(train_files) > samples_per_class_train:
                    train_files = train_files[:samples_per_class_train]
                else:
                     print(f"Warning: Class {class_name} only has {len(train_files)} training samples, fewer than requested {samples_per_class_train}.")

            # Load Train
            for fname in train_files:
                fpath = os.path.join(class_path, fname)
                img = Image.open(fpath).convert('RGB')
                img_array = np.array(img) / 255.0
                self.X_train.append(img_array)
                self.y_train.append(class_label)
            
            # Load Test
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
        
        print(f"Data loaded: Train: {len(self.X_train)} (Total), Test: {len(self.X_test)} (Total)")
        
        # Balance Check
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"Training Class Distribution: {dict(zip(unique, counts))}")

    # No global shuffle here to preserve class grouping logic if needed, 
    # but main.py converts to Dataset/DataLoader which will shuffle.
