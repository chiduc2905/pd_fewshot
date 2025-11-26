import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import random
import torchvision.transforms as transforms

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
        
        # Initial transform (without normalization) to compute mean/std
        self.resize_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        # Will be set after computing dataset statistics
        self.transform = None
        self.mean = None
        self.std = None

        print(f'Using dataset path: {self.data_path}')
        self._compute_mean_std()
        self._load_data()
        self._shuffle()
    
    def _compute_mean_std(self):
        """Compute mean and std from all images in the dataset."""
        print("Computing dataset mean and std...")
        all_pixels = []
        
        for class_name in class_idx.keys():
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                continue
            
            files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for fname in files:
                fpath = os.path.join(class_path, fname)
                img = Image.open(fpath).convert('RGB')
                img_tensor = self.resize_transform(img)  # (3, H, W)
                all_pixels.append(img_tensor)
        
        # Stack all images: (N, 3, H, W)
        all_images = np.stack([t.numpy() for t in all_pixels], axis=0)
        
        # Compute mean and std per channel
        self.mean = all_images.mean(axis=(0, 2, 3)).tolist()  # [mean_R, mean_G, mean_B]
        self.std = all_images.std(axis=(0, 2, 3)).tolist()    # [std_R, std_G, std_B]
        
        print(f"Dataset mean: {self.mean}")
        print(f"Dataset std: {self.std}")
        
        # Define final transform with computed normalization
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
    def _load_data(self):
        # Dataset split: 70% train, 30% test
        
        for class_name, class_label in class_idx.items():
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Path not found {class_path}")
                continue
                
            files = sorted([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            if len(files) == 0:
                print(f"Warning: Class {class_name} has no images.")
                continue
            
            # Shuffle files with fixed seed for reproducibility
            random.Random(42).shuffle(files)
            
            # Calculate split sizes (70% train, 30% test)
            total_files = len(files)
            train_size = int(total_files * 0.7)
            # Remaining goes to test
            
            # Apply training_samples limit if specified
            if self.total_training_samples is not None:
                samples_per_class_train = self.total_training_samples // self.nclasses
                train_size = min(train_size, samples_per_class_train)
                print(f"Limiting training data: {samples_per_class_train} samples per class "
                      f"(Total: {self.total_training_samples})")
            
            # Split files
            train_files = files[:train_size]
            test_files = files[train_size:]
            
            # Load Train
            for fname in train_files:
                fpath = os.path.join(class_path, fname)
                img = Image.open(fpath).convert('RGB')
                img_tensor = self.transform(img)
                self.X_train.append(img_tensor.numpy())
                self.y_train.append(class_label)
            
            # Load Test
            for fname in test_files:
                fpath = os.path.join(class_path, fname)
                img = Image.open(fpath).convert('RGB')
                img_tensor = self.transform(img)
                self.X_test.append(img_tensor.numpy())
                self.y_test.append(class_label)
                
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        
        print(f"Data loaded: Train: {len(self.X_train)} (Total), Test: {len(self.X_test)} (Total)")
        
        # Balance Check
        unique_train, counts_train = np.unique(self.y_train, return_counts=True)
        unique_test, counts_test = np.unique(self.y_test, return_counts=True)
        print(f"Training Class Distribution: {dict(zip(unique_train, counts_train))}")
        print(f"Test Class Distribution: {dict(zip(unique_test, counts_test))}")

    def _shuffle(self):
        # shuffle training samples
        index = list(range(self.X_train.shape[0]))
        random.Random(0).shuffle(index)
        self.X_train = self.X_train[index]
        self.y_train = np.array(tuple(self.y_train[i] for i in index))

        # shuffle test samples
        index = list(range(self.X_test.shape[0]))
        random.Random(0).shuffle(index)
        self.X_test = self.X_test[index]
        self.y_test = np.array(tuple(self.y_test[i] for i in index))
