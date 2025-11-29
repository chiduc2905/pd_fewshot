"""PD Scalogram Dataset Loader.

- Input: 64×64 RGB images
- Normalization: Auto-computed from dataset
- Split: 30 samples/class for val, 30 samples/class for test, remainder for training
"""
import os
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


CLASS_MAP = {'corona': 0, 'no_pd': 1, 'surface': 2}


class PDScalogram:
    """Dataset loader with auto-computed normalization."""
    
    def __init__(self, data_path, val_per_class=30, test_per_class=30):
        """
        Args:
            data_path: Path to dataset directory
            val_per_class: Samples reserved for validation per class (default: 50)
            test_per_class: Samples reserved for test per class (default: 50)
        """
        self.data_path = os.path.abspath(data_path)
        self.val_per_class = val_per_class
        self.test_per_class = test_per_class
        self.classes = sorted(CLASS_MAP.keys(), key=lambda c: CLASS_MAP[c])
        
        # Placeholders
        self.X_train, self.y_train = [], []
        self.X_val, self.y_val = [], []
        self.X_test, self.y_test = [], []
        self.mean, self.std = None, None
        
        # Base transform (no normalization yet)
        self._base_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        print(f'Dataset: {self.data_path}')
        self._compute_stats()
        self._load_data()
        self._shuffle_all()
    
    def _compute_stats(self):
        """Compute per-channel mean and std."""
        print('Computing mean/std...')
        pixels = []
        
        for class_name in CLASS_MAP:
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                continue
            
            for fname in os.listdir(class_path):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img = Image.open(os.path.join(class_path, fname)).convert('RGB')
                pixels.append(self._base_transform(img).numpy())
        
        all_imgs = np.stack(pixels)  # (N, 3, H, W)
        self.mean = all_imgs.mean(axis=(0, 2, 3)).tolist()
        self.std = all_imgs.std(axis=(0, 2, 3)).tolist()
        
        print(f'  Mean: {[f"{m:.3f}" for m in self.mean]}')
        print(f'  Std:  {[f"{s:.3f}" for s in self.std]}')
        
        # Final transform with normalization
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def _load_data(self):
        """Load and split data into train/val/test."""
        # Find min class size
        class_sizes = {}
        for class_name in CLASS_MAP:
            path = os.path.join(self.data_path, class_name)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                class_sizes[class_name] = len(files)
        
        min_size = min(class_sizes.values())
        val_size = min(self.val_per_class, min_size)
        test_size = min(self.test_per_class, min_size - val_size)
        eval_total = val_size + test_size
        
        print(f'Split: {val_size}/class for val, {test_size}/class for test, rest for train')
        
        for class_name, label in CLASS_MAP.items():
            path = os.path.join(self.data_path, class_name)
            if not os.path.exists(path):
                print(f'  Warning: {path} not found')
                continue
            
            files = sorted([f for f in os.listdir(path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            random.Random(42).shuffle(files)
            files = files[:min_size]  # Balance classes
            
            # Split: val_size for val, test_size for test, rest for train
            val_files = files[:val_size]
            test_files = files[val_size:eval_total]
            train_files = files[eval_total:]
            
            # Load images
            for fname in train_files:
                img = Image.open(os.path.join(path, fname)).convert('RGB')
                self.X_train.append(self.transform(img).numpy())
                self.y_train.append(label)
            
            for fname in val_files:
                img = Image.open(os.path.join(path, fname)).convert('RGB')
                self.X_val.append(self.transform(img).numpy())
                self.y_val.append(label)
            
            for fname in test_files:
                img = Image.open(os.path.join(path, fname)).convert('RGB')
                self.X_test.append(self.transform(img).numpy())
                self.y_test.append(label)
        
        # Convert to arrays
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_val = np.array(self.X_val)
        self.y_val = np.array(self.y_val)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        
        print(f'Loaded: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}')
    
    def _shuffle_all(self):
        """Shuffle all splits with fixed seed."""
        for X, y, seed in [(self.X_train, self.y_train, 0),
                           (self.X_val, self.y_val, 1),
                           (self.X_test, self.y_test, 2)]:
            idx = list(range(len(X)))
            random.Random(seed).shuffle(idx)
            X[:] = X[idx]
            y[:] = y[idx]
