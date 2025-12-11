"""PD Scalogram Dataset Loader.

- Input: Configurable size RGB images (64×64 default, 84×84 for standard benchmarks)
- Normalization: Auto-computed from dataset
- Split: 40 samples/class for val, 40 samples/class for test, remainder for training
"""
import os
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


CLASS_MAP = {'surface': 0, 'corona': 1, 'no_pd': 2}


class PDScalogram:
    """Dataset loader with auto-computed normalization (from training set only)."""
    
    def __init__(self, data_path, val_per_class=40, test_per_class=40, image_size=64):
        """
        Args:
            data_path: Path to dataset directory
            val_per_class: Samples reserved for validation per class
            test_per_class: Samples reserved for test per class
            image_size: Input image size (default: 64, use 84 for standard benchmarks)
        """
        self.data_path = os.path.abspath(data_path)
        self.val_per_class = val_per_class
        self.test_per_class = test_per_class
        self.image_size = image_size
        self.classes = sorted(CLASS_MAP.keys(), key=lambda c: CLASS_MAP[c])
        
        # Placeholders
        self.X_train, self.y_train = [], []
        self.X_val, self.y_val = [], []
        self.X_test, self.y_test = [], []
        self.mean, self.std = None, None
        
        # File lists placeholders
        self.train_files = []
        self.val_files = []
        self.test_files = []
        
        # Base transform (no normalization yet)
        self._base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        print(f'Dataset: {self.data_path}')
        
        # 1. Prepare splits (identify files for train/val/test)
        self._prepare_splits()
        
        # 2. Compute stats (ONLY on training data)
        self._compute_stats()
        
        # 3. Load images (apply normalization)
        self._load_images()
        
        self._shuffle_all()
    
    def _prepare_splits(self):
        """Scan directories and split files into train/val/test lists."""
        # Find min class size
        class_sizes = {}
        for class_name in CLASS_MAP:
            path = os.path.join(self.data_path, class_name)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                class_sizes[class_name] = len(files)
            else:
                class_sizes[class_name] = 0
        
        if not class_sizes:
            raise ValueError(f"No data found in {self.data_path}")

        min_size = min(class_sizes.values())
        if min_size == 0:
            print("Warning: Found empty class or no images.")
            return {}, {}, {}

        val_size = min(self.val_per_class, min_size)
        test_size = min(self.test_per_class, min_size - val_size)
        eval_total = val_size + test_size
        
        print(f'Split: {val_size}/class for val, {test_size}/class for test, rest for train')
        
        for class_name in CLASS_MAP:
            path = os.path.join(self.data_path, class_name)
            if not os.path.exists(path):
                continue
                
            files = sorted([f for f in os.listdir(path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            random.Random(42).shuffle(files)
            files = files[:min_size]  # Balance classes
            
            # Split: val_size for val, test_size for test, rest for train
            val_files_class = files[:val_size]
            test_files_class = files[val_size:eval_total]
            train_files_class = files[eval_total:]
            
            # Store as (full_path, label) tuples
            label = CLASS_MAP[class_name]
            self.val_files.extend([(os.path.join(path, f), label) for f in val_files_class])
            self.test_files.extend([(os.path.join(path, f), label) for f in test_files_class])
            self.train_files.extend([(os.path.join(path, f), label) for f in train_files_class])

    def _compute_stats(self):
        """Compute per-channel mean and std using ONLY training data."""
        print('Computing mean/std on training set...')
        pixels = []
        
        for fpath, _ in self.train_files:
            img = Image.open(fpath).convert('RGB')
            pixels.append(self._base_transform(img).numpy())
        
        if not pixels:
            print("Warning: No training data found for stats computation. Using default mean/std.")
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        else:
            all_imgs = np.stack(pixels)  # (N, 3, H, W)
            self.mean = all_imgs.mean(axis=(0, 2, 3)).tolist()
            self.std = all_imgs.std(axis=(0, 2, 3)).tolist()
        
        print(f'  Mean: {[f"{m:.3f}" for m in self.mean]}')
        print(f'  Std:  {[f"{s:.3f}" for s in self.std]}')
        
        # Final transform with normalization
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def _load_images(self):
        """Load images using the pre-computed splits and normalization."""
        # Load Train
        for fpath, label in self.train_files:
            img = Image.open(fpath).convert('RGB')
            self.X_train.append(self.transform(img).numpy())
            self.y_train.append(label)
            
        # Load Val
        for fpath, label in self.val_files:
            img = Image.open(fpath).convert('RGB')
            self.X_val.append(self.transform(img).numpy())
            self.y_val.append(label)
            
        # Load Test
        for fpath, label in self.test_files:
            img = Image.open(fpath).convert('RGB')
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
            if len(X) > 0:
                idx = list(range(len(X)))
                random.Random(seed).shuffle(idx)
                X[:] = X[idx]
                y[:] = y[idx]
