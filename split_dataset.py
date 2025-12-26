"""
Split scalogram_minh dataset into train/val/test folders.

Input structure:
    scalogram_minh/
        corona/     (2500 images)
        hf_nopd/    (2500 images)
        surface/    (2500 images)
        void/       (2500 images)

Output structure:
    scalogram_minh/
        train/      (1500 images/class)
            corona/
            hf_nopd/
            surface/
            void/
        val/        (500 images/class)
            ...
        test/       (500 images/class)
            ...

Usage:
    python split_dataset.py --data_path /path/to/scalogram_minh
"""

import os
import shutil
import random
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Split scalogram_minh dataset')
    parser.add_argument('--data_path', type=str, 
                        default='/mnt/disk2/nhatnc/res/scalogram_fewshot/pulse_fewshot/scalogram_minh',
                        help='Path to scalogram_minh dataset')
    parser.add_argument('--train_per_class', type=int, default=1500)
    parser.add_argument('--val_per_class', type=int, default=500)
    parser.add_argument('--test_per_class', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--copy', action='store_true', 
                        help='Copy files instead of moving them')
    return parser.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    
    data_path = args.data_path
    classes = ['corona', 'hf_nopd', 'surface', 'void']
    splits = ['train', 'val', 'test']
    split_sizes = {
        'train': args.train_per_class,
        'val': args.val_per_class,
        'test': args.test_per_class
    }
    
    print(f"Dataset path: {data_path}")
    print(f"Split sizes: train={args.train_per_class}, val={args.val_per_class}, test={args.test_per_class}")
    print(f"Total per class: {sum(split_sizes.values())}")
    print(f"Mode: {'COPY' if args.copy else 'MOVE'}")
    print("=" * 60)
    
    # Check if already split
    if os.path.exists(os.path.join(data_path, 'train')):
        print("ERROR: Dataset already has train/ folder!")
        print("       Please backup and remove existing train/val/test folders first.")
        return
    
    # Create split folders
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(data_path, split, cls), exist_ok=True)
    
    # Process each class
    for cls in classes:
        class_path = os.path.join(data_path, cls)
        
        if not os.path.exists(class_path):
            print(f"WARNING: Class folder not found: {class_path}")
            continue
        
        # Get all images
        images = sorted([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"\n{cls}: {len(images)} images")
        
        # Shuffle
        random.shuffle(images)
        
        # Check if enough images
        total_needed = sum(split_sizes.values())
        if len(images) < total_needed:
            print(f"  WARNING: Only {len(images)} images, need {total_needed}")
            # Adjust proportionally
            ratio = len(images) / total_needed
            split_sizes_adjusted = {k: int(v * ratio) for k, v in split_sizes.items()}
        else:
            split_sizes_adjusted = split_sizes.copy()
        
        # Split images
        idx = 0
        for split in splits:
            count = split_sizes_adjusted[split]
            split_images = images[idx:idx + count]
            idx += count
            
            # Move/copy files
            for img in tqdm(split_images, desc=f"  {split}", leave=False):
                src = os.path.join(class_path, img)
                dst = os.path.join(data_path, split, cls, img)
                
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(src, dst)
            
            print(f"  {split}: {count} images")
        
        # Remove empty class folder if moved (not copied)
        if not args.copy:
            remaining = os.listdir(class_path)
            if len(remaining) == 0:
                os.rmdir(class_path)
                print(f"  Removed empty folder: {class_path}")
            else:
                print(f"  Remaining in original folder: {len(remaining)}")
    
    print("\n" + "=" * 60)
    print("Dataset split complete!")
    print("=" * 60)
    
    # Print final statistics
    print("\nFinal structure:")
    for split in splits:
        split_path = os.path.join(data_path, split)
        total = 0
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            if os.path.exists(cls_path):
                count = len(os.listdir(cls_path))
                total += count
        print(f"  {split}: {total} images")


if __name__ == '__main__':
    main()
