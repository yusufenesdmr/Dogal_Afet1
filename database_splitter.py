import os
import shutil
import random

# Base dir: current directory (isleme1)
base_dir = os.getcwd()
src_data_dir = os.path.join(base_dir, 'database', 'original_data')
train_dir = os.path.join(base_dir, 'database', 'train')
test_dir = os.path.join(base_dir, 'database', 'test')

# Test ratio (Training: 80%, Test: 20%)
test_ratio = 0.2

def split_data():
    if not os.path.exists(src_data_dir):
        print(f"ERROR: Source directory not found: {src_data_dir}")
        return

    # Find class folders
    classes = [d for d in os.listdir(src_data_dir) if os.path.isdir(os.path.join(src_data_dir, d))]
    
    if not classes:
        print("ERROR: No class directories found in original_data!")
        return

    print(f"Found classes: {classes}")

    for class_name in classes:
        print(f"Processing {class_name}...")
        src_class_path = os.path.join(src_data_dir, class_name)
        
        train_class_path = os.path.join(train_dir, class_name)
        test_class_path = os.path.join(test_dir, class_name)
        
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)
        
        files = [f for f in os.listdir(src_class_path) if os.path.isfile(os.path.join(src_class_path, f))]
        
        # Shuffle for random split
        random.shuffle(files)
        
        split_idx = int(len(files) * (1 - test_ratio))
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        print(f"  - Total: {len(files)} | Train: {len(train_files)} | Test: {len(test_files)}")
        
        # Move files
        for f in train_files:
            try:
                shutil.move(os.path.join(src_class_path, f), os.path.join(train_class_path, f))
            except Exception as e:
                print(f"    Error moving {f} to train: {e}")
            
        for f in test_files:
            try:
                shutil.move(os.path.join(src_class_path, f), os.path.join(test_class_path, f))
            except Exception as e:
                print(f"    Error moving {f} to test: {e}")
            
    print("\nDATABASE SPLIT COMPLETED SUCCESSFULLY!")
    print(f"Train data: {train_dir}")
    print(f"Test data: {test_dir}")

if __name__ == '__main__':
    split_data()
