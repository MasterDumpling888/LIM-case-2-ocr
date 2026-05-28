import os
import json
import glob

class SROIEDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'img')
        self.entities_dir = os.path.join(root_dir, 'entities')
        self.box_dir = os.path.join(root_dir, 'box')
        
        self.filenames = []
        if os.path.exists(self.img_dir):
            self.filenames = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
    def __len__(self):
        return len(self.filenames)
        
    def get_example(self, index):
        filename = self.filenames[index]
        img_path = glob.glob(os.path.join(self.img_dir, f"{filename}.*"))[0]
        entity_path = os.path.join(self.entities_dir, f"{filename}.txt")
        box_path = os.path.join(self.box_dir, f"{filename}.txt")
        
        entities = {}
        if os.path.exists(entity_path):
            with open(entity_path, 'r', encoding='utf-8', errors='ignore') as f:
                try:
                    entities = json.load(f)
                except json.JSONDecodeError:
                    # Some files might not be perfect JSON
                    pass
                    
        boxes = []
        if os.path.exists(box_path):
            with open(box_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split(',', 8)
                    if len(parts) == 9:
                        boxes.append({
                            'coords': [int(p) for p in parts[:8]],
                            'text': parts[8]
                        })
                        
        return {
            'filename': filename,
            'img_path': img_path,
            'entities': entities,
            'boxes': boxes
        }
