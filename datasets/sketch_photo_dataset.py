import os
import random
from PIL import Image
from torch.utils.data import Dataset

class SketchPhotoDataset(Dataset):
    def __init__(self, root_dir, paired=True, transform=None, sketch_sets=None, photo_sets=None):
        super().__init__()
        self.root_dir = root_dir
        self.paired = paired
        self.transform = transform
        # 
        self.sketch_sets = sketch_sets if sketch_sets is not None else ['tx_000000000000']
        self.photo_sets = photo_sets if photo_sets is not None else ['tx_000000000000']
        # 
        self.sketch_imgs = []
        self.photo_imgs = []
        self.sketch_cls_imgs = []  # [set][cls][img]
        self.photo_cls_imgs = []
        for sketch_set in self.sketch_sets:
            sketch_dir = os.path.join(root_dir, 'sketch', sketch_set)
            cls_imgs = []
            for cls in sorted(os.listdir(sketch_dir)):
                cls_path = os.path.join(sketch_dir, cls)
                if os.path.isdir(cls_path):
                    imgs = [os.path.join(cls_path, img) for img in sorted(os.listdir(cls_path)) if img.lower().endswith('.png')]
                    if imgs:
                        self.sketch_imgs.extend(imgs)
                        cls_imgs.append(imgs)
            self.sketch_cls_imgs.append(cls_imgs)
        for photo_set in self.photo_sets:
            photo_dir = os.path.join(root_dir, 'photo', photo_set)
            cls_imgs = []
            for cls in sorted(os.listdir(photo_dir)):
                cls_path = os.path.join(photo_dir, cls)
                if os.path.isdir(cls_path):
                    imgs = [os.path.join(cls_path, img) for img in sorted(os.listdir(cls_path)) if img.lower().endswith(('.jpg','.jpeg','.png'))]
                    if imgs:
                        self.photo_imgs.extend(imgs)
                        cls_imgs.append(imgs)
            self.photo_cls_imgs.append(cls_imgs)
        assert len(self.sketch_imgs) > 0 and len(self.photo_imgs) > 0, f"No images found in {self.sketch_sets} or {self.photo_sets}"
        #  __len__
        self.length = min(len(self.sketch_imgs), len(self.photo_imgs))
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        # ： set，
        sketch_set_idx = random.randint(0, len(self.sketch_sets)-1)
        photo_set_idx = random.randint(0, len(self.photo_sets)-1)
        # 
        sketch_cls_imgs = self.sketch_cls_imgs[sketch_set_idx]
        photo_cls_imgs = self.photo_cls_imgs[photo_set_idx]
        sketch_cls = random.choice(sketch_cls_imgs)
        photo_cls = random.choice(photo_cls_imgs)
        # 
        sketch_path = random.choice(sketch_cls)
        if self.paired:
            photo_path = random.choice(photo_cls)
        else:
            # 
            photo_path = random.choice(self.photo_imgs)
        sketch = Image.open(sketch_path).convert('L')
        photo = Image.open(photo_path).convert('RGB')
        if self.transform:
            sketch = self.transform(sketch)
            photo = self.transform(photo)
        return {'sketch': sketch, 'photo': photo}
