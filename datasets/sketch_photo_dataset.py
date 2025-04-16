import os
from PIL import Image
from torch.utils.data import Dataset

class SketchPhotoDataset(Dataset):
    def __init__(self, root_dir, paired=True, transform=None, sketch_set='tx_000000000000', photo_set='tx_000000000000'):
        super().__init__()
        self.root_dir = root_dir
        self.paired = paired
        self.transform = transform
        self.sketch_dir = os.path.join(root_dir, 'sketch', sketch_set)
        self.photo_dir = os.path.join(root_dir, 'photo', photo_set)
        # 递归收集所有类别下的图片
        self.sketch_imgs = []
        self.photo_imgs = []
        for cls in sorted(os.listdir(self.sketch_dir)):
            cls_path = os.path.join(self.sketch_dir, cls)
            if os.path.isdir(cls_path):
                for img in sorted(os.listdir(cls_path)):
                    if img.lower().endswith('.png'):
                        self.sketch_imgs.append(os.path.join(cls_path, img))
        for cls in sorted(os.listdir(self.photo_dir)):
            cls_path = os.path.join(self.photo_dir, cls)
            if os.path.isdir(cls_path):
                for img in sorted(os.listdir(cls_path)):
                    if img.lower().endswith('.jpg') or img.lower().endswith('.jpeg') or img.lower().endswith('.png'):
                        self.photo_imgs.append(os.path.join(cls_path, img))
        assert len(self.sketch_imgs) > 0 and len(self.photo_imgs) > 0, f"No images found in {self.sketch_dir} or {self.photo_dir}"
        self.length = min(len(self.sketch_imgs), len(self.photo_imgs))
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        sketch_path = self.sketch_imgs[idx % len(self.sketch_imgs)]
        if self.paired:
            photo_path = self.photo_imgs[idx % len(self.photo_imgs)]
        else:
            import random
            rand_idx = random.randint(0, len(self.photo_imgs) - 1)
            photo_path = self.photo_imgs[rand_idx]
        sketch = Image.open(sketch_path).convert('L')  # 单通道
        photo = Image.open(photo_path).convert('RGB')
        if self.transform:
            sketch = self.transform(sketch)
            photo = self.transform(photo)
        return {'sketch': sketch, 'photo': photo}
