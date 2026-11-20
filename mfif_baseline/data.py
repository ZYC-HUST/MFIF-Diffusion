"""Small optional PyTorch reader; all three tensors are RGB floats in [0,1]."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class MFIFTriplets(Dataset):
    def __init__(self,root):
        self.root=Path(root)
        self.rows=[]
        with (self.root/'manifest.jsonl').open() as f:
            for line in f:
                row=json.loads(line)
                self.rows.append(dict(sample_id=row['sample_id'],
                                      paths=[row['files'][kind]['path'] for kind in ['A','B','GT']]))

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];images=[]
        for path in row['paths']:
            with Image.open(self.root/path) as image:
                pixels=np.array(image.convert('RGB'),copy=True)
            images.append(torch.from_numpy(pixels).permute(2,0,1).float()/255)
        # Mirrors the public Mask-DiFuser loader's outer structure.
        return images,row['sample_id']
