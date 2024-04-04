"""
Claude adaptation of https://github.com/google-research/google-research/blob/master/cmmd/io_util.py
Opted for using Claude over this adaptation https://github.com/sayakpaul/cmmd-pytorch 
because it seems some changes were made in the latter.

IO utilities.
"""
"""IO utilities."""

import os
from typing import List
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# written by claude as adaptation and not tested
def _get_image_list(path: str) -> List[str]:
    ext_list = ['png', 'jpg', 'jpeg']
    image_list = []
    for ext in ext_list:
        image_list.extend(list(filter(os.path.isfile, 
                            [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(ext)])))
    # Sort the list to ensure a deterministic output.
    image_list.sort()
    return image_list

# written by claude as adaptation and not tested
class ImageDataset(Dataset):
    def __init__(self, image_dir: str, processor: CLIPProcessor, max_count: int = -1):
        self.image_paths = _get_image_list(image_dir)
        if max_count > 0:
            self.image_paths = self.image_paths[:max_count]
        self.processor = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        return self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze()

# written by claude as adaptation and not tested
@torch.no_grad()
def compute_embeddings_for_dir(
    img_dir: str,
    embedding_model: CLIPModel,
    processor: CLIPProcessor,
    batch_size: int,
    max_count: int = -1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Computes embeddings for the images in the given directory."""
    dataset = ImageDataset(img_dir, processor, max_count=max_count)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    embedding_model.to(device)
    embedding_model.eval()

    all_embs = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = batch.to(device)
        embs = embedding_model.get_image_features(pixel_values=batch)
        all_embs.append(embs.cpu())

    all_embs = torch.cat(all_embs, dim=0)
    return all_embs