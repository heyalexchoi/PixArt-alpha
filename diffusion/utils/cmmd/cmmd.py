"""
Claude adaptation of https://github.com/google-research/google-research/blob/master/cmmd/io_util.py
Opted for using Claude over this adaptation https://github.com/sayakpaul/cmmd-pytorch 
because it seems some changes were made in the latter.

IO utilities.
"""
"""IO utilities."""

import os
from typing import List, Union
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from .clip import ClipImageEmbeddingModel
from .mmd import compute_mmd
from transformers import CLIPImageProcessor

@torch.no_grad()
def get_embeddings_for_images(
    clip_model: ClipImageEmbeddingModel,
    images: List[Union[Image.Image, str]],
    batch_size: int,
    device: str,
    num_workers: int,
    max_count: int = -1,
) -> torch.Tensor:
    """
    Computes embeddings for the images in the given directory.
    arg `images` can be PIL images or paths to images.
    """
    processor = clip_model.processor

    dataset = ImageDataset(images, processor, max_count=max_count)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_embs = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = batch.to(device)
        embs = clip_model.get_embeddings(images=batch, preprocess=False)
        all_embs.append(embs)

    all_embs = torch.cat(all_embs, dim=0)
    return all_embs

def get_cmmd_for_images(
    ref_images: List[Union[Image.Image, str]],
    eval_images: List[Union[Image.Image, str]],
    batch_size: int = 64,
    num_workers: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    clip_model = ClipImageEmbeddingModel(device=device)
    ref_embs = get_embeddings_for_images(
        images=ref_images,
        clip_model=clip_model,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    eval_embs = get_embeddings_for_images(
        images=eval_images,
        clip_model=clip_model,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    clip_model.unload()
    return compute_mmd(ref_embs, eval_embs)

class ImageDataset(Dataset):
    def __init__(
            self, 
            images: List[Union[Image.Image, str]], 
            processor: CLIPImageProcessor, 
            max_count: int = -1,
        ):
        self.images = images
        if max_count > 0:
            self.images = self.images[:max_count]
        self.processor = processor

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = self.images[idx]
        if isinstance(image, str):
            image = Image.open(image)
        return self.processor(images=image, return_tensors="pt")

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