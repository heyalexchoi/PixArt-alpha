"""
Claude adaptation of https://github.com/google-research/google-research/blob/master/cmmd/embedding.py
Opted for using Claude over this adaptation https://github.com/sayakpaul/cmmd-pytorch 

Embedding models used in the CMMD calculation.
"""
import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection

class ClipImageEmbeddingModel:
    """CLIP image embedding calculator."""

    def __init__(self, 
                 pretrained_model_name_or_path="openai/clip-vit-large-patch14-336",
                 device=None,
                 ):
        self.model = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path).eval()
        if (device):
            self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)

    def get_embeddings(self, images):
        """Computes CLIP embeddings for the given PIL images.

        Args:
            images: A list of PIL images.

        Returns:
            Embedding tensor of shape (batch_size, embedding_width).
        """

        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.image_embeds