
"""Embedding models used in the CMMD calculation."""

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
import numpy as np
from diffusion.utils.logger import get_logger

logger = get_logger(__name__)

_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
_CUDA_AVAILABLE = torch.cuda.is_available()


# def _resize_bicubic(images, size):
#     images = torch.from_numpy(images.transpose(0, 3, 1, 2))
#     images = torch.nn.functional.interpolate(images, size=(size, size), mode="bicubic")
#     images = images.permute(0, 2, 3, 1).numpy()
#     return images

# do not intend to convert to and from numpy, so can remove the transpose and permute
# want to use default clip image processor functionality for normalization and resizing

class ClipEmbeddingModel:
    """CLIP image embedding calculator."""

    def __init__(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(_CLIP_MODEL_NAME)

        self._model = CLIPVisionModelWithProjection.from_pretrained(_CLIP_MODEL_NAME).eval()
        if _CUDA_AVAILABLE:
            self._model = self._model.cuda()

        # self.input_image_size = self.image_processor.crop_size["height"]
        # this should probably be 336
        # logger.info(f'CLIP Embedding model input_image_size: {self.input_image_size}')

    @torch.no_grad()
    def embed(self, images):
        """Computes CLIP embeddings for the given images.

        Args:
            images: A list of PIL images
            
          xxxximages: An image array of shape (batch_size, height, width, 3). Values are
            in range [0, 1].

        Returns:
          Embedding array of shape (batch_size, embedding_width).
        """

        # allow default behavior for rescale, resize, center crop, etc.
        # images = _resize_bicubic(images, self.input_image_size)
        inputs = self.image_processor(
            images=images,
            # do_normalize=True,
            # do_center_crop=False,
            # do_resize=False,
            # do_rescale=False,
            return_tensors="pt",
            padding=True,
        )
        if _CUDA_AVAILABLE:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # image_embs = self._model(**inputs).image_embeds.cpu()
        image_embs = self._model(**inputs).image_embeds
        # apparently this is wrong:
        # image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        image_embs /= torch.linalg.norm(image_embs, dim=-1, keepdim=True)
        return image_embs