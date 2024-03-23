import os
from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np
import json
from tqdm import tqdm
import argparse
import threading
from queue import Queue
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler
from accelerate import Accelerator
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets.folder import default_loader

from diffusion.model.t5 import T5Embedder
from diffusion.utils.logger import get_logger
from diffusers.models import AutoencoderKL
from diffusion.data.datasets.InternalData import InternalData
from diffusion.utils.misc import SimpleTimer
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.data.builder import DATASETS
from diffusion.data import ASPECT_RATIO_512, ASPECT_RATIO_1024, ASPECT_RATIO_256
from diffusion.data.datasets.utils import get_vae_feature_path, get_t5_feature_path

logger = get_logger(__name__)

def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)

def get_vae_signature(resolution, is_multiscale):
    assert resolution in [256, 512, 1024]
    first_part = 'multiscale' if is_multiscale else 'cropped'
    return f"{first_part}-{resolution}"

@DATASETS.register_module()
class DatasetMS(InternalData):
    def __init__(self, root, image_list_json=None, transform=None, load_vae_feat=False, aspect_ratio_type=None, start_index=0, end_index=100000000, **kwargs):
        if image_list_json is None:
            image_list_json = ['data_info.json']
        assert os.path.isabs(root), 'root must be a absolute path'
        self.root = root
        self.transform = transform
        self.load_vae_feat = load_vae_feat
        self.meta_data_clean = []
        self.img_samples = []
        self.txt_feat_samples = []
        self.aspect_ratio = aspect_ratio_type
        assert self.aspect_ratio in [ASPECT_RATIO_1024, ASPECT_RATIO_512, ASPECT_RATIO_256]
        self.ratio_index = {}
        self.ratio_nums = {}
        for k, v in self.aspect_ratio.items():
            self.ratio_index[float(k)] = []     # used for self.getitem
            self.ratio_nums[float(k)] = 0      # used for batch-sampler

        vae_already_processed = []
        image_list_json = image_list_json if isinstance(image_list_json, list) else [image_list_json]
        for json_file in image_list_json:
            meta_data = self.load_json(os.path.join(self.root, 'partition', json_file))
            logger.info(f'json_file: {json_file} has {len(meta_data)} meta_data')
            for item in meta_data:
                if item['ratio'] <= 4:
                    sample_path = os.path.join(self.root, item['path'])
                    # this dataset seems to be for multiscale vae extraction only
                    signature = get_vae_signature(resolution=image_resize, is_multiscale=True)
                    output_file_path = get_vae_feature_path(
                        vae_save_root=vae_save_root, 
                        image_path=sample_path,
                        signature=signature,
                        relative_root_dir=self.root,
                        )
                    if not os.path.exists(output_file_path):
                        self.meta_data_clean.append(item)
                        self.img_samples.append(sample_path)
                    else:
                        vae_already_processed.append(sample_path)

        logger.info(f"VAE processing skipping {len(vae_already_processed)} images already processed")

        self.img_samples = self.img_samples[start_index: end_index]
        # scan the dataset for ratio static
        for i, info in enumerate(self.meta_data_clean[:len(self.meta_data_clean)//3]):
            ori_h, ori_w = info['height'], info['width']
            closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
            self.ratio_nums[closest_ratio] += 1
            if len(self.ratio_index[closest_ratio]) == 0:
                self.ratio_index[closest_ratio].append(i)

        # Set loader and extensions
        if self.load_vae_feat:
            raise ValueError("No VAE loader here")
        self.loader = default_loader

    def __getitem__(self, idx):
        data_info = {}
        for _ in range(20):
            try:
                img_path = self.img_samples[idx]
                img = self.loader(img_path)
                if self.transform:
                    img = self.transform(img)
                # Calculate closest aspect ratio and resize & crop image[w, h]
                if isinstance(img, Image.Image):
                    h, w = (img.size[1], img.size[0])
                    assert h, w == (self.meta_data_clean[idx]['height'], self.meta_data_clean[idx]['width'])
                    closest_size, closest_ratio = get_closest_ratio(h, w, self.aspect_ratio)
                    closest_size = list(map(lambda x: int(x), closest_size))
                    transform = T.Compose([
                        T.Lambda(lambda img: img.convert('RGB')),
                        T.Resize(closest_size, interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
                        T.CenterCrop(closest_size),
                        T.ToTensor(),
                        T.Normalize([.5], [.5]),
                    ])
                    img = transform(img)
                    data_info['img_hw'] = torch.tensor([h, w], dtype=torch.float32)
                    data_info['aspect_ratio'] = closest_ratio
                # change the path according to your data structure
                return img, self.img_samples[idx]
            except Exception as e:
                logger.exception(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')

    def get_data_info(self, idx):
        data_info = self.meta_data_clean[idx]
        return {'height': data_info['height'], 'width': data_info['width']}


# def extract_caption_t5_do(q):
#     while not q.empty():
#         item = q.get()
#         extract_caption_t5_job(item)
#         q.task_done()


# def extract_caption_t5_job(item):
#     global mutex
#     global t5
#     global t5_save_dir

#     with torch.no_grad():
#         caption = item['prompt'].strip()
#         if isinstance(caption, str):
#             caption = [caption]

#         output_path = get_t5_feature_path(
#             t5_save_dir=t5_save_dir, 
#             image_path=item['path'],
#             relative_root_dir=dataset_root,
#             max_token_length=t5_max_token_length,
#             )
        
#         output_dir = os.path.dirname(output_path)
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir, exist_ok=True)
        
#         if os.path.exists(output_path):
#             return

#         try:
#             mutex.acquire()
#             caption_emb, emb_mask = t5.get_text_embeddings(caption)
#             mutex.release()
#             emb_dict = {
#                 'caption_feature': caption_emb.float().cpu().data.numpy(),
#                 'attention_mask': emb_mask.cpu().data.numpy(),
#             }
#             np.savez_compressed(output_path, **emb_dict)
#         except Exception as e:
#             print(e)

def extract_caption_t5_batch(batch):
    global mutex
    global t5
    global t5_save_dir

    with torch.no_grad():
        captions = [item['prompt'].strip() for item in batch]
        output_paths = [get_t5_feature_path(
            t5_save_dir=t5_save_dir, 
            image_path=item['path'],
            relative_root_dir=dataset_root,
            max_token_length=t5_max_token_length,
        ) for item in batch]

        # Create output directories if they don't exist
        for output_path in output_paths:
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

        try:
            mutex.acquire()
            caption_embs, emb_masks = t5.get_text_embeddings(captions)
            mutex.release()

            for i, output_path in enumerate(output_paths):
                emb_dict = {
                    'caption_feature': caption_embs[i].float().cpu().data.numpy(),
                    'attention_mask': emb_masks[i].cpu().data.numpy(),
                }
                np.savez_compressed(output_path, **emb_dict)
        except Exception as e:
            print(e)

def extract_caption_t5(t5_batch_size):
    global t5
    global t5_save_dir
    global mutex
    global json_path
    global t5_max_token_length

    os.makedirs(t5_save_dir, exist_ok=True)

    train_data_json = json.load(open(json_path, 'r'))
    train_data = train_data_json[args.start_index: args.end_index]

    completed_paths = set([item['path'] for item in train_data if os.path.exists(get_t5_feature_path(
        t5_save_dir=t5_save_dir, 
        image_path=item['path'],
        relative_root_dir=dataset_root,
        max_token_length=t5_max_token_length,
        ))])
    print(f"Skipping t5 extraction for {len(completed_paths)} items with existing .npz files.")

    # global images_extension
    t5 = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=f'{args.pretrained_models_dir}/t5_ckpts', 
        model_max_length=t5_max_token_length
        )
    
    mutex = threading.Lock()

    batch_size = t5_batch_size
    batches = [train_data[i:i+batch_size] for i in range(0, len(train_data), batch_size)]

    threads = []
    for batch in batches:
        thread = threading.Thread(target=extract_caption_t5_batch, args=(batch,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    # jobs = Queue()

    # for item in tqdm(train_data):
    #     if item['path'] not in completed_paths:
    #         jobs.put(item)
    #         # remove later
    #         print(f"Adding {item['path']} to queue")

    # for _ in range(20):
    #     worker = threading.Thread(target=extract_caption_t5_do, args=(jobs,))
    #     worker.start()

    # jobs.join()

def save_results(results, paths, signature, vae_save_root):
    # save to npy
    new_paths = []
    os.umask(0o000)  # file permission: 666; dir permission: 777
    for res, p in zip(results, paths):
        output_path = get_vae_feature_path(
            vae_save_root=vae_save_root, 
            image_path=p, 
            signature=signature,
            relative_root_dir=dataset_root,
            )
        dirname = os.path.dirname(output_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        dirname_base = os.path.basename(dirname)
        filename = os.path.basename(output_path)
        new_paths.append(os.path.join(dirname_base, filename))
        np.save(output_path, res)
    # save paths
    with open(os.path.join(vae_save_root, f"VAE-{signature}.txt"), 'a') as f:
        f.write('\n'.join(new_paths) + '\n')


def inference(vae, dataloader, signature, vae_save_root):
    timer = SimpleTimer(len(dataloader), log_interval=1, desc="VAE-Inference")

    for batch in dataloader:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                posterior = vae.encode(batch[0]).latent_dist
                results = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy()
        path = batch[1]
        save_results(results, path, signature=signature, vae_save_root=vae_save_root)
        timer.log()


def extract_img_vae_multiscale(batch_size=1):
    assert image_resize in [256, 512, 1024]
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(vae_save_root, exist_ok=True)
    accelerator = Accelerator(mixed_precision='fp16')
    vae = AutoencoderKL.from_pretrained(f'{args.pretrained_models_dir}/sd-vae-ft-ema').to(device)

    signature = get_vae_signature(resolution=image_resize, is_multiscale=True)
    
    aspect_ratio_type = {
        256: ASPECT_RATIO_256,
        512: ASPECT_RATIO_512,
        1024: ASPECT_RATIO_1024
    }[image_resize]
    dataset = DatasetMS(dataset_root, image_list_json=[json_file], transform=None, sample_subset=None,
                        aspect_ratio_type=aspect_ratio_type, start_index=start_index, end_index=end_index,)

    # create AspectRatioBatchSampler
    sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset, batch_size=batch_size, aspect_ratios=dataset.aspect_ratio, ratio_nums=dataset.ratio_nums)

    # create DataLoader
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=13, pin_memory=True)
    dataloader = accelerator.prepare(dataloader, )

    inference(vae, dataloader, signature=signature, vae_save_root=vae_save_root)
    accelerator.wait_for_everyone()
    logger.info('finished extract_img_vae_multiscale')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_scale", action='store_true', default=False, help="multi-scale feature extraction")
    parser.add_argument("--img_size", default=512, type=int, choices=[256, 512, 1024], help="image scale for multi-scale feature extraction")
    parser.add_argument('--vae_batch_size', default=1, type=int)
    parser.add_argument('--t5_batch_size', default=1, type=int)
    parser.add_argument('--t5_max_token_length', default=120, type=int)
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--end_index', default=1000000, type=int)
    
    parser.add_argument('--t5_save_root', default='data/data_toy/caption_feature_wmask', type=str)
    parser.add_argument('--vae_save_root', default='data/data_toy/img_vae_features', type=str)
    parser.add_argument('--dataset_root', default='data/data_toy', type=str)
    parser.add_argument('--pretrained_models_dir', default='output/pretrained_models', type=str)

    parser.add_argument('--skip_t5', action='store_true', default=False, help="skip t5 feature extraction")
    parser.add_argument('--skip_vae', action='store_true', default=False, help="skip vae feature extraction")

    ### for multi-scale(ms) vae feauture extraction
    parser.add_argument('--json_file', type=str)
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_resize = args.img_size
    multi_scale = args.multi_scale
    vae_save_root = os.path.abspath(args.vae_save_root)
    t5_save_dir = args.t5_save_root
    
    json_file = args.json_file
    json_path = json_file # pretty sure this is just duplicate. can clean this up later
    t5_max_token_length = args.t5_max_token_length
    dataset_root = args.dataset_root
    vae_batch_size = args.vae_batch_size
    t5_batch_size = args.t5_batch_size

    start_index = args.start_index
    end_index = args.end_index

    if not args.skip_t5:
        # prepare extracted caption t5 features for training
        logger.info(f"Extracting T5 features for {json_path}\nMax token length: {t5_max_token_length}\nDevice: {device}\nSave to: {t5_save_dir}")
        extract_caption_t5(t5_batch_size=t5_batch_size)

    if not args.skip_vae:
        # prepare extracted image vae features for training
        logger.info(f"Extracting VAE features for {json_path}\nmulti_scale: {multi_scale}\nimage_resize: {image_resize}\nDevice: {device}\nSave to: {vae_save_root}")
        if not multi_scale:
            # basically seemed like the two did the same thing except one code path was shittier
            # and the non-multi-scale cropped to square instead of looking for nearest aspect ratio
            logger.warning('Single scale feature extraction is not supported currently. Images not be forced into squares.')

        # recommend bs = 1 for AspectRatioBatchSampler
        # not sure why bs = 1 is recommended. bigger batches are used in training. try higher.
        extract_img_vae_multiscale(batch_size=vae_batch_size)