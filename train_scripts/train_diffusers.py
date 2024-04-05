import argparse
import datetime
import os
import sys
import time
import types
import warnings
from pathlib import Path
import random
import hashlib

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import accelerate
import gc
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from copy import deepcopy
from diffusers import AutoencoderKL, Transformer2DModel, PixArtAlphaPipeline, DPMSolverMultistepScheduler
from mmcv.runner import LogBuffer
from packaging import version
from torch.utils.data import RandomSampler
from transformers import T5Tokenizer, T5EncoderModel

from diffusion import IDDPM
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.utils.data_sampler import AspectRatioBatchSampler, BalancedAspectRatioBatchSampler
from diffusion.utils.dist_utils import get_world_size, clip_grad_norm_, flush
from diffusion.utils.logger import get_root_logger, rename_file_with_creation_time
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
# from diffusion.data.datasets.utils import get_clip_feature_path

warnings.filterwarnings("ignore")  # ignore warning


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'Transformer2DModel'


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


def token_drop(y, y_mask, force_drop_ids=None):
    """
    Drops labels to enable classifier-free guidance.
    """
    if force_drop_ids is None:
        drop_ids = torch.rand(y.shape[0]).cuda() < config.class_dropout_prob
    else:
        drop_ids = force_drop_ids == 1
    y = torch.where(drop_ids[:, None, None], uncond_prompt_embeds, y)
    y_mask = torch.where(drop_ids[:, None], uncond_prompt_attention_mask, y_mask)
    return y, y_mask

@torch.inference_mode()
def get_null_embed(npz_file, max_length=120):
    if os.path.exists(npz_file) and (npz_file.endswith('.npz') or npz_file.endswith('.pth')):
        data = torch.load(npz_file)
        uncond_prompt_embeds = data['uncond_prompt_embeds'].to(accelerator.device)
        uncond_prompt_attention_mask = data['uncond_prompt_attention_mask'].to(accelerator.device)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(args.pipeline_load_from, subfolder="text_encoder")
        uncond = tokenizer("", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        uncond_prompt_embeds = text_encoder(uncond.input_ids, attention_mask=uncond.attention_mask)[0]

        torch.save({
            'uncond_prompt_embeds': uncond_prompt_embeds.cpu(),
            'uncond_prompt_attention_mask': uncond.attention_mask.cpu()
        }, npz_file)

        uncond_prompt_embeds = uncond_prompt_embeds.to(accelerator.device)
        uncond_prompt_attention_mask = uncond.attention_mask.to(accelerator.device)

    return uncond_prompt_embeds, uncond_prompt_attention_mask

def get_path_for_validation_prompt(prompt, max_length):
    hash_object = hashlib.sha256(prompt.encode())
    hex_dig = hash_object.hexdigest()
    return f'output/tmp/{hex_dig}_{max_length}.pth'

# embed eval sample prompts and CMMD prompts so we don't need T5 during training
@torch.inference_mode()
def generate_t5_prompt_embeddings():
    batch_size = config.t5.batch_size
    cmmd_train_items, cmmd_val_items = get_cmmd_train_and_val_samples()
    if not eval_sample_prompts and not cmmd_train_items and not cmmd_val_items:
        logger.info("No evaluation prompts, or CMMD sample prompts provided. Skipping prompt embedding generation.")
        return
    
    prompts = eval_sample_prompts + \
        [item['prompt'] for item in cmmd_train_items] + \
        [item['prompt'] for item in cmmd_val_items]
    
    saved_prompts = []
    logger.info(f"Embedding {len(prompts)} prompts...")
    if accelerator.is_main_process:
        for prompt in prompts:
            path = get_path_for_validation_prompt(prompt, max_length)
            if os.path.exists(path):
                saved_prompts.append(prompt)

    logger.info(f"Found {len(saved_prompts)} saved prompt embeddings.")
    prompts = [prompt for prompt in prompts if prompt not in saved_prompts]
    
    if not prompts:
        logger.info('All prompts are already embedded. Exiting...')
        return
    
    logger.info(f"Embedding {len(prompts)} prompts...")    
    logger.info(f"Loading T5 text encoder and tokenizer from {args.pipeline_load_from} ...")
    tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.pipeline_load_from, subfolder="text_encoder").to(accelerator.device)
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        caption_tokens = tokenizer(
            batch_prompts, 
            max_length=max_length, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt",
        ).to(accelerator.device)
        
        caption_embeds = text_encoder(
            caption_tokens.input_ids, 
            attention_mask=caption_tokens.attention_mask,
        )[0]

        # Assuming you are modifying to save each prompt's data separately
        for j, prompt in enumerate(batch_prompts):
            # Extract embeddings and attention mask for the j-th item in the batch
            single_caption_emb = caption_embeds[j, :, :].unsqueeze(0)  # Add batch dimension back
            single_emb_mask = caption_tokens.attention_mask[j, :].unsqueeze(0)  # Add batch dimension back
            
            # Generate a unique path for each prompt
            save_path = get_path_for_validation_prompt(prompt, max_length)
            
            torch.save({
                'caption_embeds': single_caption_emb, 
                'emb_mask': single_emb_mask
                }, 
                save_path
            )
    flush()

def get_pipeline(
        transformer=None,
    ):
    logger.info(f"Getting pipeline {args.pipeline_load_from} ...")
    if not transformer:
        transformer = PixArtAlphaPipeline.load_transformer(args.pipeline_load_from)
                
    pipeline = PixArtAlphaPipeline.from_pretrained(
            args.pipeline_load_from,
            transformer=transformer,
            tokenizer=None,
            text_encoder=None,
            torch_dtype=weight_dtype,
            device=accelerator.device,
        )
    pipeline.set_progress_bar_config(disable=True)
    
    return pipeline

@torch.inference_mode()
def generate_images(
        pipeline,
        caption_embeds,
        batch_size,
        num_inference_steps,
        width,
        height,
        seed=0,
        guidance_scale=4.5,
    ):
    logger.info(f"Generating {len(caption_embeds)} images...")    
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)
    images = []
    # Generate images in batches
    for i in range(0, len(caption_embeds), batch_size):
        batch_caption_embeds = caption_embeds[i:i+batch_size]
        caption_embs = batch_caption_embeds['caption_embeds'].to(accelerator.device)
        emb_masks = batch_caption_embeds['emb_mask'].to(accelerator.device)

        batch_images = pipeline(
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=generator,
            guidance_scale=guidance_scale,
            prompt_embeds=caption_embs,
            prompt_attention_mask=emb_masks,
            negative_prompt=None,
            negative_prompt_embeds=uncond_prompt_embeds,
            negative_prompt_attention_mask=uncond_prompt_attention_mask,
        ).images
        images.extend(batch_images)

    return images


@torch.inference_mode()
def log_eval_images(pipeline, transformer, step):
    logger.info("Generating validation images... ")

    # transformer = accelerator.unwrap_model(model)
    # pipeline = get_pipeline(transformer=transformer)

    caption_embeds = []
    image_logs = []
    images = []
    latents = []
    for prompt in eval_sample_prompts:
        embed = torch.load(get_path_for_validation_prompt(prompt, max_length), map_location='cpu')
        caption_embeds.append(embed)

    images = generate_images(
        pipeline=pipeline,
        caption_embeds=caption_embeds,
        batch_size=config.eval.batch_size,
        num_inference_steps=config.eval.num_inference_steps,
        width=config.image_size,
        height=config.image_size,
        seed=config.eval.seed,
        guidance_scale=config.eval.guidance_scale,
    )

    flush()

    for prompt, image in zip(eval_sample_prompts, images):
        image_logs.append({"validation_prompt": prompt, "images": image})

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            import wandb
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"eval_images": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    return image_logs

@torch.inference_mode()
def log_validation_loss(model, global_step):
    if not val_dataloader:
        logger.warning("No validation data provided. Skipping validation.")
        return
    
    model.eval()
    validation_losses = []

    for batch in val_dataloader:
        if load_vae_feat:
            z = batch[0]
        else:
            with torch.cuda.amp.autocast(enabled=config.mixed_precision == 'fp16'):
                posterior = vae.encode(batch[0]).latent_dist
                if config.sample_posterior:
                    z = posterior.sample()
                else:
                    z = posterior.mode()
        latents = (z * config.scale_factor).to(weight_dtype)
        y = batch[1].squeeze(1).to(weight_dtype)
        y_mask = batch[2].squeeze(1).squeeze(1).to(weight_dtype)
        data_info = {'resolution': batch[3]['img_hw'].to(weight_dtype), 'aspect_ratio': batch[3]['aspect_ratio'].to(weight_dtype),}

        # Sample multiple timesteps for each image
        bs = latents.shape[0]
        timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=latents.device).long()

        # Predict the noise residual and compute the validation loss
        with accelerator.autocast():
            loss_term = train_diffusion.training_losses_diffusers(
                model, latents, timesteps,
                model_kwargs = dict(encoder_hidden_states=y, encoder_attention_mask=y_mask, added_cond_kwargs=data_info),
            )
            loss = loss_term['loss'].mean()
            validation_losses.append(accelerator.gather(loss).cpu().numpy())

    validation_loss = np.mean(validation_losses)
    logger.info(f"Global Step {global_step}: Validation Loss: {validation_loss:.4f}")
    accelerator.log({"validation_loss": validation_loss}, step=global_step)

    model.train()

def get_cmmd_train_and_val_samples():
    if not config.cmmd:
        logger.info("No CMMD config provided. Skipping get_cmmd_train_and_val_samples")
        return [], []
    
    train_sample_size = config.cmmd.train_sample_size
    val_sample_size = config.cmmd.val_sample_size

    # deterministically sample image-text pairs from train and val sets
    train_items = dataset.meta_data_clean
    val_items = val_dataset.meta_data_clean

    if train_sample_size > len(train_items):
        logger.warning("train_sample_size is larger than the training dataset size. Using the entire training dataset.")
        train_sample_size = len(train_items)
    if val_sample_size > len(val_items):
        logger.warning("val_sample_size is larger than the validation dataset size. Using the entire validation dataset.")
        val_sample_size = len(val_items)
    
    # deterministic sampler
    sampler = random.Random(42)
    train_items = sampler.sample(train_items, train_sample_size)
    val_items = sampler.sample(val_items, val_sample_size)
    
    return train_items, val_items


def log_cmmd(
        model, 
        global_step,
        ):
    if not config.cmmd:
        logger.warning("No CMMD data provided. Skipping CMMD calculation.")
        return
    
    train_items, val_items = get_cmmd_train_and_val_samples()

    # skip this for now. MAYBE later cache the original image embeddings
    # check if there is embeddings for the 'real' images
    # actually maybe skip this. probably doesnt take that long to just re-embed
    # embeddings_dir = os.path.join(work_dir, 'orig_image_clip_embeddings')
    # if not os.path.exists(embeddings_dir):
    #     os.makedirs(embeddings_dir)
    # train_image_embeddings_path = os.path.join(embeddings_dir, 'train.pth')
    # val_image_embeddings_path = os.path.join(embeddings_dir, 'val.pth')
    # if not os.path.exists(train_image_embeddings_path):
    #     logger.info("Generating CLIP embeddings for the 'real' images...")
    #     train_image_embeddings = get_embeddings_for_images(train_image_paths, clip_model, batch_size=16, num_workers=4, device=accelerator.device)
    #     torch.save(train_image_embeddings, train_image_embeddings_path)
    #     clip_model.unload()
    # if not, generate embeddings for the 'real' images and save

    # generate images using the text captions
    logger.info("Generating images using the text captions...")
    train_prompts = [item['prompt'] for item in train_items]
    val_prompts = [item['prompt'] for item in val_items]


    # compare cmmd between the generated images and the 'real' images
    # log cmmd value and the images with their captions

def train(model):
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    global_step = start_step + 1

    if config.eval.at_start:
        log_eval_images(model=model, global_step=global_step)

    if config.cmmd.at_start:
        pass

    # Now you train the model
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start= time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            data_time_all += time.time() - data_time_start
            if load_vae_feat:
                z = batch[0]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=config.mixed_precision == 'fp16'):
                        posterior = vae.encode(batch[0]).latent_dist
                        if config.sample_posterior:
                            z = posterior.sample()
                        else:
                            z = posterior.mode()
            latents = (z * config.scale_factor).to(weight_dtype)
            y = batch[1].squeeze(1).to(weight_dtype)
            y_mask = batch[2].squeeze(1).squeeze(1).to(weight_dtype)
            y, y_mask = token_drop(y, y_mask)   # classifier-free guidance
            data_info = {'resolution': batch[3]['img_hw'].to(weight_dtype), 'aspect_ratio': batch[3]['aspect_ratio'].to(weight_dtype),}

            # Sample a random timestep for each image
            bs = latents.shape[0]
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=latents.device).long()
            grad_norm = None
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss_term = train_diffusion.training_losses_diffusers(
                    model, latents, timesteps,
                    model_kwargs = dict(encoder_hidden_states=y, encoder_attention_mask=y_mask, added_cond_kwargs=data_info),
                )
                loss = loss_term['loss'].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()

                # if accelerator.sync_gradients:
                #     ema_update(model_ema, accelerator.unwrap_model(model), config.ema_rate)

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step - start_step)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                # avg_loss = sum(loss_buffer) / len(loss_buffer)
                log_buffer.average()
                info = f"Step/Epoch [{global_step}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, " \
                       f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}," \
                       f"s:({data_info['resolution'][0][0].item()}, {data_info['resolution'][0][1].item()}), "
                       # f"s:({data_info['resolution'][0][0].item() * relative_to_1024 // 8}, {data_info['resolution'][0][1].item() * relative_to_1024 // 8}), "
                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step)

            global_step += 1
            data_time_start= time.time()

            accelerator.wait_for_everyone()
            if config.save_model_steps and global_step % config.save_model_steps == 0:
                save_state(global_step)

        accelerator.wait_for_everyone()
        if (config.save_model_epochs and epoch % config.save_model_epochs == 0) or epoch == config.num_epochs:
            save_state(global_step)
        
        if (config.log_val_loss_epochs and epoch % config.log_val_loss_epochs == 0) or epoch == config.num_epochs:
            log_validation_loss(model=model, global_step=global_step)

        if config.eval.every_n_epochs and epoch % config.eval.every_n_epochs == 0:
            log_eval_images(model=model, global_step=global_step)

def save_state(global_step):
    if accelerator.is_main_process:
        save_path = os.path.join(os.path.join(config.work_dir, 'checkpoints'), f"checkpoint-{global_step}")
        os.umask(0o000)
        logger.info(f"Start to save state to {save_path}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to resume the training')
    parser.add_argument('--load-from', default=None, help='the dir to load a ckpt for training')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--pipeline_load_from", default='output/pretrained_models/pixart_omega_sdxl_256px_diffusers_from512', type=str, help="path for loading text_encoder, tokenizer and vae")
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-pixart-omega",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--loss_report_name", type=str, default="loss")
    args = parser.parse_args()
    return args

def validate_config(config):
    if not config.work_dir:
        raise ValueError("work_dir is not defined in the config file")
    if not config.eval:
        config.eval = {}

if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        config.work_dir = args.work_dir
    if args.cloud:
        config.data_root = '/data/data'
    if args.resume_from is not None:
        config.resume_from = args.resume_from
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 32
        config.valid_num = 100

    validate_config(config)

    load_vae_feat = config.data.load_vae_feat

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches=False,

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )

    log_name = 'train_log.log'
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(config.work_dir, log_name)):
            rename_file_with_creation_time(os.path.join(config.work_dir, log_name))
    logger = get_root_logger(os.path.join(config.work_dir, log_name))

    logger.info(accelerator.state)
    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size  # @param [256, 512, 1024]
    latent_size = int(image_size) // 8
    relative_to_1024 = float(image_size / 1024)
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma

    # Create for unconditional prompt embedding for classifier free guidance
    logger.info("Embedding for classifier free guidance")
    max_length = config.model_max_length
    uncond_prompt_embeds, uncond_prompt_attention_mask = get_null_embed(
        get_path_for_validation_prompt('', max_length), max_length=max_length
    )

    eval_sample_prompts = config.eval.prompts

    # preparing embeddings for visualization. We put it here for saving GPU memory
    if accelerator.is_main_process:
        generate_t5_prompt_embeddings()

    # build models
    train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
    model = Transformer2DModel.from_pretrained(config.load_from, subfolder="transformer").train()
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"lewei scale: {model.pos_embed.interpolation_scale} base size: {model.pos_embed.base_size}")
    # model_ema = deepcopy(model).eval()

    # 9. Handle mixed precision and device placement
    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 11. Enable optimizations
    # model.enable_xformers_memory_efficient_attention()    # not available for now

    # for name, params in model.named_parameters():
    #     if params.requires_grad == False: logger.info(f"freeze param: {name}")
    #
    # for name, params in model.named_parameters():
    #     if params.requires_grad == True: logger.info(f"trainable param: {name}")

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                transformer_ = accelerator.unwrap_model(models[0])
                # save weights in peft format to be able to load them back
                transformer_.save_pretrained(output_dir)

                for _, model in enumerate(models):
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = Transformer2DModel.from_pretrained(input_dir)
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if config.grad_checkpointing:
        model.enable_gradient_checkpointing()

    if not load_vae_feat:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained).cuda()

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    train_data = config.data
    val_data = config.val_data

    # build dataloader
    set_data_root(config.data_root)
    # logger.info(f"ratio of real user prompt: {config.real_prompt_ratio}")
    dataset = build_dataset(
        train_data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type,
        # real_prompt_ratio=config.real_prompt_ratio, 
        max_length=max_length, config=config,
    )

    val_dataset = None
    val_dataloader = None

    if config.val_data:
        val_dataset = build_dataset(
            val_data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type,
            max_length=max_length, config=config,
        )

    if config.multi_scale:
        batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
                                                batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                                ratio_nums=dataset.ratio_nums, config=config, valid_num=config.valid_num)
        # used for balanced sampling
        # batch_sampler = BalancedAspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
        #                                                 batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio,
        #                                                 ratio_nums=dataset.ratio_nums)
        train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.num_workers)

        if val_dataset:
            val_batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(val_dataset), dataset=val_dataset,
                                                batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                                ratio_nums=dataset.ratio_nums, config=config, valid_num=config.valid_num)
            val_dataloader = build_dataloader(val_dataset, batch_sampler=val_batch_sampler, num_workers=config.num_workers)

    else:
        train_dataloader = build_dataloader(dataset, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True)

    # build optimizer and lr scheduler
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer, **config.auto_lr)
    optimizer = build_optimizer(model, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    timestamp = timestamp.replace(":", "-")

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        accelerator.init_trackers(f"tb_{timestamp}_{args.tracker_project_name}")
        logger.info(f"Training tracker at tb_{timestamp}_{args.tracker_project_name}")

    start_epoch = 0
    start_step = 0
    total_steps = len(train_dataloader) * config.num_epochs

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    # model, model_ema = accelerator.prepare(model, model_ema)
    model = accelerator.prepare(model)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    if config.resume_from is not None:
        if config.resume_from != "latest":
            path = os.path.basename(config.resume_from)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(os.path.join(config.work_dir, 'checkpoints'))
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{config.resume_from}' does not exist. Starting a new training run.")
            config.resume_from = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.work_dir, 'checkpoints', path))
            start_step = int(path.split("-")[1])
            start_epoch = start_step // len(train_dataloader)

    train(model)