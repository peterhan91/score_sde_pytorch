import os
import tqdm
import imageio
import logging
import numpy as np
from absl import flags

import torch
import tensorflow as tf

import cs
import losses
import datasets
import sde_lib
from models import utils as mutils
from utils import restore_checkpoint
from models.ema import ExponentialMovingAverage

FLAGS = flags.FLAGS

def eval_sample(config, workdir, ckpt_name, eval_folder="eval", mar=False):
  """Evaluate trained models by draw samples from it.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(os.path.join(eval_dir, 'y_true'))
  tf.io.gfile.makedirs(os.path.join(eval_dir, 'y_pred'))
  tf.io.gfile.makedirs(os.path.join(eval_dir, 'total'))

  # Build data pipeline
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                            uniform_dequantization=config.data.uniform_dequantization,
                                            evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)                   # Rescale data from [0, 1] to [-1, 1]
  inverse_scaler = datasets.get_data_inverse_scaler(config)   # Scale back from [-1, 1] to [0, 1]

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")
  
  # Build the sampling function
  sampling_shape = (config.eval.batch_size,
                    config.data.image_size, config.data.image_size,
                    config.data.num_channels)
  cs_solver = cs.get_cs_solver(config, sde, sampling_shape, inverse_scaler, eps=sampling_eps)
  
  # load checkpoint to 'score_model'
  ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
  state = restore_checkpoint(ckpt_path, state, device=config.device)
  ema.copy_to(score_model.parameters())

  # prepare sampling hyperparameters 
  hyper_params = {
    'projection': [config.sampling.coeff, config.sampling.snr],
    'langevin_projection': [config.sampling.coeff, config.sampling.snr],
    'langevin': [config.sampling.projection_sigma_rate, config.sampling.snr],
    'baseline': [config.sampling.projection_sigma_rate, config.sampling.snr]
  }[config.sampling.cs_solver]

  all_samples = []
  all_imgs = []
  all_undersamples = []
  all_masks = []
  
  eval_iter = iter(eval_ds) 
  for r, batch in tqdm.tqdm(enumerate(eval_iter)):
    logging.info("sampling -- ckpt: %s, round: %d" % (ckpt_name, r))
    if not mar:
      current_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
      current_batch = current_batch.permute(0, 3, 1, 2)
      current_batch = current_batch.reshape((-1, *sampling_shape))
      img = scaler(current_batch)
    else:
      raise NotImplementedError("task unknown.")

    samples, mask, unders = cs_solver(score_model, img, *hyper_params) # masks generated inside the cs_solver
    
    samples = np.clip(np.asarray(samples), 0., 1.)
    samples = samples.reshape((-1, config.data.image_size, config.data.image_size, 1))
    unders = np.clip(np.asarray(unders), 0., 1.)
    unders = unders.reshape((-1, config.data.image_size, config.data.image_size, 1))
    img = inverse_scaler(img).cpu().detach().numpy()
    img = np.clip(np.asarray(img), 0., 1.)
    img = img.reshape((-1, config.data.image_size, config.data.image_size, 1))
    mask = mask.reshape((-1, config.data.image_size, config.data.image_size, 1))

    all_samples.extend(samples)
    all_imgs.extend(img)
    all_masks.extend(mask)
    all_undersamples.extend(unders)

  
  for n in range(len(all_samples)):
    # save original, undersampled, reconstructed imgs, and mask
    # also, save original and reconstructed into seperate folder for PSNR and SSIM computation
    sample = all_samples[n]
    sample  = np.clip(np.rint(sample * 255.0), 0.0, 255.0).astype(np.uint8).squeeze() # [-1,1] => [0,255]
    img = all_imgs[n]
    img = np.clip(np.rint(img * 255.0), 0.0, 255.0).astype(np.uint8).squeeze()
    
    imageio.imwrite(os.path.join(eval_dir, 'y_true', str(n)+'.jpg'))
    imageio.imwrite(os.path.join(eval_dir, 'y_pred', str(n)+'.jpg'))

    mask = all_masks[n]
    mask = np.clip(np.rint(mask * 255.0), 0.0, 255.0).astype(np.uint8).squeeze()
    undersampled = all_undersamples[n]
    undersampled = np.clip(np.rint(undersampled * 255.0), 0.0, 255.0).astype(np.uint8).squeeze()
    total = np.hstack((mask, undersampled, sample, img))
    
    imageio.imwrite(os.path.join(eval_dir, 'total', str(n)+'.jpg'), total)