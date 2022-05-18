import os
import tqdm
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import tensorflow as tf
import torch
import torchvision.transforms as transforms
import flax
import jax
import jax.numpy as jnp

import cs
import losses
import datasets
import sde_lib
import piq
from models import utils as mutils
from utils import restore_checkpoint
from models.ema import ExponentialMovingAverage


def evaluate(config, workdir, eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Build data pipeline
  test_data_dir = {
    'ct2d_320': 'LIDC_320.npz',
    'ldct_512': 'LDCT.npz',
    'brats': 'BraTS.npz'
  }[config.data.dataset]
  test_data_dir = os.path.join('test_data', test_data_dir)
  test_imgs = np.load(test_data_dir)['all_imgs']
  test_imgs = test_imgs.reshape((jax.process_count(), -1, *test_imgs.shape[1:]))[jax.process_index()]
  mar = False

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

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
  cs_solver = cs.get_cs_solver(config, sde, score_model, sampling_shape, inverse_scaler, eps=sampling_eps)
  
  # load checkpoint to 'score_model'
  ckpt = 15
  ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
  state = restore_checkpoint(ckpt_path, state, device=config.device)
  ema.copy_to(score_model.parameters())

  hyper_params = {
    'projection': [config.sampling.coeff, config.sampling.snr],
    'langevin_projection': [config.sampling.coeff, config.sampling.snr],
    'langevin': [config.sampling.projection_sigma_rate, config.sampling.snr],
    'baseline': [config.sampling.projection_sigma_rate, config.sampling.snr]
  }[config.sampling.cs_solver]

  per_host_batch_size = config.eval.batch_size
  num_batches = int(np.ceil(len(test_imgs) / per_host_batch_size))

  # Create a circular mask
  img_size = config.data.image_size
  mask = Image.new('L', (img_size, img_size), 0)
  draw = ImageDraw.Draw(mask)
  draw.pieslice([0, 0, img_size, img_size], 0, 360, fill=255)
  toTensor = transforms.ToTensor()
  mask = toTensor(mask)[0]

  all_samples = []
  for batch in tqdm.tqdm(range(num_batches)):
    if not mar:
      current_batch = jnp.asarray(test_imgs[batch * per_host_batch_size:
                                            min((batch + 1) * per_host_batch_size,
                                                len(test_imgs))], dtype=jnp.float32) / 255.
    else:
      raise NotImplementedError("task unknown.")

    n_effective_samples = len(current_batch)
    if n_effective_samples < per_host_batch_size:
      pad_len = per_host_batch_size - len(current_batch)
      current_batch = jnp.pad(current_batch, ((0, pad_len), (0, 0), (0, 0)),
                              mode='constant', constant_values=0.)

    current_batch = current_batch.reshape((-1, *sampling_shape))
    img = scaler(current_batch)

    rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
    step_rng = jnp.asarray(step_rng)
    samples = cs_solver(step_rng, pstate, img, *hyper_params)

    samples = np.clip(np.asarray(samples), 0., 1.)
    samples = samples.reshape((-1, config.data.image_size, config.data.image_size, 1))[:n_effective_samples]
    all_samples.extend(samples)