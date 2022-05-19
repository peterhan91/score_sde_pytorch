
import functools
import numpy as np

import torch
import torch.fft as fft

from utils import batch_mul
from models import utils as mutils
from sampling import get_predictor, get_corrector, shared_predictor_update_fn, shared_corrector_update_fn



def get_cartesian_mask(shape, n_keep=30):
  # shape [Tuple]: (H, W)
  size = shape[0]
  center_fraction = n_keep / 1000
  acceleration = size / n_keep

  num_rows, num_cols = shape[0], shape[1]
  num_low_freqs = int(round(num_cols * center_fraction))

  # create the mask
  mask = np.zeros((num_rows, num_cols), dtype=np.float32)
  pad = (num_cols - num_low_freqs + 1) // 2
  mask[:, pad: pad + num_low_freqs] = True

  # determine acceleration rate by adjusting for the number of low frequencies
  adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
      num_low_freqs * acceleration - num_cols
  )

  offset = round(adjusted_accel) // 2

  accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
  accel_samples = np.around(accel_samples).astype(np.uint32)
  mask[:, accel_samples] = True

  return mask


def get_cartesian_mask_coordinates(size, n_keep):
  i, j = np.nonzero(get_cartesian_mask((size, size), n_keep))
  y_grid = i.reshape((n_keep, -1))
  x_grid = j.reshape((n_keep, -1))
  return x_grid, y_grid


def get_kspace(img, axes):
  # img should be a complex tensor
  shape = img.shape[axes[0]]
  return fft.fftshift(
    fft.fftn(fft.ifftshift(
      img, dim=axes
    ), dim=axes),
    dim=axes
  ) / shape


def kspace_to_image(kspace, axes):
  shape = kspace.shape[axes[0]]
  return fft.fftshift(
    fft.ifftn(fft.ifftshift(
      kspace, dim=axes
    ), dim=axes),
    dim=axes
  ) * shape


def get_masks(config):
  if config.sampling.task == 'mri':
    mask = get_cartesian_mask((config.data.image_size, config.data.image_size), n_keep=config.sampling.n_projections)
    mask = mask[None, :, :, None].astype(np.float32)
    return torch.from_numpy(mask)
  else:
    raise ValueError(f"task {config.sampling.mask} not recognized.")


def get_known(config, img):
  if config.sampling.task == 'mri':
    return get_kspace(img, axes=(1, 2))

  else:
    raise ValueError(f"task {config.sampling.mask} not recognized.")


def merge_known_with_mask(config, x_space, known, mask, coeff=1.):
  if config.sampling.task == 'mri':
    return known * mask * coeff + x_space * (1. - mask * coeff)
  else:
    raise ValueError(f"task {config.sampling.mask} not recognized.")


def get_cs_solver(config, sde, shape, inverse_scaler, eps=1e-5):
  cs_solver = config.sampling.cs_solver
  # Probability flow ODE sampling with black-box ODE solvers
  predictor = get_predictor(config.sampling.predictor.lower())
  corrector = get_corrector(config.sampling.corrector.lower())

  if cs_solver.lower() == 'projection':
    sampling_fn = get_projection_sampler(config, sde, shape, predictor, corrector,
                                         inverse_scaler,
                                         n_steps=config.sampling.n_steps_each,
                                         probability_flow=config.sampling.probability_flow,
                                         continuous=config.training.continuous,
                                         denoise=config.sampling.noise_removal,
                                         eps=eps)
  else:
    raise ValueError(f"CS solver name {cs_solver} unknown.")
  return sampling_fn


def get_projection_sampler(config, sde, shape, predictor, corrector,
                           inverse_scaler, n_steps=1,
                           probability_flow=False, continuous=True,
                           denoise=True, eps=1e-5, device='cuda'):
  if config.sampling.task == 'mri':
      to_space = lambda x: get_kspace(x, (1, 2))
      from_space = lambda x: kspace_to_image(x, (1, 2)).real
  else:
      raise ValueError(f'Task {config.sampling.task} not recognized.')

  def get_inpaint_update_fn(update_fn):
    def inpaint_update_fn(model, x, t, mask, known, coeff):
      with torch.no_grad():
          x_space = to_space(x)
          mean, std = sde.marginal_prob(known, t)
          noise = torch.randn_like(x)
          noise_space = to_space(noise)
          noisy_known = mean + batch_mul(std, noise_space)
          
          x_space = merge_known_with_mask(config, x_space, noisy_known, mask, coeff)
          x = from_space(x_space)
          
          x, _ = update_fn(x, t, model=model)
          return x
    return inpaint_update_fn   

  def projection_sampler(model, img, coeff, snr):
    with torch.no_grad():
      x = sde.prior_sampling(shape).to(device)
      mask = get_masks(config).to(device)
      known = get_known(config, img).to(device)

      predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                              sde=sde,
                                              predictor=predictor,
                                              probability_flow=probability_flow,
                                              continuous=continuous)
      corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                              sde=sde,
                                              corrector=corrector,
                                              continuous=continuous,
                                              snr=snr,
                                              n_steps=n_steps)

      cs_predictor_update_fn = get_inpaint_update_fn(predictor_update_fn)
      cs_corrector_update_fn = get_inpaint_update_fn(corrector_update_fn)

      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

      def loop_body(x, i):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        x = cs_corrector_update_fn(model, x, vec_t, mask, known, coeff)
        x = cs_predictor_update_fn(model, x, vec_t, mask, known, coeff)
        output = x
        return x, output
      
      all_samples = []
      for i in range(sde.N):
        x, all_sample = loop_body(x, i)
        all_samples.append(all_sample)
      output = all_samples[-1]    

      if denoise:
        t_eps = torch.full((output.shape[0],), eps)
        k, std = sde.marginal_prob(torch.ones_like(output), t_eps) # somehow both args are not specifiy 'device'
        score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous, return_state=False)
        score = score_fn(output, t_eps)
        output = output / k + batch_mul(std ** 2, score / k)
        output_space = to_space(output)
        output_space = merge_known_with_mask(config, output_space, known, mask, 1.)
        output = from_space(output_space)
      
      return inverse_scaler(output), mask.cpu().detach().numpy(), inverse_scaler(from_space(known)).cpu().detach().numpy()

  return projection_sampler