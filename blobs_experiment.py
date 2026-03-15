import torch
import CASSI_net
import importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

import importlib
import torch.nn as nn
import torch
import scipy.io as scio
import glob
import imageio.v2 as iio
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def generate_nonoverlapping_blobs(
    H=128,
    W=128,
    n_blobs=12,
    sample_spectra=None,
    radius_range=(6, 14),
    max_tries=5000,
    replace=True,
    blob_indices=None,
    background_spectrum=None,
    seed=None,
):
    """
    Generate a hyperspectral image with non-overlapping circular blobs,
    where each blob is assigned a spectrum sampled from `sample_spectra`.

    Parameters
    ----------
    H, W : int
        Spatial size.
    n_blobs : int
        Number of blobs to place.
    sample_spectra : ndarray, shape (N, C)
        Bank of candidate spectra. Each row is one spectrum.
    radius_range : tuple(int, int)
        Min/max blob radius.
    max_tries : int
        Maximum attempts for blob placement.
    replace : bool
        Whether spectra can be reused across blobs.
    blob_indices : ndarray or list, optional
        Explicit indices into `sample_spectra` for each blob.
        If provided, overrides random spectral sampling.
    background_spectrum : ndarray, shape (C,), optional
        Spectrum assigned to background. If None, background is zero.
    seed : int, optional
        Random seed.

    Returns
    -------
    hsi : ndarray, shape (C, H, W)
        Generated hyperspectral cube.
    label_map : ndarray, shape (H, W)
        Blob labels. -1 means background.
    centers : list of tuple
        Blob centers [(cy, cx), ...].
    radii : list of int
        Blob radii.
    assigned_spectra : ndarray, shape (K, C)
        Spectrum assigned to each placed blob.
    chosen_indices : ndarray, shape (K,)
        Indices from `sample_spectra` used for each blob.
    """
    rng = np.random.default_rng(seed)

    if sample_spectra is None:
        raise ValueError("You must provide `sample_spectra` with shape (N, C).")

    sample_spectra = np.asarray(sample_spectra, dtype=np.float32)
    if sample_spectra.ndim != 2:
        raise ValueError("`sample_spectra` must have shape (N, C).")

    N, C = sample_spectra.shape

    if blob_indices is not None:
        blob_indices = np.asarray(blob_indices)
        if blob_indices.ndim != 1:
            raise ValueError("`blob_indices` must be a 1D array/list.")
        n_blobs = len(blob_indices)
        if np.any(blob_indices < 0) or np.any(blob_indices >= N):
            raise ValueError("`blob_indices` contains invalid indices.")
    else:
        if (not replace) and (n_blobs > N):
            raise ValueError(
                f"Requested {n_blobs} blobs but only {N} spectra are available with replace=False."
            )

    label_map = -np.ones((H, W), dtype=np.int32)
    yy, xx = np.mgrid[0:H, 0:W]

    centers = []
    radii = []

    placed = 0
    tries = 0

    while placed < n_blobs and tries < max_tries:
        tries += 1

        r = rng.integers(radius_range[0], radius_range[1] + 1)
        cy = rng.integers(r, H - r)
        cx = rng.integers(r, W - r)

        valid = True
        for (py, px), pr in zip(centers, radii):
            dist2 = (cy - py) ** 2 + (cx - px) ** 2
            min_dist = r + pr
            if dist2 < min_dist ** 2:
                valid = False
                break

        if not valid:
            continue

        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        label_map[mask] = placed
        centers.append((cy, cx))
        radii.append(r)
        placed += 1

    if placed < n_blobs:
        print(f"Warning: only placed {placed}/{n_blobs} blobs.")
        n_blobs = placed
        label_map[label_map >= n_blobs] = -1

    # Choose spectra
    if blob_indices is not None:
        chosen_indices = blob_indices[:n_blobs]
    else:
        chosen_indices = rng.choice(N, size=n_blobs, replace=replace)

    assigned_spectra = sample_spectra[chosen_indices]  # (K, C)

    # Build HSI
    hsi = np.zeros((C, H, W), dtype=np.float32)

    if background_spectrum is not None:
        background_spectrum = np.asarray(background_spectrum, dtype=np.float32)
        if background_spectrum.shape != (C,):
            raise ValueError(
                f"`background_spectrum` must have shape ({C},), got {background_spectrum.shape}."
            )
        hsi[:] = background_spectrum[:, None, None]

    for k in range(n_blobs):
        mask = (label_map == k)
        hsi[:, mask] = assigned_spectra[k][:, None]

    return hsi, label_map, centers, radii, assigned_spectra, chosen_indices



def blob_experiment(model_path, spectra_path, hyperparameters, save_dir='outputs/blob_experiment'):
    os.makedirs(save_dir, exist_ok=True)

    N              = hyperparameters['N']
    blob_rad_little = hyperparameters['blob_radii_small']
    blob_rad_bigs = hyperparameters['blob_radii_big']
    dispersion = hyperparameters['dispersion']

    model = CASSI_net.UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    outputgt4 = np.load(spectra_path)
    lams = np.arange(400, 701, 10)

    # sample spectra and generate blobs
    randx = np.random.randint(0, 512, N)
    randy = np.random.randint(0, 512, N)
    sample_spectra = np.zeros((N, 31))
    for i in range(N):
        sample_spectra[i, :] = outputgt4[randx[i], randy[i], :]

    hsi, label_map, centers, radii, assigned_spectra, chosen_inds = \
        generate_nonoverlapping_blobs(512, 512, N, sample_spectra, (blob_rad_little, blob_rad_bigs))
    hsi = np.moveaxis(hsi, 0, 2)

    # forward + reconstruction
    H, W, B    = hsi.shape
    mask       = np.random.binomial(1, 0.5, (H, W + B * dispersion))
    mask3d     = CASSI_net.mask_2d_to_3d(mask, dispersion, hsi.shape)
    tstcube    = CASSI_net.TestingCassiCube(hsi, mask3d, 64)
    tstloader  = DataLoader(tstcube, batch_size=len(tstcube))
    recon, total_loss = CASSI_net.test_recon(model, tstloader, hsi.shape)

    # ── Save arrays ───────────────────────────────────────────────────────────
    np.save(os.path.join(save_dir, 'hsi_gt.npy'),       hsi)
    np.save(os.path.join(save_dir, 'recon.npy'),         recon.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'label_map.npy'),     label_map)
    np.save(os.path.join(save_dir, 'sample_spectra.npy'), sample_spectra)
    np.save(os.path.join(save_dir, 'losses.npy'),        np.array(total_loss))
    np.save(os.path.join(save_dir, 'hyperparameters.npy'), hyperparameters)

    # ── Save figures ──────────────────────────────────────────────────────────
    def save_hpim(cube, lams, filename):
        fig = CASSI_net.draw_hpim(cube, lams, draw=True)  # assumes draw=False returns fig
        plt.imshow(fig)
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    save_hpim(outputgt4, lams, 'spectra_source.png')
    save_hpim(hsi,       lams, 'hsi_gt.png')
    save_hpim(recon,     lams, 'recon.png')

    # ── Save label map ────────────────────────────────────────────────────────
    plt.figure(figsize=(6, 6))
    plt.imshow(label_map, cmap='tab20')
    plt.title('Blob Label Map')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'label_map.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Experiment saved to {save_dir}")
    return hsi, recon, total_loss, label_map


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N',                  type=int,   default=120)
    parser.add_argument('--dispersion',          type=int,   default=1)
    parser.add_argument('--blob_radii_small',   type=int,   default=12)
    parser.add_argument('--blob_radii_big',      type=int,   default=28)
    args = parser.parse_args()

    hyperparameters = vars(args)
    blob_experiment('./outputs/run_test_simple_big_iter/model.pth', './outputs/run_test_simple_big_iter/gt_4.npy', hyperparameters)