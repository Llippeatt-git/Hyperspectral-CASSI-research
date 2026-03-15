import importlib
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import math
import numpy as np
import torch
import scipy.io as scio
import glob
import imageio.v2 as iio
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import optics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def mask_2d_to_3d(mask, disp, cube_dims):
    H, W, B = cube_dims
    mask3d = np.zeros(cube_dims)

    for i in range(B):
        mask3d[:, :, i] = mask[:, i*disp : i*disp + W]

    return mask3d


def forward(mask3d, cube):
    # dispersion is usually nonlinear --> array

    new_cube = cube * mask3d
    new_cube = np.sum(new_cube, axis = 2)

    return new_cube

def AT_pinv_apply(y, mask, eps=1e-12):
    denom = np.sum(mask * mask, axis=2)
    inv = np.where(denom > 0, 1.0 / (denom + eps), np.zeros_like(denom))
    return mask * (y[:, :, None] * inv[:, :, None])


### training object; load in all cubes into one object
class TrainingCassiDataset(Dataset):
    def __init__(self, cubes, masks3d, patch_size=64, patches_per_scene=64):
        """
        cubes:  list of (H, W, B) numpy arrays
        masks:  list of corresponding 3D masks (H, W, B), already expanded from 2d
        """
        self.cubes = cubes
        self.masks = masks3d
        self.patch_size = patch_size
        self.patches_per_scene = patches_per_scene


    def __len__(self):
        return len(self.cubes) * self.patches_per_scene

    def __getitem__(self, idx):
        scene_idx = idx % len(self.cubes)
        cube = self.cubes[scene_idx]
        mask = self.masks[scene_idx]
        H, W, B = cube.shape
        P = self.patch_size

        # random patch
        h0 = np.random.randint(0, H - P)
        w0 = np.random.randint(0, W - P)


        cube_patch = cube[h0:h0+P, w0:w0+P, :]    
        mask3d_patch = mask[h0:h0+P, w0:w0+P, :] 
        
        measurement = forward(mask3d_patch, cube_patch)  

        x_init = AT_pinv_apply(measurement, mask3d_patch)    

        x_init = torch.tensor(x_init, dtype=torch.float32)
        target = torch.tensor(cube_patch, dtype=torch.float32)

        return x_init, target
    

### testing object; load in one cube per object
class TestingCassiCube(Dataset):
    def __init__(self, cube, mask3d, patch_size=64):
        """
        cubes:  list of (H, W, B) numpy arrays
        masks:  list of corresponding 3D masks (H, W, B), already expanded from 2d
        """

        ## Changeup ##
        # do one cube at a time. ie make a dataset per cube; this should give more control

        self.patch_size = patch_size

        self.cube = cube
        self.mask = mask3d

        H, W, _ = cube.shape
        self.x = np.arange(0, H, patch_size)
        self.y = np.arange(0, W, patch_size)


    def __len__(self):
        return self.x.shape[0] * self.y.shape[0]
    
    def __getitem__(self, idx):
        y_offset = idx // self.x.shape[0]
        x_offset = idx % self.x.shape[0]

        x = self.x[x_offset]
        y = self.y[y_offset]

        cube_patch = self.cube[x:x + self.patch_size, y:y + self.patch_size, :]        
        mask3d_patch = self.mask[x:x + self.patch_size, y:y + self.patch_size, :] 
        
        measurement = forward(mask3d_patch, cube_patch)   

        x_init = AT_pinv_apply(measurement, mask3d_patch)

        x_init = torch.tensor(x_init, dtype=torch.float32)
        posn = torch.tensor([x, y])
        target = torch.tensor(cube_patch, dtype=torch.float32)

        return x_init, posn, target
    




class encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        convd = self.block(x)
        poold = self.pool(convd)
        return poold, convd
    
class decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        upsampld = self.upsample(x)
        skipd = torch.cat([skip, upsampld], dim=1)
        return self.block(skipd)
    

class UNet(nn.Module):
    def __init__(self, in_channels=31, out_channels=31):
        super().__init__()
        self.enc1 = encoder(in_channels, 128)
        self.enc2 = encoder(128, 256)
        self.enc3 = encoder(256, 512)
        self.enc4 = encoder(512, 1024)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        self.dec4 = decoder(2048, 1024)
        self.dec3 = decoder(1024, 512)
        self.dec2 = decoder(512, 256)
        self.dec1 = decoder(256, 128)

        self.output = nn.Conv2d(128, out_channels, 3, padding=1)

    def forward(self, x):
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        b1 = self.bottleneck(x)

        x = self.dec4(b1, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        return self.output(x), b1



def train(model, loader, num_epochs=50, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = nn.MSELoss().to(device)
    
    model.train()
    all_loss = []
    for epoch in range(num_epochs):
        total_loss = 0
        for x_init, target in loader:
            x_init, target = x_init.to(device), target.to(device)

            x_init = torch.permute(x_init, (0, 3, 1 ,2))
            target = torch.permute(target, (0, 3, 1, 2))

            pred, _ = model(x_init)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        all_loss.append(avg)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg:.6f}")


    return all_loss


def test_recon(model, testloader, bounds):
    model.eval()

    loss = nn.MSELoss()
    total_loss = []

    recon = torch.zeros(bounds).permute(2, 0, 1)
    
    
    for X in testloader:
        x_init, posn, target = X
        x_init, target = x_init.to(device), target.to(device)
        
        x_init = torch.permute(x_init, (0, 3, 1 ,2))
        target = torch.permute(target, (0, 3, 1, 2))

        pred, _ = model(x_init)

        loss = loss(pred, target).item()
        total_loss.append(loss)
    
        for i in range(len(pred)):
            tile = pred[i].cpu()
            h, w = posn[i][0].item(), posn[i][1].item()
            P1 = tile.shape[1]
            P2 = tile.shape[2]
            recon[:, h:h+P1, w:w+P2] = tile    

        recon = recon.permute(1, 2, 0)
    return recon, total_loss



def draw_hpim(hpim_i: torch.tensor, lams_i:torch.tensor = None, draw:bool = False, method:str = '1931', normalize = False) -> np.ndarray:
    """
    Draw hyperspectral image in RGB format.
    hpim: hyperspectral image
    lams: wavelength of each band
    draw: whether to draw the image or just return the rgb array.
    method: '1931' or 'Gaussians'
    1931: use the CIE 1931 standard observer to convert the hyperspectral image to RGB image.
    Gaussians: use Gaussian functions to convert the hyperspectral image to RGB image.
    return rgb_img in HWC format.
    """
    if type(hpim_i) == torch.Tensor:
        hpim = torch.clone(hpim_i)
        hpim = hpim.detach().cpu().numpy()
    else:
        hpim = np.copy(hpim_i)
    if type(lams_i) == torch.Tensor:
        lams = torch.clone(lams_i)
        lams = lams.detach().cpu().numpy()
    else:
        lams = np.copy(lams_i)
    
    hpim = np.moveaxis(hpim, 2, 0)
    l,h,w = np.shape(hpim)


    if lams_i is None:
        lams = np.linspace(400, 700, num = l)
    if method == '1931':
        data = np.reshape(hpim, (l, -1), order ='C')
        CIEXYZ_1931_table = np.load('./CIEXYZ_1931_table.npy')
        CIEXYZ_1931_table[:, 1:] = CIEXYZ_1931_table[:, 1:]
        table_h = CIEXYZ_1931_table.shape[0]
        X = np.interp(lams, CIEXYZ_1931_table[:, 0], CIEXYZ_1931_table[:, 1], left = 0, right = 0)
        Y = np.interp(lams, CIEXYZ_1931_table[:, 0], CIEXYZ_1931_table[:, 2], left = 0, right = 0)
        Z = np.interp(lams, CIEXYZ_1931_table[:, 0], CIEXYZ_1931_table[:, 3], left = 0, right = 0)
        hs2xyz = np.stack([X, Y, Z], axis=1).T
        #xyz_data = hs2xyz @ data
        xyz2rgb = np.asarray([2.3706743, -0.9000405, -0.4706338, -0.5138850, 1.4253036, 0.0885814, 0.0052982, -0.0146949, 1.0093968]).reshape(3,3)
        hsi2rgb = xyz2rgb@hs2xyz / 9
        np.save('./hsi2rgb.npy', hsi2rgb)
        rgb_data = hsi2rgb @ data
        #rgb_data = rgb_data / 9
        if normalize:
            rgb_data = (rgb_data - np.min(rgb_data, axis = 1, keepdims=True)) / (np.max(rgb_data, axis = 1, keepdims=True) - np.min(rgb_data, axis = 1, keepdims=True) + 1e-16)
        rgb_img = np.reshape(rgb_data, (3, h, w), order='C')
        rgb_img = np.moveaxis(rgb_img, 0, -1)
        rgb_img = np.clip(rgb_img, 0, 1)
        if draw:
            plt.figure()
            plt.imshow(rgb_img)
            plt.axis('off')
            plt.show()
        return rgb_img
    elif method == 'Gaussians':
        R = .55*np.exp(-(lams-600)**2/2500)
        G = np.exp(-(lams-525)**2/4000)
        B = .85*np.exp(-(lams-450)**2/5000)
        wb = [0.972952272645604, 2.0642231002049565, 1.6066716258050628]
        R = R/wb[0]
        G = G/wb[1]
        B = B/wb[2]
        Rim = np.moveaxis(np.tile(R,[h,w,1]),2,0)
        Gim = np.moveaxis(np.tile(G,[h,w,1]),2,0)
        Bim = np.moveaxis(np.tile(B,[h,w,1]),2,0)
        rgb_img = np.stack([np.sum(Rim*hpim,axis=0),np.sum(Gim*hpim,axis=0),np.sum(Bim*hpim,axis=0)],axis=-1)
        rgb_img = np.clip((1/3) * rgb_img, 0,1)
        if draw:
            plt.figure()
            plt.imshow(rgb_img)
            plt.axis('off')
            plt.show()
        return rgb_img
    



### LOADING IN CAVE DATA, SPLITTING HAPPENS IN THE EXPERIMENT

def make_cube(dir_path):
    hypercube = np.zeros((512, 512, 31))
    
    dir_name = f"{dir_path}/*.png"
    band_files = sorted(glob.glob(dir_name))
    for i in range(len(band_files)):
        band_idx = band_files[i]
        img = iio.imread(band_idx)

        # imported from chat to solve weird RGB bugs
        # handle unexpected channel dimensions
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3].mean(axis=2)   # RGBA -> drop alpha, average RGB
        elif img.ndim == 3 and img.shape[2] == 3:
            img = img.mean(axis=2)              # RGB -> grayscale
        elif img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]                  # single channel with extra dim
        # if ndim == 2, already grayscale, no action needed

        ## normalize
        img = img / np.max(img)

        hypercube[:, :, i] = img

    return hypercube


def load_cave_data():
    cubes = []
    dirs_CAVE = sorted(glob.glob("complete_ms_data/*"))
    cubes = []

    for i in range(len(dirs_CAVE)):
        filep = glob.glob(f"{dirs_CAVE[i]}/*")[0]
        cubes.append(make_cube(filep))


    return cubes


def save_experiment(name, model, recons, gts, training_losses, testing_losses, hyperparameters, save_dir='outputs'):
    os.makedirs(save_dir, exist_ok=True)
    
    # unique timestamp so runs don't overwrite each other
   
    run_dir = os.path.join(save_dir, f'run_{name}')
    os.makedirs(run_dir, exist_ok=True)
    
    # save model weights
    torch.save(model.state_dict(), os.path.join(run_dir, 'model.pth'))

    # save reconstructions and ground truths
    for i, (recon, gt) in enumerate(zip(recons, gts)):
        np.save(os.path.join(run_dir, f'recon_{i}.npy'), recon.detach().numpy())
        np.save(os.path.join(run_dir, f'gt_{i}.npy'), gt)

    # save losses and hyperparameters
    np.save(os.path.join(run_dir, 'training_losses.npy'), np.array(training_losses))
    np.save(os.path.join(run_dir, 'testing_losses.npy'),  np.array(testing_losses))
    np.save(os.path.join(run_dir, 'hyperparameters.npy'), hyperparameters)

    print(f"Experiment saved to {run_dir}")
    return run_dir


def experiment(hyperparameters, save_dir):
    max_iter = hyperparameters['max_iter']
    lr = hyperparameters['lr']
    patch_size = hyperparameters['patch_size']
    dispersion = hyperparameters['dispersion']
    patches_per = hyperparameters['patches_per_scene']
    tbatch = hyperparameters['training_batch_size']
    name = hyperparameters['name']
    aperture_file = hyperparameters['aperture_file']
    pass_filter = hyperparameters['pass_filter']
    blur = hyperparameters['blur']

    data = load_cave_data()
    # assume 70/10/20 split, but for now just a couple datapoints per
    # will do train/test splitting later
    if blur:
        aperture = torch.load(aperture_file, weights_only=False, map_location=device)
        configs = {
            'device' : device,
            'min_wavelength' : 400,
            'max_wavelength' : 700,
            'wavelength_resolution' : 10,
            'generate_filter_stack' : True,
            'result_path' : './outputs/blur_experiment',
            'chromatic_focal_shift_file' : './NBK7.txt',
            'base_focal_length' : 50,
            'sampling_strategy' : 'z-axis',
            'n_camera_locs' : 2, ### come back to this
            'object_depth' : 10e6,
            'aperture_diameter' : 15,
            'pixel_size' : 5.76e-3,
            'aperture_code' : aperture, ### array that you optimize
        }

        for i in range(len(data)):
            img = data[i]

            objs = optics.defocus(img, configs)
            imp = objs[pass_filter].detach().cpu().numpy()
            imp = np.moveaxis(imp, 0, 2)

            data[i] = imp


    training, testing = train_test_split(data, test_size=0.2, shuffle=True)

    training_masks = []
    testing_masks = []

    H, W, B = training[0].shape
    for i in range(len(training)):
        #np.random.seed(i)
        mask = np.random.binomial(1, 0.5, (H, W + B * dispersion))
        training_masks.append(mask_2d_to_3d(mask, dispersion, training[i].shape))
    
    trainingdata = TrainingCassiDataset(training, training_masks, patch_size, patches_per)
    trainingloader = DataLoader(trainingdata, batch_size = tbatch)

    model = UNet().to(device)
    training_losses = train(model, trainingloader, max_iter, lr)
    

    recons, testing_losses = [], []


    for i in range(len(testing)):
        #np.random.seed(i)
        mask = np.random.binomial(1, 0.5, (H, W + B * dispersion))
        mask3d_testing = mask_2d_to_3d(mask, dispersion, testing[i].shape)
        testing_masks.append(mask3d_testing)
    
        tstcube = TestingCassiCube(testing[i], mask3d_testing, patch_size)
        tstloader = DataLoader(tstcube, batch_size=len(tstcube))

        recon, total_loss = test_recon(model, tstloader, testing[i].shape)

        recons.append(recon)
        testing_losses.append(total_loss)

    save_experiment(name, model, recons, testing, training_losses, testing_losses, hyperparameters, save_dir)
    return model, recons, testing, training_losses, testing_losses

    


if __name__ == '__main__':
    
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',                type=str,   default="test1")
    parser.add_argument('--max_iter',            type=int,   default=20)
    parser.add_argument('--lr',                  type=float, default=1e-4)
    parser.add_argument('--patch_size',          type=int,   default=64)
    parser.add_argument('--dispersion',          type=int,   default=2)
    parser.add_argument('--patches_per_scene',   type=int,   default=64)
    parser.add_argument('--training_batch_size', type=int,   default=4)
    parser.add_argument('--aperture_file',       type=str,   default='./circ.pt')
    parser.add_argument('--pass_filter',         type=int,   default=1)
    parser.add_argument('--blur',                type=bool,  default=False)
    parser.add_argument('--save_dir',            type=str,   default='outputs')
    args = parser.parse_args()

    hyperparameters = vars(args)
    experiment(hyperparameters, save_dir=args.save_dir)
    



