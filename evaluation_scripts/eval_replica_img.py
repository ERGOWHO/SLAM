import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import lpips
import glob
from tqdm import tqdm
import torch
from pytorch_msssim import ms_ssim

data_path = '../splatmap/output/saverenderedimage/'

# LPIPS
lpips_model = lpips.LPIPS(net='alex')

gt_images = glob.glob(os.path.join(data_path, '265_*_gt.png'))
rendered_images = glob.glob(os.path.join(data_path, '265_*_rendered.png'))

gt_dict = {os.path.basename(gt).replace('_gt.png', ''): gt for gt in gt_images}
rendered_dict = {os.path.basename(rendered).replace('_rendered.png', ''): rendered for rendered in rendered_images}

# PSNR, MS-SSIM, LPIPS
psnr_values = []
msssim_values = []
lpips_values = []

for key in tqdm(gt_dict.keys(), desc="Processing pairs"):
    if key in rendered_dict:
        gt_image_path = gt_dict[key]
        rendered_image_path = rendered_dict[key]

        try:
            gt_image = np.array(Image.open(gt_image_path).convert('RGB'))
            rendered_image = np.array(Image.open(rendered_image_path).convert('RGB'))
        except Exception as e:
            print(f'Error opening images: {e}')
            continue

        if gt_image.shape != rendered_image.shape:
            continue

        try:
            # PSNR
            psnr_value = psnr(gt_image, rendered_image)
            psnr_values.append(psnr_value)
        except Exception as e:
            print(f'Error computing PSNR for {key}: {e}')
            continue

        try:
            # MS-SSIM
            gt_tensor = torch.tensor(gt_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            rendered_tensor = torch.tensor(rendered_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            msssim_value = ms_ssim(gt_tensor, rendered_tensor, data_range=1.0).item()
            msssim_values.append(msssim_value)
        except Exception as e:
            print(f'Error computing MS-SSIM for {key}: {e}')
            continue

        try:
            # LPIPS
            lpips_value = lpips_model(gt_tensor, rendered_tensor)
            lpips_values.append(lpips_value.item())
        except Exception as e:
            print(f'Error computing LPIPS for {key}: {e}')
            continue

average_psnr = np.mean(psnr_values) if psnr_values else float('nan')
average_msssim = np.mean(msssim_values) if msssim_values else float('nan')
average_lpips = np.mean(lpips_values) if lpips_values else float('nan')

print(f'Average PSNR: {average_psnr}')
print(f'Average MS-SSIM: {average_msssim}')
print(f'Average LPIPS: {average_lpips}')
