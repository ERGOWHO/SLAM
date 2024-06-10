import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import lpips
import glob
from tqdm import tqdm

# ????
data_path = '../splatmap/output/saverenderedimage/'

# LPIPS??
lpips_model = lpips.LPIPS(net='alex')

# ????????
gt_images = glob.glob(os.path.join(data_path, '265_*_gt.png'))
rendered_images = glob.glob(os.path.join(data_path, '265_*_rendered.png'))

# ?????????
gt_dict = {os.path.basename(gt).replace('_gt.png', ''): gt for gt in gt_images}
rendered_dict = {os.path.basename(rendered).replace('_rendered.png', ''): rendered for rendered in rendered_images}

# ??PSNR?SSIM?LPIPS??
psnr_values = []
ssim_values = []
lpips_values = []

# ?????????
for key in tqdm(gt_dict.keys(), desc="Processing pairs"):
    if key in rendered_dict:
        gt_image_path = gt_dict[key]
        rendered_image_path = rendered_dict[key]
        
        # ????
        try:
            gt_image = np.array(Image.open(gt_image_path).convert('RGB'))
            rendered_image = np.array(Image.open(rendered_image_path).convert('RGB'))
        except Exception as e:
            print(f'Error opening images: {e}')
            continue
        
        # ??????
        if gt_image.shape != rendered_image.shape:
            continue
        
        try:
            # ??PSNR
            psnr_value = psnr(gt_image, rendered_image)
            psnr_values.append(psnr_value)
        except Exception as e:
            print(f'Error computing PSNR for {key}: {e}')
            continue
        
        try:
            # ??SSIM?????
            win_size = min(gt_image.shape[0], gt_image.shape[1], 7) // 2 * 2 + 1
            # ??SSIM
            ssim_value = ssim(gt_image, rendered_image, multichannel=True, win_size=win_size, channel_axis=2)
            ssim_values.append(ssim_value)
        except Exception as e:
            print(f'Error computing SSIM for {key}: {e}')
            continue
        
        try:
            # ??LPIPS
            gt_tensor = lpips.im2tensor(gt_image)
            rendered_tensor = lpips.im2tensor(rendered_image)
            lpips_value = lpips_model(gt_tensor, rendered_tensor)
            lpips_values.append(lpips_value.item())
        except Exception as e:
            print(f'Error computing LPIPS for {key}: {e}')
            continue

# ?????
average_psnr = np.mean(psnr_values) if psnr_values else float('nan')
average_ssim = np.mean(ssim_values) if ssim_values else float('nan')
average_lpips = np.mean(lpips_values) if lpips_values else float('nan')

print(f'Average PSNR: {average_psnr}')
print(f'Average SSIM: {average_ssim}')
print(f'Average LPIPS: {average_lpips}')
