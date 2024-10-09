import torch
from transformer_maskgit import CTViT
import numpy as np
import nibabel as nib
import argparse
import torch.nn.functional as F


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

def nii_img_to_tensor(path, slope, intercept, xy_spacing, z_spacing):
    nii_img = nib.load(str(path))
    img_data = nii_img.get_fdata()

    # Define the target spacing values
    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5

    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    img_data = slope * img_data + intercept
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)

    img_data = img_data.transpose(2, 0, 1)

    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    img_data = resize_array(tensor, current, target)
    img_data = img_data[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))

    img_data = (((img_data) / 1000)).astype(np.float32)
    slices = []

    tensor = torch.tensor(img_data)
    # Get the dimensions of the input tensor
    target_shape = (480, 480, 240)

    # Extract dimensions
    h, w, d = tensor.shape

    # Calculate cropping/padding values for height, width, and depth
    dh, dw, dd = target_shape
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    # Crop or pad the tensor
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before

    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before

    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

    tensor = tensor.permute(2, 0, 1)

    tensor = tensor.unsqueeze(0)

    return tensor.cuda()

def main():
    parser = argparse.ArgumentParser(description='Process NIfTI image and encode it using a transformer model.')

    parser.add_argument('--path', type=str, required=True, help='Path to the NIfTI image file.')
    parser.add_argument('--slope', type=float, default=1, help='Slope for rescaling the image.')
    parser.add_argument('--intercept', type=float, default=0, help='Intercept for rescaling the image.')
    parser.add_argument('--xy_spacing', type=float, default=1, help='XY spacing of the image.')
    parser.add_argument('--z_spacing', type=float, default=1, help='Z spacing of the image.')

    args = parser.parse_args()

    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8
    ).cuda().eval()

    image_encoder.load("./CT_CLIP_encoder/clip_visual_encoder.pth")

    image = nii_img_to_tensor(path=args.path, slope=args.slope, intercept=args.intercept, xy_spacing=args.xy_spacing, z_spacing=args.z_spacing)

    image_encoded = image_encoder(image.unsqueeze(0), return_encoded_tokens=True)

    image_name = args.path.split("/")[-1].split(".")[0]
    np.savez(f'./embeddings/{image_name}.npz', arr=image_encoded.cpu().detach().numpy())

if __name__ == '__main__':
    main()