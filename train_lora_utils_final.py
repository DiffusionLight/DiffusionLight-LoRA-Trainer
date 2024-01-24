from multiprocessing import Pool
import numpy as np
import skimage
import torch
from torch.distributions.categorical import Categorical
from PIL import Image
from envmap import EnvironmentMap, rotation_matrix

try:
    from hdrio import imread as exr_read
except:
    pass

class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(numpy_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : nd.array batch of images : [H, W, C]
        output : nd.array batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(numpy_img) * percentile) to

    def __call__(self, numpy_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_numpy_img = np.power(numpy_img, 1 / self.gamma)
        else:
            power_numpy_img = numpy_img
        non_zero = power_numpy_img > 0
        if non_zero.any():
            r_percentile = np.percentile(power_numpy_img[non_zero], self.percentile)
        else:
            r_percentile = np.percentile(power_numpy_img, self.percentile)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = np.multiply(alpha, power_numpy_img)

        if clip:
            tonemapped_img_clip = np.clip(tonemapped_img, 0, 1)

        return tonemapped_img_clip.astype('float32'), alpha, tonemapped_img

def read_image(path):
    if path.endswith(".exr") or path.endswith(".hdr"):
        image = exr_read(path)
    elif path.endswith(".npy"):
        image = np.load(path)
        # flip from bgr to rgb
        image = image[...,::-1]
    else:
        raise ValueError("Unknown file type")
    return image

def apply_auto_gamma(image):
    arr = image.copy()
    mask = arr > 0.0031308
    arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
    return arr

def hdr2ldr_envmap(
    path,
    exposure_value,
    gamma=2.4,
    auto_gamma=False,
    alpha=None,
):
    path = str(path)
    # read image
    image = read_image(path)[...,:3]
    
    # get gamma value
    to_tonemap_image = apply_auto_gamma(image.copy()) if auto_gamma else image.copy()

    # get alpha
    if alpha is None:
        hdr2ldr = TonemapHDR(gamma=gamma, percentile=99, max_mapping=0.9)
        _, alpha, _ = hdr2ldr(to_tonemap_image, gamma=not auto_gamma)
    # else:
    #     print("existing alpha")

    # apply exposure
    exposure_image = image * 2 ** exposure_value
    
    # apply gamma correction
    if auto_gamma:
        exposure_image = apply_auto_gamma(exposure_image)
    else:
        exposure_image = exposure_image ** (1/gamma)

    # apply alpha tone mapping (out = alpha*(in**(1/gamma)))
    exposure_image = alpha * exposure_image
            
    # clip to 0-1
    exposure_image = np.clip(exposure_image, 0, 1)

    # save image
    exposure_image = skimage.img_as_ubyte(exposure_image)

    return exposure_image

def get_reflection_vector_map(I: np.array, N: np.array):
    """
    UNIT-TESTED
    Args:
        I (np.array): Incoming light direction #[None,None,3]
        N (np.array): Normal map #[H,W,3]
    @return
        R (np.array): Reflection vector map #[H,W,3]
    """
    
    # R = I - 2((I⋅ N)N) #https://math.stackexchange.com/a/13263
    dot_product = (I[...,None,:] @ N[...,None])[...,0]
    R = I - 2 * dot_product * N
    return R

def cartesian_to_spherical(cartesian_coordinates):
    """Converts Cartesian coordinates to spherical coordinates.

    Args:
        cartesian_coordinates: A NumPy array of shape [..., 3], where each row
        represents a Cartesian coordinate (x, y, z).

    Returns:
        A NumPy array of shape [..., 3], where each row represents a spherical
        coordinate (r, theta, phi).
    """

    x, y, z = cartesian_coordinates[..., 0], cartesian_coordinates[..., 1], cartesian_coordinates[..., 2]
    r = np.linalg.norm(cartesian_coordinates, axis=-1)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return np.stack([r, theta, phi], axis=-1)

def get_ideal_normal_ball(size):
    
    """
    UNIT-TESTED
    BLENDER CONVENTION
    X: forward
    Y: right
    Z: up
    
    @params
        - size (int) - single value of height and width
    @return:
        - normal_map (np.array) - normal map [size, size, 3]
        - mask (np.array) - mask that make a valid normal map [size,size]
    """
    y = torch.linspace(-1, 1, size)
    z = torch.linspace(1, -1, size)
    
    #use indexing 'xy' torch match vision's homework 3
    y,z = torch.meshgrid(y, z ,indexing='xy') 
    
    x = (1 - y**2 - z**2)
    mask = x >= 0

    # clean up invalid value outsize the mask
    x = x * mask
    y = y * mask
    z = z * mask
    
    # get real z value
    x = torch.sqrt(x)
    
    # clean up normal map value outside mask 
    normal_map = torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)
    normal_map = normal_map.numpy()
    mask = mask.numpy()
    return normal_map, mask

def envmap2ball(env_map, scale=4):
    normal_ball, _ = get_ideal_normal_ball(1024)
    _, mask = get_ideal_normal_ball(256)

    # verify that x of normal is in range [0,1]
    assert normal_ball[:,:,0].min() >= 0 
    assert normal_ball[:,:,0].max() <= 1 
    
    # camera is pointing to the ball, assume that camera is othographic as it placing far-away from the ball
    I = np.array([1, 0, 0]) 
        
    ball_image = np.zeros_like(normal_ball)
    
    # read environment map 
    env_map = skimage.img_as_float(env_map)

    reflected_rays = get_reflection_vector_map(I[None,None], normal_ball)
    spherical_coords = cartesian_to_spherical(reflected_rays)
    
    theta_phi = spherical_coords[...,1:]
    
    # scale to [0, 1]
    # theta is in range [-pi, pi],
    theta_phi[...,0] = (theta_phi[...,0] + np.pi) / (np.pi * 2)
    # phi is in range [0,pi] 
    theta_phi[...,1] = theta_phi[...,1] / np.pi
    
    # mirror environment map because it from inside to outside
    theta_phi = 1.0 - theta_phi
    
    with torch.no_grad():
        # convert to torch to use grid_sample
        theta_phi = torch.from_numpy(theta_phi[None])
        env_map = torch.from_numpy(env_map[None]).permute(0,3,1,2)
        # grid sample use [-1,1] range
        grid = (theta_phi * 2.0) - 1.0
        ball_image = torch.nn.functional.grid_sample(env_map, grid, mode='bilinear', padding_mode='border', align_corners=True)
        ball_image = ball_image[0].permute(1,2,0).numpy()
        ball_image = np.clip(ball_image, 0, 1)
        ball_image = skimage.transform.resize(ball_image, (ball_image.shape[0] // scale, ball_image.shape[1] // scale), anti_aliasing=True)
        ball_image[~mask] = np.array([0,0,0])
        ball_image = skimage.img_as_ubyte(ball_image)
        # skimage.io.imsave(os.path.join(args.ball_dir, file_name), ball_image)

    return ball_image

def overlay_ball(scene, ball):
    r = 256
    x = y = (512 - r // 2)
    _, mask = get_ideal_normal_ball(size=r)

    img = np.array(scene)
    ball = np.array(ball)

    ball = (ball * mask[..., None])
    img[y:y+r, x:x+r][mask] = ball[mask]
    
    return Image.fromarray(img)

def preprocess(hdr_path, hfov_path, exposure_value):
    # precomputed alpha from FOV
    FOV = 60
    auto_gamma = False
    gamma = 2.4
    image = read_image(str(hdr_path))[...,:3]
    e = EnvironmentMap(image, 'latlong')
    dcm = rotation_matrix(azimuth=0,elevation=0,roll=0)    
    crop = e.project(
        vfov=FOV, # degrees
        rotation_matrix=dcm,
        ar=1./1.,
        resolution=(1024, 1024),
        projection="perspective",
        mode="normal"
    ) #for cropping
    # print(crop.shape)

    image = crop[...,:3]
    to_tonemap_image = apply_auto_gamma(image.copy()) if auto_gamma else image.copy()
    hdr2ldr = TonemapHDR(gamma=gamma, percentile=99, max_mapping=0.9)
    ldr, alpha, _ = hdr2ldr(to_tonemap_image, gamma=not auto_gamma)

    import skimage
    ldr = skimage.img_as_ubyte(ldr)
    skimage.io.imsave("check.png", ldr)
    
    env_map = hdr2ldr_envmap(hdr_path, exposure_value=exposure_value, alpha=alpha)
    ball = envmap2ball(env_map)
    return ball, ldr

# import os 
# import skimage

# from tqdm.auto import tqdm
# from multiprocessing import Pool

FOV = 60

input_dir = "./test_lora_scene_upsampled/ldr_roll"
output_dir = "./test_lora_scene_upsampled/ldr_roll_hfov60"

# input_dir = "./train_lora_scenes_upsampled/ldr_roll"
# output_dir = "./train_lora_scenes_upsampled/ldr_roll_hfov60"

# def crop_image(filename):
#     if not filename.endswith(".png"):
#         return None 
#     out_path = os.path.join(output_dir, filename)
#     os.makedirs(output_dir, exist_ok=True)
#     # if os.path.exists(out_path):
#     #     return None
#     env_path = os.path.join(input_dir, filename)
#     e = EnvironmentMap(env_path, 'latlong')
#     # next we can rotate on this
#     dcm = rotation_matrix(azimuth=0,elevation=0,roll=0)    
#     crop = e.project(
#         vfov=FOV, # degrees
#         rotation_matrix=dcm,
#         ar=1./1.,
#         resolution=(1024, 1024),
#         projection="perspective",
#         mode="normal"
#     ) #for cropping
#     crop = skimage.img_as_ubyte(crop)
#     skimage.io.imsave(out_path, crop)
#     return None

def get_lora_training_mask(
    batch_size=1,
    height=1024,
    width=1024,
    vae_scale_factor=8,
):
    # create 256 x 256 ball mask
    r = 256
    x = y = (512 - r // 2)
    _, ball_mask = get_ideal_normal_ball(size=r)
    ball_mask = torch.from_numpy(ball_mask)

    # create 1024 x 1024 mask
    mask = torch.zeros((height, width))
    mask[y:y+r, x:x+r][ball_mask] = True

    # downsample mask to bs x 128 x 128
    mask = torch.nn.functional.interpolate(
        mask[None, None, ...], size=(height // vae_scale_factor, width // vae_scale_factor),
        mode="bilinear", align_corners=False
    )
    mask = mask.squeeze().to(dtype=bool)
    mask = torch.repeat_interleave(mask[None, ...], 4, dim=0) # sdxl uses 4 latent channel
    mask = torch.repeat_interleave(mask[None, ...], batch_size, dim=0) # according to batch size

    return mask

def get_timestep_sampler_ramp(max_timestep=1000):
    probs = np.array(list(range(max_timestep)))
    probs = torch.from_numpy(probs / np.sum(probs))
    sampler = Categorical(probs=probs)
    return sampler

def get_timestep_sampler_invert_ramp(max_timestep=1000):
    probs = np.array(list(range(max_timestep))[::-1])
    probs = torch.from_numpy(probs / np.sum(probs))
    sampler = Categorical(probs=probs)
    return sampler

def get_timestep_sampler_largeT(interval=100):
    probs = np.ones(shape=(interval,))
    probs = torch.from_numpy(probs / np.sum(probs))
    sampler = Categorical(probs=probs)
    return sampler

def get_timestep_sampler_exponential(max_timestep=1000):
    probs = np.array(list(range(max_timestep)))
    probs = torch.from_numpy(2**(0.007 * probs))
    sampler = Categorical(probs=probs)
    return sampler