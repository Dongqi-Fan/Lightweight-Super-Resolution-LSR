import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import convert_rgb_to_y
from tqdm import tqdm
import sys


def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
        cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
    return y,cb,cr


def train(args):
    h5_file = h5py.File(args.output_path, 'w')
    if args.with_aug:
        print("---use images augment")
    else:
        print("---no images augment")
    if args.use_patch:
        print("---use images patches")
        args.patch_size = 50
    else:
        print("---no images patches")

    lr_patches = []
    hr_patches = []
    hr_list = []
    lr_list = []

    paths1 = sorted(glob.glob('{}/*'.format(args.images_dir1)))
    paths2 = sorted(glob.glob('{}/*'.format(args.images_dir2)))
    paths = paths1 + paths2
    length = len(paths1) + len(paths2)

    for step in tqdm(range(length), desc="image processing", file=sys.stdout):
        hr = pil_image.open(paths[step]).convert('RGB')
        hr_images = []

        if args.with_aug:
            for s in [1.0, 0.8, 0.6]:
                for r in [0, 90, 180]:
                    tmp = hr.resize((int(hr.width * s), int(hr.height * s)), resample=pil_image.BICUBIC)
                    tmp = tmp.rotate(r, expand=True)
                    hr_images.append(tmp)
        else:
            hr_images.append(hr)


        for hr in hr_images:
            hr_width = (hr.width // args.scale) * args.scale
            hr_height = (hr.height // args.scale) * args.scale
            hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
            lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)
            hr_list.append(hr)
            lr_list.append(lr)

            if args.use_patch:
                for i in range(0, lr.shape[0] - args.patch_size + 1, 30):
                    for j in range(0, lr.shape[1] - args.patch_size + 1, 30):
                            lr_patches.append(lr[i:i+args.patch_size, j:j+args.patch_size])
                            hr_patches.append(hr[i*args.scale:i*args.scale+args.patch_size*args.scale, j*args.scale:j*args.scale+args.patch_size*args.scale])

    if args.use_patch:
        print("writting......")
        lr_patches = np.array(lr_patches)
        hr_patches = np.array(hr_patches)

        h5_file.create_dataset('lr', data=lr_patches)
        h5_file.create_dataset('hr', data=hr_patches)

    else:
        print("writting......")
        hr_list = np.array(hr_list)
        lr_list = np.array(lr_list)

        h5_file.create_dataset('lr', data=lr_list)
        h5_file.create_dataset('hr', data=hr_list)

    print("done")
    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        #hr = hr.resize((hr.width // 2, hr.height // 2), resample=pil_image.BICUBIC)

        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)
    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir2', type=str, default='/data_nvme/XNVMSR_Group/fandongqi/T91')
    parser.add_argument('--output-path', type=str, default='./Set14_x4.h5')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--with-aug', type=bool, default=True)
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--use_patch', type=bool, default=True)
    args = parser.parse_args()

    if not args.eval:
        print("[train model]")
        train(args)
    else:
        print("[eval model]")
        eval(args)

