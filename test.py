import argparse
import torch
from data import datasets
from torch.utils.data import DataLoader
import os
from model import lsr
from data import utils
import numpy as np


def main(args):
    device = 'cpu'
    if os.path.isfile(args.eval_dir):
        eval_dataset = datasets.h5_eval_datasets(args.eval_dir)
    else:
        eval_dataset = datasets.h5_eval_datasets(args)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    weight_file = torch.load(args.weight_file, map_location=device)
    model = lsr.LSR().to(device)
    model.load_state_dict(weight_file, strict=True)
    print('Total params: %.2fK' % (sum(p.numel() for p in model.parameters()) / 1000.0))

    model.eval()
    epoch_psnr = utils.AverageMeter()
    SSIM = []
    for data in eval_dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0, 1.0)

        output = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        SSIM_label = labels.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        SSIM.append(utils.calculate_ssim(output, SSIM_label))
        epoch_psnr.update(utils.calc_psnr(preds, labels), len(inputs))
    print('Set5 psnr:{:.4f}, ssim:{:.4f}'.format(epoch_psnr.avg, np.mean(SSIM)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', type=str, default='./set5_x2.h5')
    parser.add_argument('--weight_file', type=str, default='./set5_x2.pth')
    args = parser.parse_args()
    main(args)


