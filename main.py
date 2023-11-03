import argparse
import torch
import torch.backends.cudnn as cudnn
from data import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
from torch import nn
import torch.optim as optim
from model import lsr
from data import utils
import numpy as np
import copy


def main(args):
    cudnn.benchmark = True

    device = torch.device('cuda:0')
    GPU_device = torch.cuda.get_device_properties(device)
    print(f"torch {torch.__version__} device {device} ({GPU_device.name}, {GPU_device.total_memory / 1024 ** 2}MB")
    torch.manual_seed(args.seed)

    if os.path.isfile(args.train_dir):
        train_dataset = datasets.h5_train_datasets(args.train_dir)
        eval_dataset = datasets.h5_eval_datasets(args.eval_dir)
    else:
        train_dataset = datasets.my_train_datasets(args)
        eval_dataset = datasets.my_eval_datasets(args)
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=True, drop_last=True)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    model = lsr.LSR().to(device)

    if args.weight_file != "NULL":
        model.load_state_dict(torch.load(args.weight_file, map_location=device))
    print('Total params: %.2fK' % (sum(p.numel() for p in model.parameters()) / 1000.0))

    best_epoch_psnr = 0
    best_epoch_ssim = 0
    best_psnr = 0.0
    best_ssim = 0.0
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        total_loss = 0
        model.train()

        print("epoch", epoch + 1, ":")
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs).clamp(0, 1.0)
            loss = criterion(preds, labels)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

        if epoch_psnr.avg >= best_psnr:
            best_epoch_psnr = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
            if args.save_weight:
                torch.save(best_weights, os.path.join(args.outputs_dir, './best.pth'))

        if np.mean(SSIM) >= best_ssim:
            best_epoch_ssim = epoch
            best_ssim = np.mean(SSIM)

        print('loss:{:.4f}, Set5 psnr:{:.4f}, ssim:{:.4f}'.format
              (total_loss / args.batch_size, epoch_psnr.avg, best_ssim))
        print('Set5 psnr:{:.4f}, ssim:{:.4f}'.format
              (epoch_psnr.avg, best_ssim))
        print('best psnr epoch:{}, best psnr:{:.4f}'.format(best_epoch_psnr + 1, best_psnr))
        print('best ssim epoch:{}, best ssim:{:.4f}'.format(best_epoch_ssim + 1, best_ssim))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='D:/CODE/datasets/91-images_x2_aug_patch_30.h5')
    parser.add_argument('--eval_dir', type=str, default='D:/CODE/datasets/set5_x2.h5')
    parser.add_argument('--outputs_dir', type=str, default='./')
    parser.add_argument('--weight_file', type=str, default='NULL')

    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--data_aug', type=bool, default=True)
    parser.add_argument('--data_channel', type=int, default=1)
    parser.add_argument('--data_patch', type=bool, default=True)
    parser.add_argument('--data_patch_size', type=int, default=50)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)

    parser.add_argument('--save_weight', type=bool, default=False)
    args = parser.parse_args()
    main(args)





