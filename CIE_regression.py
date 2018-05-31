import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as udata
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from dataset import (HyperDataset, process_data)
from dataset_CIE import (HyperDataset_CIE, process_data_CIE)
from utilities import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

parser = argparse.ArgumentParser(description="CIE regression")
parser.add_argument("--preprocess", type=bool, default=False, help="Whether to run prepare_data")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path log files')

opt = parser.parse_args()

def main():
    f = open('regressed_CIE.csv', 'w', newline='')
    writer = csv.writer(f)

    print("\nloading dataset ...\n")
    trainDataset = HyperDataset(crop_size=64, mode='train')
    trainLoader = udata.DataLoader(trainDataset, batch_size=opt.batchSize, shuffle=True, num_workers=4)
    testDataset = HyperDataset(crop_size=1024, mode='test')

    print("\nbuilding models ...\n")
    net = nn.Linear(in_features=31, out_features=3, bias=False)
    criterion = nn.MSELoss()

    device_ids = [0,1]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    model.load_state_dict(torch.load('net.pth'))

    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

    step = 0
    writer = SummaryWriter(opt.outf)
    for epoch in range(opt.epochs):
        for i, data in enumerate(trainLoader, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            real_hyper, real_rgb = data
            real_hyper = real_hyper.permute((0,2,3,1))
            real_rgb = real_rgb.permute((0,2,3,1))
            H = real_hyper.size(1)
            W = real_hyper.size(2)
            real_hyper, real_rgb = Variable(real_hyper.cuda()), Variable(real_rgb.cuda())
            # train
            fake_rgb = model.forward(real_hyper)
            loss = criterion(fake_rgb, real_rgb)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                model.eval()
                with torch.no_grad():
                    fake_rgb = model.forward(real_hyper)
                loss_train = criterion(fake_rgb, real_rgb).item()
                writer.add_scalar('Loss_train', loss_train, step)
                print("[epoch %d][%d/%d] Loss: %.4f" % (epoch, i, len(trainLoader), loss_train))
            step += 1
        # validate
        num = len(testDataset)
        avg_loss = 0
        for k in range(num):
            # data
            real_hyper, real_rgb = testDataset[k]
            real_hyper = torch.unsqueeze(real_hyper, 0).permute((0,2,3,1))
            real_rgb = torch.unsqueeze(real_rgb, 0).permute((0,2,3,1))
            real_hyper, real_rgb = Variable(real_hyper.cuda()), Variable(real_rgb.cuda())
            # forward
            with torch.no_grad():
                fake_rgb = model.forward(real_hyper)
            avg_loss += criterion(fake_rgb, real_rgb).item()
        writer.add_scalar('Loss_val', avg_loss/num, avg_loss/num)
        print("[epoch %d] Validation Loss: %.4f" % (epoch, avg_loss/num))

        for param in model.parameters():
            writer.writerows(param.data.cpu().numpy().T)
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))

if __name__ == "__main__":
    if opt.preprocess:
        process_data(patch_size=64, stride=40, path='NTIRE2018', mode='train')
        process_data(patch_size=64, stride=40, path='NTIRE2018', mode='test')
    main()
