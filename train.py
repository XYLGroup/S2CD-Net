import os
import torch
torch.autograd.set_detect_anomaly(True)
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from option import opt
import random
import numpy as np
from model import S2CD_Net
from data_utils import TrainsetFromFolder, ValsetFromFolder
from eval import HSI_calculate_psnr, HSI_calculate_ssim, HSI_calculate_sam, HSI_calculate_ergas
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class MetricBoard:

    def __init__(self, log_dir='runs/exp'):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def update(self, step, loss, psnr):

        self.writer.add_scalar('Loss', loss, step)
        self.writer.add_scalar('PSNR', psnr, step)

    def close(self):
        self.writer.close()

def main():

    if opt.show:
        if not os.path.exists("logs/"):
            os.makedirs("logs/")
        
        global writer
        writer = SummaryWriter(log_dir='logs') 
       
    if opt.cuda:
        print("=> Use GPU ID: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    random.seed(opt.seed) 
    np.random.seed(opt.seed) 
    torch.manual_seed(opt.seed)
    if opt.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)  
        torch.cuda.manual_seed_all(opt.seed)  
        cudnn.deterministic = True
    
    # Loading datasets
    train_set = TrainsetFromFolder('/home/descfly/zly/S2CD-Net/datasets/{}/mcodes/dataset/{}_x{}/train'.format(opt.datasetName, opt.datasetName,
                                                                           opt.upscale_factor))
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    val_set = ValsetFromFolder('/home/descfly/zly/S2CD-Net/datasets/{}/mcodes/dataset/{}_x{}/test'.format(opt.datasetName, opt.datasetName,
                                                                          opt.upscale_factor))
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    # Buliding model
    model = S2CD_Net(inch=opt.inch,
                        dim=256,
                        upscale=opt.upscale_factor,
                        d_state=16,
                        inner_rank=64,
                        num_tokens=128,
                        mlp_ratio=2.0,
                        n_sample=64,
                        lamuda=1.11).cuda()


    criterion = nn.L1Loss() 
    
    if opt.cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu()   
    print('# parameters:', sum(param.numel() for param in model.parameters())) 
                   
    # Setting Optimizer
    optimizer = optim.Adam(model.parameters(),  lr=opt.lr,  betas=(0.9, 0.999), eps=1e-08)    

    # optionally resuming from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)         
            opt.start_epoch = checkpoint['epoch'] + 1 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=>te no checkpoint found at '{}'".format(opt.resume))

    # Setting learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.nEpochs, eta_min=1e-5)

    # Training
    mb = MetricBoard('runs/S2CD_{}_{}_X{}'.format(opt.modelName, opt.datasetName, opt.upscale_factor))
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        scheduler.step()
        print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"])) 
        loss = train(train_loader, optimizer, model, criterion, epoch)
        psnr = val(val_loader, model)
        mb.update(epoch, loss, psnr)
        if epoch % 10 == 0:
            save_checkpoint(epoch, model, optimizer)

    mb.close()


def train(train_loader, optimizer, model, criterion, epoch):
	
    model.train()
    max_norm = 1.0
      
    for iteration, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"), 1):
        input, label = Variable(batch[0]),  Variable(batch[1], requires_grad=False)

        # if opt.cuda:
        input = input.cuda()
        label = label.cuda()

        SR = model(input)            

        loss = criterion(SR, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()         

        if iteration % 100 == 0:
            tqdm.write("===> Epoch[{}]({}/{}): Loss: {:.10f}"
                       .format(epoch, iteration, len(train_loader), loss.item()))

        LOSS = loss.item()

    return LOSS


def val(val_loader, model):
	            
    model.eval()
    torch.cuda.empty_cache()
    val_psnr = 0

    for iteration, batch in enumerate(val_loader, 1):
        input, HR = Variable(batch[0], volatile=True),  Variable(batch[1])

        # if opt.cuda:
        input = input.cuda()
        HR = HR.cuda()

        with torch.no_grad():
            SR = model(input)

        val_psnr += HSI_calculate_psnr(SR.cpu().data[0].numpy(), HR.cpu().data[0].numpy(), 0)
    val_psnr = val_psnr / len(val_loader) 
    print("PSNR = {:.3f}".format(val_psnr))
    return val_psnr

    
def save_checkpoint(epoch, model, optimizer):
    model_out_path = "checkpoint/" + "{}_{}_{}_epoch_{}.pth".format(opt.modelName, opt.datasetName, opt.upscale_factor, epoch)
    state = {"epoch": epoch , "model": model.state_dict(), "optimizer":optimizer.state_dict()}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")     	
    torch.save(state, model_out_path)
 
          
if __name__ == "__main__":
    main()

