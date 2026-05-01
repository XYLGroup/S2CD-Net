import os
from os import listdir
from fvcore.nn import parameter_count
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from option import opt
from data_utils import is_image_file
from model import S2CD_Net
import scipy.io as scio
from profile import profile_macs
from eval import HSI_calculate_psnr, HSI_calculate_ssim, HSI_calculate_sam, HSI_calculate_ergas, HSI_calculate_rmse, \
    HSI_calculate_cc


def main():
    input_path = '/home/descfly/zly/S2CD-Net/datasets/{}/mcodes/dataset/{}_x{}/test/'.format(opt.datasetName, opt.datasetName, opt.upscale_factor)
    out_path = '/home/descfly/zly/S2CD-Net/result/' + opt.datasetName + '/' + str(opt.upscale_factor) + '/' + str(opt.modelName) + '/'

    PSNRs = []
    SSIMs = []
    SAMs = []
    ERGASs = []
    RMSEs = []
    CCs = []

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if opt.cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    model = S2CD_Net(inch=opt.inch,
                            dim=256,
                            upscale=opt.upscale_factor,
                            d_state=16,
                            inner_rank=64,
                            num_tokens=128,
                            mlp_ratio=2.0,
                            n_sample=64,
                            lamuda=1.11)
        

    if opt.cuda:
        model = nn.DataParallel(model).cuda()

    checkpoint = torch.load(opt.model_name)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    images_name = [x for x in listdir(input_path) if is_image_file(x)]
    total_time = 0

    for index in range(len(images_name)):
        print('Current data index is {}'.format(index))

        mat = scio.loadmat(input_path + images_name[index])
        hyperLR = mat['lq'].transpose(2, 0, 1).astype(np.float32)
        bic = mat['bic'].transpose(2, 0, 1).astype(np.float32)
        bic = Variable(torch.from_numpy(bic).float(), volatile=True).contiguous().view(1, -1, bic.shape[1],
                                                                                       bic.shape[2])
        input = Variable(torch.from_numpy(hyperLR).float(), volatile=True).contiguous().view(1, -1, hyperLR.shape[1],
                                                                                             hyperLR.shape[2])
        if opt.cuda:
            input = input.cuda()

        # 3. times
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        with torch.no_grad():
            output = model(input, save_omega=True, image_name='image_{}'.format(index))
        torch.cuda.synchronize()
        total_time += time.time() - start

        HR = mat['gt'].transpose(2, 0, 1).astype(np.float32)
        SR = output.cpu().data[0].numpy().astype(np.float32)
        SR[SR < 0] = 0
        SR[SR > 1.] = 1.
        psnr = HSI_calculate_psnr(SR, HR, 0)
        ssim = HSI_calculate_ssim(SR, HR, 0)
        sam = HSI_calculate_sam(SR, HR, 0)
        ergas = HSI_calculate_ergas(SR, HR, 0)
        rmse = HSI_calculate_rmse(SR, HR, 0)
        cc = HSI_calculate_cc(SR, HR, 0)

        PSNRs.append(psnr)
        SSIMs.append(ssim)
        SAMs.append(sam)
        ERGASs.append(ergas)
        RMSEs.append(rmse)
        CCs.append(cc)

        SR = SR.transpose(1, 2, 0)
        HR = HR.transpose(1, 2, 0)

        scio.savemat(out_path + images_name[index], {'gt': HR, 'img': SR})
        # import matplotlib.pyplot as plt
        # rgb = SR[:, :, [27, 11, 5]] 
        # base_name = os.path.splitext(images_name[index])[0]  
        # out_file = os.path.join(out_path, base_name + '.png') 
        # plt.imsave(out_file, rgb)

    avg_time = total_time / (index + 1)  # s
    fps = 1 / avg_time  # frames/s
    max_mem = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB

    compute_metrics = {"FPS": fps,
                       "AvgTime(s)": avg_time,
                       "MaxMem(MB)": max_mem}
    print(
        "=====averPSNR/SSIM/SAM/ERGAS:{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}".format(np.mean(PSNRs), np.mean(SSIMs),
                                                                                        np.mean(SAMs), np.mean(ERGASs),
                                                                                        np.mean(RMSEs), np.mean(CCs)))
    print("======================================")
    for k, v in compute_metrics.items():
        print(f"{k:10s}: {v:.4f}")
    print("======================================")


if __name__ == "__main__":
    main()
