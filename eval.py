from functools import partial
import numpy as np
import torch
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-3]
        bwindex = []
        for ch in range(C):
            x = torch.squeeze(X[..., ch, :, :].data).cpu().numpy()
            y = torch.squeeze(Y[..., ch, :, :].data).cpu().numpy()
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex


cal_bwssim = Bandwise(partial(compare_ssim, data_range=1))
cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=1))


def cal_sam(X, Y, eps=1e-8):
    X = torch.squeeze(X.data).cpu().numpy()
    Y = torch.squeeze(Y.data).cpu().numpy()
    tmp = np.sum(X * Y, axis=0) / (np.sqrt(np.sum(X ** 2, axis=0)) * np.sqrt(np.sum(Y ** 2, axis=0)) + eps)
    cos_sim = np.clip(tmp, -1.0, 1.0)
    return np.mean(np.real(np.arccos(cos_sim))) * 180 / np.pi


def cal_ergas(X, Y):
    if len(X.shape) == 4:
        X = X[None, ...]
        Y = Y[None, ...]
    # Metric = iv.spectra_metric(Y[0, 0, ...].permute(1,2,0).detach().cpu().numpy(), X[0, 0, ...].permute(1,2,0).detach().cpu().numpy(),
    # scale=1)
    # ERGAS = Metric.ERGAS()

    ergas = 0
    for i in range(X.size(2)):
        ergas = ergas + torch.nn.functional.mse_loss(X[:, :, i, ...], Y[:, :, i, ...]) / (
                    torch.mean(X[:, :, i, ...]) + 1e-6)  # ** 2
    ergas = 100 * torch.sqrt(ergas / X.size(2))
    ergas = ergas.item()
    return ergas


def compare_rmse(x_true, x_pred):
    """
    Calculate Root mean squared error
    :param x_true:
    :param x_pred:
    :return:
    """
    x_true = x_true.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    x_pred = x_pred.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    return np.linalg.norm(x_true - x_pred) / (np.sqrt(x_true.shape[0] * x_true.shape[1] * x_true.shape[2]))


def compare_cc(x_true, x_pred):
    """
    Calculate the cross correlation between x_pred and x_true.
    Calculate the correlation coefficient of corresponding bands, and take the mean.
    CC is a spatial measure.
    """
    x_true = x_true.detach().squeeze(0).flatten(1).cpu().numpy()
    x_pred = x_pred.detach().squeeze(0).flatten(1).cpu().numpy()
    x_true = x_true - np.mean(x_true, axis=1).reshape(-1, 1)
    x_pred = x_pred - np.mean(x_pred, axis=1).reshape(-1, 1)
    numerator = np.sum(x_true * x_pred, axis=1).reshape(-1, 1)
    denominator = np.sqrt(np.sum(x_true * x_true, axis=1) * np.sum(x_pred * x_pred, axis=1)).reshape(-1, 1)
    return (numerator / denominator).mean()


def MSIQA(X, Y):
    X = torch.from_numpy(X).float().unsqueeze(0)
    Y = torch.from_numpy(Y).float().unsqueeze(0)
    psnr = np.mean(cal_bwpsnr(X, Y))
    ssim = np.mean(cal_bwssim(X, Y))
    sam = cal_sam(X, Y)
    ergas = cal_ergas(X, Y)
    rmse = compare_rmse(X, Y)
    cc = compare_cc(X, Y)

    return psnr, ssim, sam, ergas, rmse, cc


def HSI_calculate_psnr(img, img2, crop_border, **kwargs):
    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    psnr = MSIQA(img, img2)[0]
    return psnr


def HSI_calculate_ssim(img, img2, crop_border, **kwargs):
    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    ssim = MSIQA(img, img2)[1]
    return ssim


def HSI_calculate_sam(img, img2, crop_border, **kwargs):
    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    sam = MSIQA(img, img2)[2]
    return sam


def HSI_calculate_ergas(img, img2, crop_border, **kwargs):
    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    ergas = MSIQA(img, img2)[3]
    return ergas


def HSI_calculate_rmse(img, img2, crop_border, **kwargs):
    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    sam = MSIQA(img, img2)[4]
    return sam


def HSI_calculate_cc(img, img2, crop_border, **kwargs):
    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    ergas = MSIQA(img, img2)[5]
    return ergas