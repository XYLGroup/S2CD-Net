import argparse
# Training settings
parser = argparse.ArgumentParser(description="Hyperspectral Image Super-Resolution")
parser.add_argument("--upscale_factor", default=4, type=int, help="super resolution upscale factor")
parser.add_argument('--seed', type=int, default=4,  help='random seed')
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="maximum number of epochs to train")
parser.add_argument("--show", action="store_true", help="show Tensorboard")

parser.add_argument("--lr", type=int, default=1e-4, help="initial  lerning rate")
parser.add_argument("--cuda", action="store_true", help="Use cuda")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")
parser.add_argument("--threads", type=int, default=8, help="number of threads for dataloader to use")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")               

parser.add_argument("--datasetName", default="CAVE", type=str, help="data name")
parser.add_argument("--modelName", default="S2CD_Net", type=str, help="model name")
parser.add_argument("--inch", type=int, default=31, help="input channel")

# Network settings
# Test image
parser.add_argument('--model_name', default='', type=str, help='super resolution model name ')
opt = parser.parse_args() 


# 4hard 0