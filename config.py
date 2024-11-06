import torch
import argparse
import warnings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="modern-office-31-seperated", type=str)
    parser.add_argument('--source_domain', default="amazon", type=str)
    parser.add_argument('--target_domain', default="webcam", type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--adv_learning_rate', type=float, default=5e-6)
    parser.add_argument('--feature_extractor_id', type=str, default="google/vit-base-patch16-224-in21k")
    
    return parser.parse_args()

def set_seed():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning)