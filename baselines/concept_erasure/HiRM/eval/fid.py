import warnings
import argparse
import torch
from cleanfid import fid

warnings.filterwarnings("ignore")




def fid_score(args):
    print("Evaluating...")
    
    fid_value = fid.compute_fid(args.img_dir, args.orig_dir, batch_size=64, num_workers=4,mode="clean")
    print(f"FID: {fid_value}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

   
    parser.add_argument("--img_dir", type=str, help="path to dir of generated images")
    parser.add_argument("--device", type=int, help="FID uses the all gpus by defualt.", default=0)
    parser.add_argument("--orig_dir", type=str, help="path to the dir of SD1.4-generated images")

    args = parser.parse_args()

    fid_score(args)
