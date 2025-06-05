import lmdb
import argparse
import pickle
import numpy as np
import torch
import os
from tqdm import tqdm
from models.vae import DiagonalGaussianDistribution as DGD_MAR
import sys
sys.path.append('/path/to/continuous_tokenizer')
from modelling.quantizers.kl import DiagonalGaussianDistribution as DGD_CT

def get_std_from_list_var(Vars, shape, count):
    n_elements = np.prod(shape)
    return np.sqrt(np.sum(Vars) * n_elements / (count * n_elements - 1))


def estimate_scaling_factor(lmdb_path, num_samples=1000):
    # Open LMDB environment
    env = lmdb.open(os.path.join(lmdb_path, 'final.lmdb'),
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    subdir=False)
    
    # Start a new transaction
    with env.begin() as txn:
        cursor = txn.cursor()
        
        # Initialize statistics
        min_val = float('inf')
        max_val = float('-inf')
        abs_mean_sum = 0
        Means = []
        Vars = []
        count = 0
        shape = None
        
        # Sample multiple entries
        for key, value in tqdm(cursor, total=num_samples):
            if count >= num_samples:
                break
                
            # Deserialize the value
            data = pickle.loads(value)
            if 'moments' in data:
                moments = data['moments']
            else:
                moments = data


            # # For KL-16 in MAR
            print("KL for MAR")
            # posterior = DGD_MAR(torch.from_numpy(moments).unsqueeze(0))
            # latents = posterior.sample()

            # # For KL-16 in SoftVQ
            print("KL for SoftVQ")
            # posterior = DGD_CT(torch.from_numpy(moments).unsqueeze(0))
            # latents = posterior.sample()

            # For AE
            print("Autoencoder")
            latents = torch.from_numpy(moments)

            # print(latents.shape)

            # Record shape from first sample
            if shape is None:
                shape = latents.shape
            
            # assert shape[-1] == 32, f"shape: {shape}"
            
            # Update statistics
            min_val = min(min_val, latents.min())
            max_val = max(max_val, latents.max())
            abs_mean_sum += np.abs(latents).mean()
            Means.append(latents.mean())
            Vars.append(latents.var(unbiased=False))
            count += 1
            if count % 100 == 0:
                avg_std_dev = get_std_from_list_var(Vars, shape, count)
                print(f"1 / avg_std_dev: {1.0 / avg_std_dev:.4f}")
        
        # Calculate averages
        avg_abs_mean = abs_mean_sum / count
        avg_std_dev = get_std_from_list_var(Vars, shape, count)
        avg_mean = np.mean(Means)

        print(f"Statistics over {count} samples:")
        print(f"Latent shape: {shape}")
        print(f"Min value: {min_val:.4f}")
        print(f"Max value: {max_val:.4f}")
        print(f"Average mean absolute value: {avg_abs_mean:.4f}")
        print(f"Average standard deviation: {avg_std_dev:.4f}")
        print(f"Average mean: {avg_mean:.4f}")

        # print(f"Means: {Means}")
        # print(f"Vars: {Vars}")

        scaling_factor = 1.0 / avg_std_dev
        print(f"\nScaling factor to normalize to [-1, 1]: {scaling_factor:.4f}")
        print(f"After scaling:")
        print(f"Min value: {min_val * scaling_factor:.4f}")
        print(f"Max value: {max_val * scaling_factor:.4f}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate VAE scaling factor from LMDB')
    parser.add_argument('lmdb_path', type=str, help='Path to directory containing final.lmdb')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to analyze')
    
    args = parser.parse_args()
    estimate_scaling_factor(args.lmdb_path, args.num_samples) 