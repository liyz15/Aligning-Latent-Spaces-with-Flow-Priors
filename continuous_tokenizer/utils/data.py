import numpy as np
from PIL import Image
import os

import math
import random
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import lmdb
import pickle
import io

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])




class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


class CachedFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
            transform=None,
            img_root: str = './ImageNet2012/train',
            return_img: bool = False,
            
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
            transform=transform,
        )
        self.img_root = img_root
        self.return_img = return_img

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        data = np.load(path)
        if torch.rand(1) < 0.5:  # randomly hflip
            zq = data['zq']
        else:
            zq = data['zq_flip']
        
        if self.return_img:
            img_path = os.path.join(self.img_root, str(data['path']))
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return zq, target, img
        else:
            return zq, target

class FileListImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        """Dataset that loads images from a filelist.txt in the root directory.
        
        Args:
            root (str): Root directory containing the filelist.txt and images
            transform (callable, optional): Transform to be applied on images
            target_transform (callable, optional): Transform to be applied on targets
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        filelist_path = os.path.join(root, "filelist.txt")
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"filelist.txt not found in {root}. Please ensure filelist.txt exists in the root directory.")
        
        # Read filelist
        with open(filelist_path, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
            
        # Get unique class folders and create class mapping
        class_folders = sorted(list(set(path.split('/')[0] for path in self.image_paths)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_folders)}
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of target class
        """
        img_path = os.path.join(self.root, self.image_paths[index])
        target = self.class_to_idx[self.image_paths[index].split('/')[0]]
        
        # Load and transform image
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
    
    def __len__(self):
        return len(self.image_paths)
    


class FileListImageDatasetWithCache(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, cache_path=None):
        """Dataset that loads images from a filelist.txt in the root directory.
        
        Args:
            root (str): Root directory containing the filelist.txt and images
            transform (callable, optional): Transform to be applied on images
            target_transform (callable, optional): Transform to be applied on targets
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        filelist_path = os.path.join(root, "filelist.txt")
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"filelist.txt not found in {root}. Please ensure filelist.txt exists in the root directory.")
        
        # Read filelist
        with open(filelist_path, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
            
        # Get unique class folders and create class mapping
        class_folders = sorted(list(set(path.split('/')[0] for path in self.image_paths)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_folders)}

        self.env = lmdb.open(
            cache_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=False
        )
        self.txn = self.env.begin(write=False)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of target class
        """
        img_path = os.path.join(self.root, self.image_paths[index])
        target = self.class_to_idx[self.image_paths[index].split('/')[0]]

        key = self.image_paths[index].encode('ascii')
        value = self.txn.get(key)
        if value is None:
            raise ValueError(f"Key {img_path} not found in LMDB database")
        data = pickle.loads(value)
        moments = torch.from_numpy(data['moments'])
            
        # Load and transform image
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target, moments
    
    def __len__(self):
        return len(self.image_paths)


class LMDBImageDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path, filelist_path, transform=None, target_transform=None):
        """Dataset that loads images from LMDB, maintaining compatibility with FileListImageDataset.
        
        Args:
            lmdb_path (str): Path to the LMDB directory
            filelist_path (str): Path to filelist.txt containing relative paths
            transform (callable, optional): Transform to be applied on images
            target_transform (callable, optional): Transform to be applied on targets
        """
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        
        # Open LMDB environment
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=False
        )
        self.txn = self.env.begin(write=False)
        
        # Read filelist
        with open(filelist_path, 'r') as f:
            self.image_paths = [line.strip() for line in f]
            
        # Get unique class folders and create class mapping
        class_folders = sorted(list(set(path.split('/')[0] for path in self.image_paths)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_folders)}
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of target class
        """
        img_path = self.image_paths[index]
        target = self.class_to_idx[img_path.split('/')[0]]
        
        # Get image bytes from LMDB
        value = self.txn.get(img_path.encode('ascii'))
        if value is None:
            raise ValueError(f"Key {img_path} not found in LMDB database")
        
        # Load image from bytes
        img = Image.open(io.BytesIO(value)).convert('RGB')
            
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
    
    def __len__(self):
        return len(self.image_paths)
    
    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()


class CachedList(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path: str,
        root_dir: str,
        use_lmdb: bool = False
    ):
        """
        Args:
            filelist_path (string): Path to the text file with cached file names.
            root_dir (string): Directory with all the cached files.
            use_lmdb (bool): Whether to use LMDB for loading cached data.
        """
        self.root_dir = root_dir
        self.use_lmdb = use_lmdb
        
        # Load cached file paths
        with open(filelist_path, "r") as f:
            self.cached_paths = [line.strip() for line in f]
            
        # Extract labels from paths (assuming format: label_id/filename)
        labels = list(set(p.split("/")[0] for p in self.cached_paths))
        labels = sorted(labels)
        self.id2label = {k: i for i, k in enumerate(labels)}

        if use_lmdb:
            # Open LMDB environment in read-only mode
            self.env = lmdb.open(os.path.join(root_dir, 'final.lmdb'),
                               readonly=True,
                               lock=False,
                               readahead=False,
                               meminit=False,
                               subdir=False)
            self.txn = self.env.begin()
        
        self.warned = False

    def __len__(self):
        return len(self.cached_paths)

    def __getitem__(self, idx):
        path = self.cached_paths[idx]
        
        if self.use_lmdb:
            # Load from LMDB
            key = path.encode('ascii')
            value = self.txn.get(key)
            if value is None:
                raise ValueError(f"Key {path} not found in LMDB")
            data = pickle.loads(value)
            if 'moments' in data:
                moments = torch.from_numpy(data['moments'])
                moments_flip = torch.from_numpy(data['moments_flip'])
            else:
                if not self.warned:
                    print(f"Warning: {path} does not have moments")
                    self.warned = True
                moments = moments_flip = data
        else:
            # Load from NPZ file
            path = os.path.join(self.root_dir, path + '.npz')
            data = np.load(path)
            moments = torch.from_numpy(data['moments'])
            moments_flip = torch.from_numpy(data['moments_flip'])

        if torch.rand(1) < 0.5:  # randomly hflip
            moments = moments
        else:
            moments = moments_flip

        # Extract label from path
        label_id = self.cached_paths[idx].split("/")[0]
        label = self.id2label[label_id]

        return moments, label

    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()


class CachedTensors(torch.utils.data.Dataset):
    def __init__(self, cached_tensors_path: str):
        ckpt = torch.load(cached_tensors_path, map_location="cpu")
        if isinstance(ckpt, torch.Tensor):
            # This is a processed checkpoint, we read the tensor directly
            self.cached_tensors = ckpt
        else:
            # This is a llamagen checkpoint, we read ['model']['quantize.embedding.weight']
            self.cached_tensors = torch.load(cached_tensors_path, map_location="cpu")['model']['quantize.embedding.weight']


    def __len__(self):
        # ImageNet has 1,281,167 images
        return 1281167
    
    def __getitem__(self, idx):
        idxes = np.random.choice(len(self.cached_tensors), size=64, replace=False)
        return self.cached_tensors[idxes], 0
        

class DualImageLatentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_root: str,
        latent_path: str,
        transform=None,
    ):
        """
        Dataset that loads both real images and generated latents for adversarial training.
        
        Args:
            image_root (string): Path to the ImageNet real images
            latent_path (string): Path to the LMDB directory containing latents generated by generate_dit.py
            transform: Transformations to apply to real images
        """
        self.image_root = image_root
        self.transform = transform
        
        # Load the real image dataset
        self.real_dataset = FileListImageDataset(
            root=image_root,
            transform=transform
        )
        
        self.lmdb_env = lmdb.open(
            latent_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=False
        )
        
        # Get the number of samples in the LMDB
        with self.lmdb_env.begin(write=False) as txn:
            self.num_latents = txn.stat()['entries']
        
        print(f"DualImageLatentDataset created with {len(self.real_dataset)} real images and {self.num_latents} generated latents")
        
    def __len__(self):
        # Use the length of real dataset as the primary length
        return len(self.real_dataset)
    
    def __getitem__(self, idx):
        # Get real image and its class
        real_img, real_class = self.real_dataset[idx]
        
        # Get generated latent - sample uniformly from all latents
        gen_idx = idx % self.num_latents
        
        # Generate key for LMDB lookup
        key = f"{gen_idx:06d}".encode('ascii')
        
        # Retrieve latent from LMDB
        with self.lmdb_env.begin(write=False) as txn:
            value = txn.get(key)
            
        if value is None:
            raise ValueError(f"Key {key} not found in LMDB database")
            
        # Deserialize the data
        data = pickle.loads(value)
        latents = data['latents']
        gen_class = data['class']
        
        # Convert latent to tensor
        gen_latent = torch.from_numpy(latents)
            
        return {
            "real_img": real_img,
            "gen_latent": gen_latent,
            "real_class": real_class,
            "gen_class": gen_class
        }
    
    def __del__(self):
        # Clean up the LMDB environment
        if hasattr(self, 'lmdb_env'):
            self.lmdb_env.close()
        

class FileListPatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, patch_size=16):
        """Dataset that loads images from a filelist.txt in the root directory.
        
        Args:
            root (str): Root directory containing the filelist.txt and images
            transform (callable, optional): Transform to be applied on images
            target_transform (callable, optional): Transform to be applied on targets
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        filelist_path = os.path.join(root, "filelist.txt")
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"filelist.txt not found in {root}. Please ensure filelist.txt exists in the root directory.")
        
        # Read filelist
        with open(filelist_path, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
            
        # Get unique class folders and create class mapping
        class_folders = sorted(list(set(path.split('/')[0] for path in self.image_paths)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_folders)}
        self.patch_size = patch_size

    def _unfold_img(self, img):
        """
        Unfold the image into patches
        """
        C, H, W = img.shape
        img = img.unsqueeze(0)
        patches = F.unfold(img, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2)  # [B, num_patches, C*patch_size*patch_size]
        patches = patches.reshape(-1, patches.shape[-1])  # [B*num_patches, C*patch_size*patch_size]

        num_patch_to_sample = 128

        patches = patches[torch.randperm(patches.shape[0])[:num_patch_to_sample]]

        return patches
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of target class
        """
        img_path = os.path.join(self.root, self.image_paths[index])
        target = self.class_to_idx[self.image_paths[index].split('/')[0]]
        
        # Load and transform image
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            
        if self.transform is not None:
            img = self.transform(img)
            img = self._unfold_img(img)
        else:
            raise NotImplementedError("Transform not implemented")
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
    
    def __len__(self):
        return len(self.image_paths)
