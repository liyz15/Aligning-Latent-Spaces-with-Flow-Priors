import os
import lmdb
import pickle
import io
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader


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
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )

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
            moments = data["moments"]
        else:
            moments = data["moments_flip"]

        return moments, target


class CachedList(Dataset):
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
            moments = torch.from_numpy(data['moments'])
            moments_flip = torch.from_numpy(data['moments_flip'])
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


class ImageList(Dataset):
    def __init__(self, filelist_path, root_dir, transform=None):
        """
        Args:
            filelist_path (string): Path to the text file with image file names.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Load image paths
        with open(filelist_path, "r") as f:
            self.img_paths = [line.strip() for line in f]

        labels = list(set(p.split("/")[0] for p in self.img_paths))
        labels = sorted(labels)
        self.id2label = {k: i for i, k in enumerate(labels)}
        self.loader = default_loader

    def __len__(self):
        return len(self.img_paths)

    def __getitem_wo_retry(self, idx):
        # Construct full image path
        img_name = os.path.join(self.root_dir, self.img_paths[idx])
        # Load image
        image = self.loader(img_name)

        # Extract label id from file path and get corresponding label
        label_id = self.img_paths[idx].split("/")[0]
        label = self.id2label[label_id]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __getitem__(self, idx):
        while True:
            try:
                return self.__getitem_wo_retry(idx)
            except:
                print(f"Error loading image {self.img_paths[idx]}")
                idx = np.random.randint(0, len(self.img_paths))


class ImageListWithFilename(ImageList):
    def __getitem_wo_retry(self, idx):
        # Construct full image path
        img_name = os.path.join(self.root_dir, self.img_paths[idx])
        # Load image
        image = self.loader(img_name)

        # Extract label id from file path and get corresponding label
        label_id = self.img_paths[idx].split("/")[0]
        label = self.id2label[label_id]

        if self.transform:
            image = self.transform(image)

        # Return filename in the same format as ImageFolderWithFilename
        filename = os.path.join(label_id, os.path.basename(self.img_paths[idx]))
        return image, label, filename

    def __getitem__(self, idx):
        while True:
            try:
                return self.__getitem_wo_retry(idx)
            except:
                print(f"Error loading image {self.img_paths[idx]} {self.root_dir}")
                idx = np.random.randint(0, len(self.img_paths))


class LMDBImageListWithFilename(torch.utils.data.Dataset):
    def __init__(self, filelist_path, lmdb_path, transform=None):
        """Dataset that loads images from LMDB, maintaining compatibility with FileListImageDataset.
        
        Args:
            lmdb_path (str): Path to the LMDB directory
            filelist_path (str): Path to filelist.txt containing relative paths
            transform (callable, optional): Transform to be applied on images
            target_transform (callable, optional): Transform to be applied on targets
        """
        super().__init__()
        self.transform = transform
        
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
        # Load image paths
        with open(filelist_path, "r") as f:
            self.img_paths = [line.strip() for line in f]

        labels = list(set(p.split("/")[0] for p in self.img_paths))
        labels = sorted(labels)
        self.id2label = {k: i for i, k in enumerate(labels)}
        self.loader = default_loader
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, label, filename)
        """
        img_path = self.img_paths[idx]
        label_id = img_path.split('/')[0]
        label = self.id2label[label_id]
        
        # Get image bytes from LMDB
        value = self.txn.get(img_path.encode('ascii'))
        if value is None:
            raise ValueError(f"Key {img_path} not found in LMDB database")
        
        # Load image from bytes
        img = Image.open(io.BytesIO(value)).convert('RGB')
            
        if self.transform is not None:
            img = self.transform(img)
            
        filename = os.path.join(label_id, os.path.basename(self.img_paths[idx]))

        return img, label, filename
    
    def __len__(self):
        return len(self.img_paths)
    
    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()


class LMDBImageDataset(Dataset):
    def __init__(self, lmdb_path, keys_path, transform=None):
        """
        Args:
            lmdb_path (string): Path to the LMDB file.
            keys_path (string): Path to the pickle file with keys for accessing LMDB.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.lmdb_path = lmdb_path
        self.transform = transform
        
        # Load keys for accessing images
        with open(keys_path, 'rb') as f:
            self.keys = pickle.load(f)
        
        # Open LMDB file with subdir=False since it is directly a file
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)

    def __len__(self):
        return len(self.keys)
    
    def __getitem_wo_retry(self, idx):
        with self.env.begin(write=False) as txn:
            key = self.keys[idx]  # Assuming the keys are already in bytes format
            img_buffer = txn.get(key)
            
            if img_buffer is None:
                raise FileNotFoundError(f"No image found for key: {key}")
            
            # Load image from buffer
            img_buffer, label = pickle.loads(img_buffer)
            img = Image.open(io.BytesIO(img_buffer))
            img = img.convert('RGB')  # Ensure image is in RGB format

            label = int(label)

            if self.transform:
                img = self.transform(img)
        
        return img, label

    def __getitem__(self, idx):
        retry = 0
        while retry < 5:
            try:
                return self.__getitem_wo_retry(idx)
            except Exception as e:
                print(f"Error loading image with key {self.keys[idx]}: {e}")
                idx = np.random.randint(0, len(self.keys))  # Randomly select a new index
                retry += 1
