import os
from PIL import Image
from torch.utils.data import Dataset
import transforms as trans
from torchvision import transforms
import random
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class USODDataset(Dataset):
    def __init__(self,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 depth_prefix='',
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.depth_prefix = depth_prefix
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.depth_prefix is None or osp.isabs(self.depth_prefix)):
                self.depth_prefix = osp.join(self.data_root, self.depth_prefix)

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        return 0

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['depth_prefix'] = self.depth_prefix
        results['proposal_file'] = None
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (if `test_mode` is set True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)

        return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)


