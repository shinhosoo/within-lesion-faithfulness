from torch.utils.data import DataLoader

from SkinCancer.configs.training_config import BATCH_SIZE
from SkinCancer.data.dataset import LocalSkinCancerDataset
from SkinCancer.utils.logging import debug_loader


def make_loader(df, label_col, transform, shuffle=False):
    loader = DataLoader(
        LocalSkinCancerDataset(
            df["image_path"].tolist(),
            df[label_col].astype(float).tolist(),
            transform,
        ),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )
    debug_loader(f"{label_col}, shuffle={shuffle}", loader)
    return loader
