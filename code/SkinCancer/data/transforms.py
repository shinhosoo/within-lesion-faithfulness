import os

from torchvision import transforms

from SkinCancer.configs.model_config import IMG_SIZE


# HAM_AUGMENT=code (default, the original code recipe) | paper (the paper's Sec 2.3)
AUGMENT = os.environ.get("HAM_AUGMENT", "code")


def build_train_transform():
    if AUGMENT == "paper":
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    else:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(0.2, 0.2),
        ]
    return transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE))]
        + aug
        + [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_val_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
