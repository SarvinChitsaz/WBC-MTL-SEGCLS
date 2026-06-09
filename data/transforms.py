import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(p=0.4),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(),
        ToTensorV2()
    ])

def get_test_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2()
    ])

def get_minority_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.7),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(p=0.6),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])
