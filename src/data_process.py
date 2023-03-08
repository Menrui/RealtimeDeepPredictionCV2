from torchvision.transforms import transforms


def preprocess_transform() -> transforms.Compose:
    return transforms.Compose(
        (
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    )
