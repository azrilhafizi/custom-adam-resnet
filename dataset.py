import torch as t
import torchvision
import torchvision.transforms as transforms

def get_cifar10():
    mean = t.tensor([125.307, 122.961, 113.8575]) / 255
    std = t.tensor([51.5865, 50.847, 51.255]) / 255
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    cifar_train = torchvision.datasets.CIFAR10("cifar10_train", transform=transform, download=True, train=True)
    cifar_test = torchvision.datasets.CIFAR10("cifar10_train", transform=transform, download=True, train=False)
    return cifar_train, cifar_test