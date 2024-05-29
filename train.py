import torch as t
from torch.utils.data import DataLoader
import wandb
import time
from tqdm.notebook import tqdm
from typing import Any

from resnet import ResNet
from adam_optimizer import Adam
from dataset import get_cifar10

device = t.device("cuda" if t.cuda.is_available() else "cpu")

def train(config: dict[str, Any] = None):
    with wandb.init(config=config):
        config = wandb.config
        print(f"Training with config: {config}")
        cifar_train, cifar_test = get_cifar10()
        trainloader = DataLoader(cifar_train, batch_size=config.batch_size, shuffle=True, pin_memory=True)
        testloader = DataLoader(cifar_test, batch_size=1024, pin_memory=True) 
        model = ResNet34(n_blocks_per_group=[1, 1, 1, 1], n_classes=10).to(device).train()
        optimizer = Adam(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),  
            weight_decay=0  
        )
        train_loss_fn = t.nn.CrossEntropyLoss()
        wandb.watch(model, criterion=train_loss_fn, log="all", log_freq=10, log_graph=True)
        start_time = time.time()
        examples_seen = 0
        for epoch in range(10): 
            for i, (x, y) in enumerate(tqdm(trainloader)):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_hat = model(x)
                loss = train_loss_fn(y_hat, y)
                acc = (y_hat.argmax(dim=-1) == y).sum().item() / len(x)
                loss.backward()
                optimizer.step()
                wandb.log(
                    dict(
                        train_loss=loss.item(),
                        train_accuracy=acc,
                        elapsed=time.time() - start_time,
                    ),
                    step=examples_seen,
                )
                examples_seen += len(x)

        test_loss_fn = t.nn.CrossEntropyLoss(reduction="sum")
        with t.inference_mode():
            n_correct = 0
            n_total = 0
            loss_total = 0.0
            for i, (x, y) in enumerate(tqdm(testloader)):
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                loss_total += test_loss_fn(y_hat, y).item()
                n_correct += (y_hat.argmax(dim=-1) == y).sum().item()
                n_total += len(x)
            wandb.log(
                dict(
                    test_loss=loss_total / n_total,
                    test_accuracy=n_correct / n_total,
                    step=examples_seen,
                )
            )
        filename = f"{wandb.run.dir}/model_state_dict.pt"
        t.save(model.state_dict(), filename)
        wandb.save(filename)