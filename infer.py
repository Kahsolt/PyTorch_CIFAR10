import os
from argparse import ArgumentParser

from tqdm import tqdm
import torch
from pytorch_lightning import seed_everything

from data import CIFAR10Data
from module import all_classifiers

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.inference_mode()
def main(args):
    model = all_classifiers[args.classifier]()
    state_dict = os.path.join("cifar10_models", "state_dicts", args.classifier + ".pt")
    model.load_state_dict(torch.load(state_dict))
    model = model.eval().to(device)

    data = CIFAR10Data(args)
    testloader = data.test_dataloader()

    total, ok = 0, 0
    for X, Y in tqdm(testloader):
        X = X.to(device)
        Y = Y.to(device)

        logits = model(X)
        pred = logits.argmax(-1)

        total += len(Y)
        ok += (pred == Y).sum().item()

    print(f'{args.classifier}: {ok / total:.3%}')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--classifier", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    seed_everything(0)
    main(args)
