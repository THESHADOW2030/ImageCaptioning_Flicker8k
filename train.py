import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as trasnforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from getLoader import getLoader
from model import CNNtoRNN
from tqdm import tqdm


def train():
    trasnform = trasnforms.Compose(
        [
            trasnforms.Resize((356, 356)),
            trasnforms.RandomCrop(299, 299), 
            trasnforms.ToTensor(),
            trasnforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    trainLoader, dataset = getLoader(
        rootFolder="./data/Images",
        annotationFile="./data/captions.txt",
        transform=trasnform,
        numWorkers=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    loadModel = True
    saveModel = True

    embedSize = 256
    hiddenSize = 256
    vocabSize = len(dataset.vocab)
    numLayers = 1
    learningRate = 3e-4
    numEpoch = 100

    writer = SummaryWriter("./runs/flicker")
    step = 0

    model = CNNtoRNN(embedSize=embedSize, hiddenSize=hiddenSize, vocabSize=vocabSize, numLayers=numLayers).to(device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stringToInt["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr = learningRate)


    if loadModel:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()


    for epoch in tqdm(range(numEpoch)):
        if saveModel:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }
            save_checkpoint(checkpoint)
        for idx, (imgs, captions) in tqdm(enumerate(trainLoader), total=len(trainLoader), leave=False):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])

            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            #print the loss
            

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1  

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()


if __name__ == "__main__":
    train()
