#First we need to convert text to numerical values (so we need a vocab)
#We need to create our dataset class
#We also need to setup padding of everybactch

import pandas as pd
import os
import spacy        #tokenizer
import torch
from  torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import torchvision.transforms as transforms

spacyEng = spacy.load("en_core_web_sm")

class Vocabulary():
    def __init__(self, freqThreshold):
        self.freqThreshold = freqThreshold  
        self.intToString = {0: "<PAD>",
                            1: "<SOS>",
                            2: "<EOS>",
                            3: "<UNK>"}
        self.stringToInt = {"<PAD>": 0,
                            "<SOS>": 1,
                            "<EOS>": 2,
                            "<UNK>": 3}
    def __len__(self):
        return len(self.intToString)
    
    @staticmethod
    def tokenizerEng(text):
        return [tok.text.lower() for tok in spacyEng.tokenizer(text)]
    

    def buildVocabulary(self, captions):
        frequencies = {}
        index = 4   #we already have used the 0...3 indexed

        for caption in captions:
            for word in self.tokenizerEng(caption):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == self.freqThreshold:
                    self.intToString[index] = word
                    self.stringToInt[word] = index
                    index += 1


    def numericalize(self, caption):
        return [self.stringToInt[word] if word in self.stringToInt 
                else self.stringToInt["<UNK>"] 
                for word in self.tokenizerEng(caption)]

class FlickerDataset(Dataset):
    def __init__(self, rootDir, CaptionFile, transform = None, freqThreshold = 5):
        self.rootDir = rootDir
        self.df = pd.read_csv(CaptionFile)
        self.transform = transform

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = Vocabulary(freqThreshold=freqThreshold)    
        self.vocab.buildVocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        imgID = self.imgs[index]

        img = Image.open(os.path.join(self.rootDir, imgID)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)


        numericalizedCaption = [self.vocab.stringToInt["<SOS>"]]        #start of sentence
        numericalizedCaption += self.vocab.numericalize(caption)
        numericalizedCaption.append(self.vocab.stringToInt["<EOS>"])    #end of sentence

        return img, torch.tensor(numericalizedCaption)



class MyCollate:
    def __init__(self, padIDX):
        self.padIDX = padIDX
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(sequences=targets, batch_first=False, padding_value=self.padIDX)

        return imgs, targets
    

def getLoader(rootFolder,
        annotationFile,
        transform, 
        batchSize = 32,
        numWorkers = 8,
        shuffle = True,
        pinMemory = True):
    dataset = FlickerDataset(rootDir=rootFolder,CaptionFile= annotationFile, transform=transform)

    padIDX = dataset.vocab.stringToInt["<PAD>"]

    loader = DataLoader(dataset=dataset,
                        batch_size=batchSize,
                        num_workers=numWorkers,
                        shuffle=shuffle,
                        pin_memory=pinMemory,
                        collate_fn=MyCollate(padIDX=padIDX)
                        )
    return loader

transformsComposition = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()]
)


dataLoader = getLoader("data/Images", annotationFile="data/captions.txt", transform=transformsComposition)

for idx, (imgs, captions) in enumerate(dataLoader):
    print(imgs.shape)
    print(captions.shape)

