import torch 
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embedSize, trainCNN = False):
        super(EncoderCNN, self).__init__()
        self.trainCNN = False
        self.embedSize = embedSize

        self.inception = models.inception_v3(pretrained = True, aux_logits = True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embedSize)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)


    def forward(self, images):
        features = self.inception(images)[0]

        

        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.trainCNN

        x = self.relu(features)
        x = self.dropout(x)

        return x
    

class DecoderRNN(nn.Module):
    def __init__(self, embedSize, hiddenSize, vocabSize, numLayers):
        super(DecoderRNN, self).__init__() 

        self.embed = nn.Embedding(vocabSize, embedSize)
        self.lstm = nn.LSTM(embedSize, hiddenSize, numLayers)
        self.linear = nn.Linear(hiddenSize, vocabSize)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        #features in the output of the encoder

        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        output = self.linear(hiddens)
        return output
        


class CNNtoRNN(nn.Module):
    def __init__(self, embedSize, hiddenSize, vocabSize, numLayers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embedSize=embedSize)
        self.decoderRNN = DecoderRNN(embedSize=embedSize, hiddenSize=hiddenSize, vocabSize=vocabSize, numLayers=numLayers)
     
    def forward (self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)

        return outputs
    
    def captionImage(self, image, vocabulary, maxLenght=50):
        resultCaption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(maxLenght):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                perdicted = output.argmax(1)

                resultCaption.append(perdicted.item())

                x = self.decoderRNN.embed(perdicted).unsqueeze(0)

                if vocabulary.intToString[perdicted.item()] == "<EOS>":
                    break

        return [vocabulary.intToString[idx] for idx in resultCaption]