from src.utils import *
from src.discriminator import Discriminator
import itertools
import random
from torch import optim
from torch import nn
import torch


if __name__ == "__main__":
    epochs = 10000
    batch_size = 1
    names = load_picture_names()
    random.shuffle(names)
    torch.set_printoptions(threshold=5000000000)
    descriminator = Discriminator()
    optimizer = optim.Adam(descriminator.parameters(),lr=0.00001)
    loss_func = nn.BCELoss()
    i = 0
    #generator = generator()

    correct = 0
    total = 0

    for epoch in range(epochs):
        images = names[i:i+batch_size]
        i = i+batch_size
        if i > len(names):
            i = 0
        modified = get_modified(images)
        #generated_pictures = generator(modified)
        #labels = descriminator(generated_pictures)
        #generator.train(generated_pictures, labels)
        originals = get_originals(images)
        #descriminator.train(originals, generated_pictures)
        #print_result()

        x,y = descriminator.prepare_data(originals, modified)
        predicted_y = descriminator(x)
        loss = loss_func(predicted_y,y)

        print(predicted_y)
        correct += (torch.abs(predicted_y[:,0] - y) <= 0.5).sum()
        total += batch_size * 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1 == 0:
            print(loss.item())
            print(correct,total)
            correct = 0
            total = 0



    #shoot through Generator, save output
        #in: broken pictures, 4 channels
        #fill spots with net output
        #out: 3 channel whole pic

    #get answers from Descriminator (on generator output) (0,1)
        #in: 3 channel whole pic
        #out: labels - 0 bad, 1 good

    #backprob generator
        #in: labels as loss

    #train desciminator (output(0) vs original(1))
        #in: 3 channel whole pic with 0 label, original with 1 label

    #next batch

