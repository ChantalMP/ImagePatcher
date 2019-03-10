from src.utils import *
from src.discriminator import Discriminator
import itertools
import random



if __name__ == "__main__":
    epochs = 10000
    batch_size = 32
    names = load_picture_names()
    random.shuffle(names)
    descriminator = Discriminator()
    i = 0
    #generator = generator()
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

        data = descriminator.prepare_data(originals, modified)
        descriminator(data)


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

