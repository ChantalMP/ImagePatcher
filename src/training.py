from src.utils import *



if __name__ == "__main__":
    epochs = 10000
    for epoch in range(epochs):
        images = load_pictures()
        generator = generator()
        descriminator = descriminator()
        modified = get_modified(images)
        generated_pictures = generator(modified)
        labels = descriminator(generated_pictures)
        generator.train(generated_pictures, labels)
        originals = get_originals(images)
        descriminator.train(originals, modified)
        print_result()


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

