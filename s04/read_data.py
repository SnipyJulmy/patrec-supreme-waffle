import os
from PIL import Image
import numpy
import sys

def create_training_set(root_train, transcription_file, train_file):
    training_dict = {}
    train = open(train_file).read().splitlines()
    lines = [line.rstrip('\n') for line in open(transcription_file)]
    for line in lines:
        split = line.split(' ')
        if split[0].split('-')[0] not in train:
            continue
        image = get_boolean_image(root_train + '/' + split[0] + '.jpg')
        if split[1] in training_dict:
            training_dict[split[1]].append(image)
        else:
            training_dict[split[1]] = []
            training_dict[split[1]].append(image)
    return training_dict

def create_set(root_data):
    result_dict = {}
    dirs = os.listdir( root_data )
    for file in dirs:
        image = get_boolean_image(root_data + '/' + file)
        result_dict[file.split('.')[0]] = image
    return result_dict

def get_transcriptions(transcription_file):
    transcriptions_dict = {}
    lines = [line.rstrip('\n') for line in open(transcription_file)]
    for line in lines:
        split = line.split(' ')
        transcriptions_dict[split[0]] = split[1]
    return transcriptions_dict

def get_boolean_image(file):
    # print("binarize %s" % (file_))
    im = Image.open(file).convert("L")
    height = im.size[0]
    width = im.size[1]
    px = im.load()
    for h in range(height):
        for w in range(width):
            if px[h, w] <= 125:
                px[h, w] = True
            else:
                px[h, w] = False
    return numpy.asarray(im)