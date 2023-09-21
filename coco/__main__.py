import os 
import pandas as pd 
import numpy as np 


from coco.image import DataReader, Dataset
from coco.sampling import Sampler


if __name__ == '__main__':
    
    dataset = Dataset('train').get()
    training_csv, validation_csv = Sampler(dataset).random_sample()

    print("# of training images: ", len(training_csv.image_url.unique()))
    print("# of validation images: ", len(validation_csv.image_url.unique()))

    train_reader = DataReader(training_csv, type='train')
    train_reader.read()

    validation_reader = DataReader(validation_csv, type='val')
    validation_reader.read()

    print("Done!")

    
    
