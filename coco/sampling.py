import numpy as np 
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class Sampler:
    def __init__(self, df):
        self.df = df
        self.seed = 42
        np.random.seed(self.seed)

    def random_sample(self, n=1000):
        """
        Sample n images from self.df data
        """
        # df = self.df[self.df.label != 6]
        # # if label is bigger than 6, subtract 1
        # df.label = df.label.apply(lambda x: x-1 if x > 6 else x)
        # self.df = df 
        # sample n images 
        images_unique = self.df.image_url.unique()
        np.random.shuffle(images_unique)
        
        validation_images = images_unique[:n]
        mask = self.df.image_url.isin
        train_images = self.df[~mask(validation_images)].image_url.unique()

        training_csv, validation_csv = (
            self.df[mask(train_images)],
            self.df[mask(validation_images)]
        )   
        # reset index 
        training_csv.reset_index(drop=True, inplace=True)
        validation_csv.reset_index(drop=True, inplace=True)

        return training_csv, validation_csv 

    def stratified_sample(self, test_size=0.1):
        """
        Stratified Sample n
        """
        # remove BAD_STREETLIGHT class 
        df = self.df[self.df.label != 6]

        # if label is bigger than 6, subtract 1
        df.label = df.label.apply(lambda x: x-1 if x > 6 else x)

        # perform stratified sampling 
        training_csv, validation_csv = train_test_split(
            df,
            test_size=test_size,
            random_state=self.seed,
            stratify=df.label
        )

        # reset index 
        training_csv.reset_index(drop=True, inplace=True)
        validation_csv.reset_index(drop=True, inplace=True)

        return training_csv, validation_csv 

