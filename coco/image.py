import os 
import shutil
import pandas as pd 
from tqdm import tqdm
from threading import Thread
from PIL import Image

from coco.params_data import (
    BASE_FOLDER_PATH, TRAIN_PATH,
    VALIDATION_PATH, LABELS_FOLDER,
    WIDTH, HEIGHT
)

class ImageSaver:
    def __init__(
            self,
            image_url,
            x_center,
            y_center,
            width,
            height,
            label,
            result_folder,
            label_folder
        ):
        self.image_url = image_url
        self.x_center = x_center 
        self.y_center = y_center 
        self.width = width 
        self.height = height 
        self.label = label 
        self.type = type
        self.image_name = os.path.basename(self.image_url)
        self.result_folder = result_folder
        self.label_folder = label_folder

    def prepare_image(self):
        destination_folder = os.path.join(self.result_folder, self.image_name)
        # open image and save it in destination folder with the same width and height   
        image = Image.open(self.image_url)
        image = image.resize((WIDTH, HEIGHT))
        image.save(destination_folder)

        # shutil.copy(self.image_url, destination_folder)

    def prepare_label(self):
        image_name, _ = os.path.splitext(self.image_name)
        label_name = image_name + '.txt'
        label_path = os.path.join(self.label_folder, label_name)
        label = f"{self.label} {self.x_center} {self.y_center} {self.width} {self.height}"
        with open(label_path, 'a') as f:
            f.write(f"{label}")
            f.write("\n")
        
    def prepare(self): 
        self.prepare_image()
        self.prepare_label()

class Dataset:
    def __init__(self, data_type="train"):
        self.data_type = data_type
        self.df = pd.read_csv(BASE_FOLDER_PATH + f'/{self.data_type}.csv')
        self.df.rename(columns={'class': 'label'}, inplace=True)
        self.df = self.df.astype({'label':'int32'})
        self.normalization = {
            'x': WIDTH,
            'y': HEIGHT
        }

    def prepare(self):
        self.df['image_url'] = self.df['image_path'].apply(
            lambda x: BASE_FOLDER_PATH + '/images/' + x
        )

        # compute x-center, y-center, width, height of box 
        self.df['xmin'] = self.df['xmin'].apply(lambda x : max(x, 0))
        self.df['ymin'] = self.df['ymin'].apply(lambda x : max(x, 0))
        self.df['xmax'] = self.df['xmax'].apply(lambda x : min(x, WIDTH))
        self.df['ymax'] = self.df['ymax'].apply(lambda x : min(x, HEIGHT))

        columns = ['x_center', 'y_center', 'width', 'height']
        for var in ['x', 'y']:
            for type in ['center', 'side']:
                feature_name, feature = self.process_feature(var, type)
                self.df[feature_name] = feature
        
        final_columns = ['image_url'] + columns + ['label', 'name']

        self.df = self.df[final_columns]

    def get_function(self, axis, type='center'):
        if type == 'center':
            return lambda x: ((x[0] + x[1])/2)/self.normalization[axis]
        
        return lambda x: ((x[1] - x[0]))/self.normalization[axis]

    def process_feature(self, var, type):
        features = [f"{var}min", f"{var}max"]
        if type == 'center':
            feature_name = f"{var}_{type}"
        else:
            feature_name = 'width' if var == 'x' else 'height'

        feature = (
                self.df[features]
                .apply(
                    self.get_function(var, type),
                    axis=1
                )
        )

        return feature_name, feature 

    def get(self):
        self.prepare()
        return self.df


class DataReader:
    def __init__(self, data_frame, type='train'):
        self.data_frame = data_frame
        self.type = type
        self.result_folder = TRAIN_PATH if type == 'train' else VALIDATION_PATH
        self.label_folder = os.path.join(LABELS_FOLDER, self.type)
        # removing old data 
        if os.path.exists(self.result_folder):
            shutil.rmtree(self.result_folder, ignore_errors=True)
            shutil.rmtree(self.label_folder, ignore_errors=True)

        os.makedirs(self.result_folder, exist_ok=True)
        os.makedirs(self.label_folder, exist_ok=True)

    def process_group(self, group):
        for _, row in group.iterrows():
            saver = ImageSaver(*(row[:-1]), self.result_folder, self.label_folder)
            saver.prepare()
        
    def read(self):
        # group by image_url and process groups in parallel
        groups = self.data_frame.groupby('image_url')
        threads = []
        for _, group in tqdm(groups):
            thread = Thread(target=self.process_group, args=(group,))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        