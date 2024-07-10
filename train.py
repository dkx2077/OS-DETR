import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import os

def train_model(model_yaml=None, train_name=None):
    if 1:
        model = RTDETR(model_yaml)
        model.train(data='dataset/brain.yaml',
                    cache=False,
                    imgsz=640,
                    epochs=100,
                    batch=4,
                    workers=4,
                    device='0',
                    resume='', # last.pt path
                    project='runs/train',
                    name=train_name,
                    )

        os.remove("./dataset/dataset-Br35H/traindata.cache")
        os.remove("./dataset/dataset-Br35H/valdata.cache")


if __name__ == '__main__':
    train_model('ultralytics/cfg/models/rt-detr/rtdetr-Ortho-AIFI-DAT-Shuffle.yaml', 'test')
