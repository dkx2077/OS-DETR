import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import os


def val_model(model_yaml=None, train_name=None):
    try:
        model = RTDETR('runs/train/' + train_name + '/weights/best.pt')
        model.val(data='dataset/brain.yaml',
                    split='test',
                    imgsz=640,
                    batch=4,
                    # save_json=True, # if you need to cal coco metrice
                    project='runs/val1',
                    name=train_name,
                    )
        
    except Exception as e:
        print(e)
        

if __name__ == '__main__':
    val_model('ultralytics/cfg/models/rt-detr/rtdetr-DAT-AIFI-LPE-Shuffle.yaml', 'OS')
