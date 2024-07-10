import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/OS/weights/best.pt') # select your model.pt path
    model.predict(source='dataset/dataset-Br35H/valdata',
                  project='runs/detect',
                  name='OS',
                  save=True,
                  save_txt=True,
                #   visualize=True # visualize model features maps
                  )