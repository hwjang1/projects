import cv2
import torch
import os
import numpy as np
import json
import skimage.draw
import datetime
from matplotlib.pyplot import imshow
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
import skimage.filters
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from skimage import img_as_ubyte
import pymysql

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from mrcnn.config import Config
from mrcnn import model as modellib, utils

def cat_model_initialize():
    cfg = get_cfg()
    # SEGMENTATION
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return cfg, predictor

def human_model_initialize():
    cfg = get_cfg()
    # KEYPOINTS
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return cfg, predictor

class NumplateConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    NAME = "numplate"
    GPU_COUNT = 2
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.7
class InferenceConfig(NumplateConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
def within(x, MAX):
    if x<0:
        return int(0)
    elif x>MAX:
        return int(MAX)
    else:
        return int(x)

def get_image_list():
    pass

def update_data(id):
    pass

def main():
    numplate_config = InferenceConfig()
    numplate_model = modellib.MaskRCNN(mode="inference", config=numplate_config, model_dir=os.path.abspath('./model'))
    numplate_weights_path = os.path.abspath('./model/mask_rcnn_numplate_0030.h5')
    numplate_model.load_weights(numplate_weights_path, by_name=True)

    car_cfg, car_predictor = cat_model_initialize()
    human_cfg, human_predictor = human_model_initialize()

    with torch.no_grad():
        for image_info in get_image_list():
            image_file_name, image_file_ext = os.path.splitext(image_info[1])
            image_thumbnail = image_file_name + '_800' + image_file_ext
            image_org_thumbnail = image_file_name + '_800_org' + image_file_ext
            image_path = os.path.join('pass', image_info[0], image_org_thumbnail)
            if not os.path.exists(image_path):
                image_path = os.path.join('pass', image_info[0], image_thumbnail)

            car_outputs = car_predictor(read_image(image_path))
            car_pred_classes = car_outputs['instances'].pred_classes.cpu().numpy().tolist()
            car_pred_boxes = car_outputs['instances'].pred_boxes.tensor.cpu().numpy().tolist()
            car_idx_j = 0
            for i in range(len(car_pred_classes)):
                if not car_pred_classes[car_idx_j] in [2, 3, 5, 7]:
                    car_pred_classes.pop(car_idx_j)
                    car_pred_boxes.pop(car_idx_j)
                else:
                    car_idx_j += 1
            org_image = skimage.io.imread(image_path)
            car_image = org_image.copy()
            for box in car_pred_boxes:
                x1,y1,x2,y2 = box
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

                if x1 < 0:
                    x1 = int(0)
                if y1 < 0:
                    y1 = int(0)

                cropped = car_image[y1:y2, x1:x2]
                r = numplate_model.detect([cropped], verbose=0)[0]
                mask = r['masks']
                if mask.shape[-1] > 0:
                    mask = (np.sum(mask, -1, keepdims=True) >= 1)
                    blurred = skimage.filters.gaussian(cropped, sigma=5, truncate=3.5, multichannel=True, preserve_range=True)
                    result = np.where(mask, blurred, cropped).astype(np.uint8)
                else:
                    result = cropped

                car_image[y1:y2, x1:x2] = result
            human_image = car_image[:, :, ::-1]
            height, width, channel = human_image.shape
            human_outputs = human_predictor(read_image(image_path))
            human_keypoints = human_outputs["instances"].pred_keypoints.cpu().numpy()
            if len(human_keypoints) > 0:
                for i in range(len(human_keypoints)):
                    pred_keypoints = human_keypoints[i]
                    nose = pred_keypoints[0]
                    Ltop = (within(nose[0]-10,width),within(nose[1]-10,height))
                    Rbtm = (within(nose[0]+10,width),within(nose[1]+10,height))
                    ROI = human_image[Ltop[1]:Ltop[1] + 20, Ltop[0]:Ltop[0] + 20]
                    ROI = cv2.GaussianBlur(ROI, (15, 15), 0)
                    human_image[Ltop[1]:Ltop[1] + 20, Ltop[0]:Ltop[0] + 20] = ROI

            out_image_org_thumbnail = os.path.join('pass', image_info[0], image_org_thumbnail)
            out_image_thumbnail = os.path.join('pass', image_info[0], image_thumbnail)

            cv2.imwrite(out_image_org_thumbnail, org_image[:, :, ::-1])
            cv2.imwrite(out_image_thumbnail, human_image)

            update_data(image_info[2])

            print(out_image_thumbnail)


if __name__ == "__main__":
    main()
