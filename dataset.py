from pycocotools.coco import COCO
import numpy as np
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import paddle
from paddle.io import Dataset
import json
from paddlenlp.transformers import GPTLMHeadModel, GPTTokenizer
from paddlenlp.transformers import CLIPProcessor, CLIPModel
import paddle.distributed as dist


class CaptionDataset(Dataset):
    def __init__(self, transform=None):
        pylab.rcParams['figure.figsize'] = (8.0, 10.0)
        annFile = "./captions_train2014.json"
        self.coco = COCO(annFile)
        self.imgIds = self.coco.getImgIds(imgIds=list(self.coco.imgs.keys()))
        self.img_dict_list = self.coco.loadImgs(self.imgIds)
        self.clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.lm_tokenizer = GPTTokenizer.from_pretrained('gpt2-medium-en')
        self.root_image_path = "~/data/d/train2014/"
        self.img_path_list = [self.root_image_path + one_img_dict["file_name"] for one_img_dict in self.img_dict_list]
        annIds_list = [self.coco.getAnnIds(imgIds=one_img_dict['id']) for one_img_dict in self.img_dict_list]
        anns_list = [self.coco.loadAnns(annIds) for annIds in annIds_list]
        self.captions_list = [[ann["caption"] for ann in anns] for anns in anns_list]
        self.img_path_list0 = []  # 保存图片的list
        self.captions_list0 = []  # 保存caption的List
        for i in range(len(self.captions_list)):
            len_i = len(self.captions_list[i])
            self.img_path_list0.extend([self.img_path_list[i]] * len_i)
            for j in range(len_i):
                self.captions_list0.append(self.captions_list[i][j])
        self.max_sentence_len = 50

    def __getitem__(self, idx):
        img_path = self.img_path_list0[idx]
        one_img = cv2.imread(img_path)
        clip_img = self.clip_preprocess(images=one_img, return_tensors="pd")["pixel_values"]
        sentence_token = self.lm_tokenizer(self.captions_list0[idx])
        return clip_img, np.array(sentence_token["input_ids"][:self.max_sentence_len])

    def __len__(self):
        return len(self.captions_list0)


if __name__ == '__main__':
    dataset = CaptionDataset()
    for one_image, caption_list in dataset:
        print(one_image.shape, caption_list)
        break
