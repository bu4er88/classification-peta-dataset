from __future__ import print_function, division
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from CONFIG import label_constructor
import torchvision
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import cv2
import argparse


class MobileNetV2_custom(nn.Module):
    """
    num_classes_1 - classes for "gender" label
    num_classes_2 - classes for "reset_labels" label
    num_classes_2 - classes for "age" label
    num_classes_2 - classes for "accessory" label
    """
    def __init__(self, num_classes, dropout, groups, label_column_dict):
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True).features
        self.model[0][0] = nn.Sequential(*[
            nn.Conv2d(groups*3, 27, groups=groups, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(27, 32, groups=1, kernel_size=1, stride=1, padding=0, bias=False),
        ])
        self.size = torchvision.models.mobilenet_v2().classifier[-1].weight.shape[1]
        self.dropout = nn.Dropout(p=dropout)
        self.classifiers = [nn.Linear(self.size, len(num_classes[head])) for head in sorted(list(num_classes.keys()))]
        self.classifiers = torch.nn.ModuleList(self.classifiers)

    def forward(self, x):
        x = self.model(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        features = torch.flatten(x, 1)
        x = self.dropout(features)
        outputs = [classifier(x) for classifier in self.classifiers]
        return outputs, features


def dataframe_transformer(df, path_to_data, group_column, agg_column):
    images_path = os.path.join(path_to_data, 'images')
    df = pd.read_csv(df).head(100)
    df['image_new'] = df[agg_column].apply(lambda x: os.path.join(images_path, os.path.basename(x)))
    return df.groupby(group_column).agg({'image_new': 'unique'})


def input_tranformer(df, index, data_transforms, groups=3):
    row = df.iloc[index]
    images = row.image_new
    if len(images) < groups:
        n = groups - len(images)
        images = np.append(images, np.random.choice(images, n, replace=True))
    else:
        images = np.random.choice(images, groups=3, replace=False)
    transform = lambda x: data_transforms(image=cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB))['image']
    tensor_image_group = torch.cat([transform(img) for img in images])
    return tensor_image_group


def extract_features(model, data_transforms, df, device, groups, height, width, label_column_dict, head_name):
    """Get validation dataframe and generate prediction dataframe for it"""
    print("\nRun feature predictor module..")
    assert isinstance(df, pd.DataFrame), "The 'df' must be pandas.DataFrame"
    model.eval()
    dummy_input = torch.Tensor(1, 3 * groups, height, width).to(device)
    outputs = model(dummy_input)
    assert outputs is not None
    parameters = {'component_id': [], 'features': []}
    for head in head_name:
        name = head_name[head]
        parameters[name] = []
    for component_id, rows in tqdm(df.iterrows()):
        images = rows.image_new
        if len(images) < groups:
            n = groups - len(images)
            images = np.append(images, np.random.choice(images, n))
        else:
            images = np.random.choice(images, groups)
        transform = lambda x: data_transforms(image=cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB))['image']
        tensor_image_group = torch.cat([transform(image) for image in images]).unsqueeze(0).to(device)
        outputs, features = model(tensor_image_group)
        outputs = [out.detach().cpu().numpy() for out in outputs]

        # labels = {'head_1': ['a', 'b', 'c'], 'head_2': ['x','y','z'], 'head_3': ['m', 'n', 'o']}
        labels = label_column_dict
        logits = outputs
        # logits = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result_dict = {}
        for i, head in enumerate(sorted(labels)):
            result_dict[head] = {}
            for j, label in enumerate(labels[head]):
                result_dict[head][label] = logits[i][0][j]
        for head, name in sorted(list(head_name.items())):
            parameters[name].append(result_dict[head])
        parameters['component_id'].append(component_id)
        parameters['features'].append(features.detach().cpu().numpy())
    res_df = pd.DataFrame(parameters)
    print("Embedding predictor module is finished!\n")
    return res_df


def run(
    df,
    path_to_data,
    height,
    width,
    weights,
    index,
    agg_column,
    groups
):
    # transform datafraem
    df = dataframe_transformer(df, path_to_data, group_column=index, agg_column=agg_column)
    # set input data augmentstions
    augmentation_val = A.Compose([A.Resize(height, width, interpolation=1, p=1),
                                  A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                  ToTensorV2()])
    # data_transforms = {A.Compose(augmentation_val)}
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set feature extractor for embedding layer only
    label_column_dict, head_type, head_name = label_constructor()
    extractor = MobileNetV2_custom(num_classes=label_column_dict, dropout=0.8, groups=3,
                                   label_column_dict=label_column_dict
                                   )
    # load weights
    # weights_path = os.path.join(path_to_data, weights)
    state_dict = torch.load(weights, map_location=torch.device(device))
    extractor.load_state_dict(state_dict, strict=False)
    # prediction
    res_df = extract_features(extractor, augmentation_val, df, device, groups, height, width,
                              label_column_dict, head_name
                              )
    res_df.to_csv('predicted_features.csv')
    res_df.to_pickle('predicted_features.pkl')
    print('FINISHED!')


# def main():
#     run(df='/Users/yauhenikavaliou/Desktop/SeeDoo/code/images/images/merged_peta_bad_component_geom_std_var.csv',
#         path_to_data='/Users/yauhenikavaliou/Desktop/SeeDoo/code/images/images/',
#         height=210,
#         width=110,
#         weights='best.pt',
#         index='component_id',
#         agg_column='image_path',
#         groups=3)
#
#
# if __name__ == "__main__":
#     res_df = main()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--df', type=str, required=True)
    parser.add_argument('--path_to_data', type=str, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--index', type=str, required=True)
    parser.add_argument('--agg_column', type=str, required=True)
    parser.add_argument('--groups', type=int, required=True)
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)







