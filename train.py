from __future__ import print_function, division
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
import torch.utils.data as data
import time
import os
import torch
import argparse
from tqdm import tqdm
import copy
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, accuracy_score
from model_constructor_grouped import model_constructor_grouped

from losses import FocalLoss, sigmoid_focal_loss, multilabel_focal_loss, CELoss
from dataframe_maker import DataFrameMakerListing
import cv2
import random
from predictor import predictor
from CONFIG import label_constructor

# from cnvrgv2 import Experiment
# e = Experiment()


def geom_mean(*args):
    arr = np.array(args)
    return arr.prod() ** (1.0 / len(arr))


def count_f1_for_onehot(true, pred):
    return np.asarray([f1_score(l, p, average='binary', zero_division=0) for l, p in zip(true, pred)]).mean()


class SeeDooPetaDataset(data.Dataset):

    def __init__(self,
                 label_column_dict,
                 head_type,
                 df,
                 data_transforms,
                 height,
                 width,
                 groups
                 ):
        print(f"Initialising dataset class for {len(label_column_dict)}...")
        self.label_column_dict = label_column_dict
        self.head_type = head_type
        self.df = df
        self.height = height
        self.width = width
        self.groups = groups
        self.transforms = data_transforms
        self.classes = {key: len(label_column_dict[key]) for key in self.label_column_dict.keys()}

    def sp_transform(self, tensor, r1=0.9, r2=1.1, p=0.5):
        """Add Salt and Pepper transformation"""
        random_noise = (r1 - r2) * torch.rand(size=tensor.shape) + r2
        if random.random() < p:
            tensor = torch.clamp(tensor * random_noise, 0, 1)
        return tensor

    def __getitem__(self, index):
        row = self.df.iloc[index]
        images = row.image
        if len(images) < self.groups:
            n = self.groups - len(images)
            images = np.append(images, np.random.choice(images, n, replace=True))
        else:
            images = np.random.choice(images, self.groups, replace=False)
        transform = lambda x: self.transforms(image=cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB))['image']
        tensor_image_group = torch.cat([self.sp_transform(transform(img)) for img in images])
        true_labels = [
            torch.from_numpy(np.asarray(row[head]).max(axis=0).astype(float))
            for head in sorted(self.label_column_dict.keys())
        ]
        return tensor_image_group, true_labels

    def __len__(self):
        return len(self.df)


def one_hot_tensor(tensor):
    zeros = torch.zeros_like(tensor, dtype=torch.float32)
    indexes = torch.argmax(tensor, 1)
    for dim_0, dim_1 in enumerate(indexes):
        zeros[dim_0, dim_1] = 1.
    return zeros


def run(
        batch_size=None,
        learning_rate=None,
        epochs=None,
        annotations_file=None,
        num_workers=None,
        height=None,
        width=None,
        device='auto',
        loss=None,
        focal_alpha=1.,
        focal_gamma=None,
        model=None,
        unfreeze=0,
        forced=0,
        dropout=0.,
        weights="best.pt",
        optimizer_step=1,
        optimizer_gamma=1.0,
        index=None,
        groups=3,
        condition=None,
        dataset_path=None,
        image_prunner_column=None,
        label_pruner_column=None,
        labels_intersection=None
):
    # activate CONFIG
    label_column_dict, head_type, head_name = label_constructor()

    # DATAFRAME_READER
    df = DataFrameMakerListing(annotations_file,
                               label_column_dict,
                               head_type,
                               index,
                               condition,
                               dataset_path,
                               image_prunner_column,
                               label_pruner_column,
                               labels_intersection
                               ).df
    # df.to_csv(f'df_{condition}.csv')
    # if condition is None:
    #     a = df.shape[0]
    #     df = df.query('HEAD_1 == 0. or HEAD_1 == 1.')
    #     print(f'{a - df.shape[0]} components with bad gender were deleted!')

    # DATA AUGMENTATION
    if model == 'densenet121':
        height = width = 256

    augmentation_train = A.Compose([
        A.OneOf([A.ChannelShuffle(p=0.5), A.ToGray(p=0.5)], p=0.6),
        A.Resize(height, width, interpolation=1, p=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    augmentation_val = A.Compose([
        # A.PadIfNeeded(border_mode=cv2.BORDER_CONSTANT, p=1),
        # A.Resize(int(height*1.1), int(width*1.1), interpolation=1, p=1),
        A.Resize(height, width, interpolation=1, p=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    data_transforms = {
        'train': A.Compose(augmentation_train),
        'val': A.Compose(augmentation_val),
    }
    df_train = df.query('set=="train"')
    df_val = df.query('set=="val"')

    if index:
        image_datasets = dict(
        train=SeeDooPetaDataset(
            label_column_dict, head_type, df_train, data_transforms['train'], height, width, groups
        ),
        val=SeeDooPetaDataset(
            label_column_dict, head_type, df_val, data_transforms['val'], height, width, groups
        )
    )
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        ) for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    if device == 'auto':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = device
    print(f'Using device: {device}')
    print('Detecting target labels...')
    NUM_OF_CLASSES = {key: len(label_column_dict[key]) for key in sorted(label_column_dict.keys())}
    for head, lenght in NUM_OF_CLASSES.items():
        print(f'For {head} detected {lenght} labels')
    print('Training parameters have been set!')

    # Training loop
    def train_model(model, optimizer, scheduler, num_epochs, criterion):
        start_training = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        f1_mean_geom = 0.0
        best_f1 = 0.0

        if forced == 1:
            print("Creating a new 'val_results' table...")
            charts = pd.DataFrame([])
        else:
            if os.path.exists("charts.csv"):
                print("Continue filling 'charts' table...")
                charts = pd.read_csv('charts.csv')
            else:
                print("Creating a new 'charts' table...")
                charts = pd.DataFrame([])

        # Run inline predictor. It exports rendered html for analysis
        for epoch in range(num_epochs):

            # inline predictor with exporting html files
            if epoch > 0 and epoch % 10 == 0.:
                res_df, f1_dict = predictor(model=model,
                                            best_model_wts=f'best.pt',
                                            data_transforms=data_transforms['val'],
                                            df=df_val,
                                            device=device,
                                            groups=groups,
                                            label_column_dict=label_column_dict,
                                            head_type=head_type,
                                            head_name=head_name,
                                            height=height,
                                            width=width
                                            )
                # Loop for all heads
                for head in sorted(label_column_dict.keys()):
                    h = head_name[head]
                    for filt in f1_dict[head]:
                        print(f"cnvrg_linechart_f1_{h}_{filt}# value: {f1_dict[head][filt]}")

            # Training
            print(f'Epoch {epoch} of {num_epochs - 1}')
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode
                # Refresh metrics
                running_total_loss = 0.
                running_losses = {head: 0. for head in label_column_dict.keys()}
                running_f1s = {head: [] for head in label_column_dict.keys()}
                # Iterate over data
                for inputs, true_labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    true_labels = [label.to(device) for label in true_labels]
                    optimizer.zero_grad()
                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        # preds = [torch.round(nn.Sigmoid()(output)).float() for output in outputs]
                        preds = [one_hot_tensor(output).float() for output in outputs]

                    # Applying loss function
                        if criterion == 'sigmoid_focal_loss':
                            losses = [
                                sigmoid_focal_loss(
                                    outputs[i], true_labels[i].float(), alpha=focal_alpha, gamma=focal_gamma,
                                    reduction='mean'
                                ) for i in range(len(outputs))
                            ]
                        elif criterion == 'multilabel_focal_loss':
                            losses = [
                                multilabel_focal_loss(
                                    outputs[i], true_labels[i].float(), alpha=focal_alpha, gamma=focal_gamma
                                ) for i in range(len(outputs))
                            ]
                        else:
                            losses = [
                                criterion['ce'](outputs[i], true_labels[i].float()) for i in range(len(outputs))
                            ]
                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            # if trainable focal_gamma, add this to the loss:
                            # + torch.log(model.gamma_1 + model.gamma_2 + model.gamma_3 + model.gamma_4) / 10
                            total_loss = (sum(losses)) / len(outputs)
                            total_loss.backward()
                            optimizer.step()
                    # Calculate running metrics
                    running_total_loss += total_loss.item() * inputs.size(0)
                    for i in range(1, len(label_column_dict) + 1):
                        running_losses[f'HEAD_{i}'] += losses[i - 1].item() * inputs.size(0)
                        running_f1s[f'HEAD_{i}'].append(
                            count_f1_for_onehot(true_labels[i - 1].cpu(), preds[i - 1].detach().cpu())
                        )
                if phase == 'train':
                    scheduler.step()

                # Calculate epoch metrics
                epoch_loss = running_total_loss / dataset_sizes[phase]
                epoch_losses = [running_losses[f'HEAD_{i}'] / dataset_sizes[phase] for i in range(1, len(losses) + 1)]
                epoch_f1s = [np.array(running_f1s[f'HEAD_{i}']).mean() for i in range(1, len(losses) + 1)]
                # TODO: print out right with autogenerated heads pipeline
                print(f"{phase.upper()} loss:{epoch_loss:.7f}")
                for i in range(len(epoch_losses)):
                    h = head_name[f"HEAD_{i + 1}"]
                    print(f"loss_head_{i + 1}_{h}: {epoch_losses[i]:.7f}")
                print()
                print(f"cnvrg_linechart_Loss#{phase} value: {epoch_loss}")
                for i, head in enumerate(sorted(label_column_dict.keys())):
                    h = head_name[head]
                    print(f"cnvrg_linechart_f1_{h}#{phase} value: {epoch_f1s[i]}")

                # deep copy the model
                if phase == 'val':
                    # TODO: calculate f1_mean_geom right with autogenerated heads pipeline
                    f1_mean_geom = geom_mean(*epoch_f1s)
                    print(f'val F1 geometric mean: {f1_mean_geom}')
                    # f1_mean_geom = geom_mean(epoch_f1_1, epoch_f1_2, epoch_f1_3, epoch_f1_4)
                    if f1_mean_geom > best_f1:
                        best_f1 = f1_mean_geom
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(best_model_wts, f'best.pt')

                # save dataframe with results
                # output_df.to_csv('val_results.csv')

                # TODO: TEST charts_epoch with autogenerated heads pipeline
                charts_data = {}
                charts_data['phase'] = [phase]
                charts_data.update({f'loss_{i + 1}': [loss] for i, loss in enumerate(epoch_losses)})
                charts_data.update({f'loss_{i + 1}': [loss] for i, loss in enumerate(epoch_losses)})
                charts_data.update({f'f1_{i + 1}': [f1] for i, f1 in enumerate(epoch_f1s)})
                charts_epoch = pd.DataFrame(charts_data)
                charts = pd.concat([charts, charts_epoch], axis=0)
                charts.to_csv('charts.csv')

            print()

        time_elapsed = time.time() - start_training
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best validation F1 score: {best_f1:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, best_f1

    # MODEL
    model = model_constructor_grouped(model, unfreeze, groups, NUM_OF_CLASSES, dropout, label_column_dict)

    if forced == 1:
        print("*~*~*~*~* Force start training from scratch! *~*~*~*~*")
        if os.path.exists('best.pt'):
            print('*~*~*~*~* Removing existing "best.pt" weights! *~*~*~*~*')
            os.remove('best.pt')
    elif os.path.exists('best.pt'):
        print('*~*~*~*~* Resuming after spot restart from "best.pt"! *~*~*~*~*')
        state_dict = torch.load('best.pt', map_location=torch.device(device))
        model.load_state_dict(state_dict, strict=False)
    elif os.path.exists(weights):
        print(f"*~*~*~*~* Training can't resumed from best.pt! Resuming from weights {weights}! *~*~*~*~*")
        state_dict = torch.load(weights, map_location=torch.device(device))
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"*~*~*~*~* WARNING: couldn't resume training. Starting training from scratch! *~*~*~*~*")

    model = model.to(device)

    # SET LOSS FINCTION
    if loss == 'focal':
        focal_gamma = focal_gamma if focal_gamma else 0
        print(f'Set gamma for focal loss is: {focal_gamma}')
        assert focal_gamma > -1, f"Gamma value must be >= 0, but set {focal_gamma}"

    losses = dict({
        'focal': FocalLoss(focal_gamma=focal_gamma),
        'ce': nn.CrossEntropyLoss(),
        'bce': nn.BCELoss(),
        'sigmoid_focal_loss': 'sigmoid_focal_loss',
        'multilabel_focal_loss': 'multilabel_focal_loss'
    })

    criterion = losses[loss]

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate)
    # Decay LR by a factor of scheduler_step every scheduler_gamma epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=optimizer_step, gamma=optimizer_gamma
    )

    print("Start training...")
    model, best_f1 = train_model(
        model, optimizer_ft, exp_lr_scheduler, num_epochs=epochs, criterion=criterion,
    )
    torch.save(model.state_dict(), f'best_model_wts_{weights}')
    # e.log_param("F1", best_f1)


def main():
    run(
        batch_size=512,
        num_workers=12,
        learning_rate=0.0001,
        epochs=10,
        annotations_file='merged_peta_bad_component_geom_std_var.csv',  #'peta_merged_updated.csv',
        height=10,
        width=5,
        loss='multilabel_focal_loss',
        focal_alpha=1.,
        focal_gamma=3,
        model='mobilenet_v2',
        unfreeze=200,
        forced=0,
        device='auto',
        dropout=0.8,
        weights='model-images.pt',
        optimizer_step=1,
        optimizer_gamma=0.98,
        index='component_id',
        groups=3,
        condition='isBadComponent_var_geom_threshold_20', #~(set=="train" and isBadComponent_var_threshold_30==1)' , #'isBadComponent_var_geom_threshold_20', #isBadComponent_var_threshold_30, isBadComponent_var_geom_threshold_20
        dataset_path='./peta/images',
        # image_prunner_column='goodLabels_ratio_threshold_30',
        # label_pruner_column='goodLabels_ratio_threshold_60',,
        labels_intersection=0
    )


if __name__ == "__main__":
    main()


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--batch_size', type=int, required=True)
#     opt = parser.parse_args()
#     return opt
#
# def main(opt):
#     run(**vars(opt))
#
# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)
#
# python3 train_2_heads_package_input.py --batch_size 32
