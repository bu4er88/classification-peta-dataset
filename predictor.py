import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch import nn
import cv2
from datetime import date
import traceback
from PIL import Image

import seedoo.vision
import sys
path = os.path.dirname(os.path.abspath(seedoo.vision.__file__))
sys.path.append(path)
from seedoo.vision.utils.image_loading import ImageLoader, ListOfImageLoaders


def logit_writer(image, row):
    logit = str(row.pred_logit_1.item())
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.35
    thickness = 1
    org1 = (int(image.shape[1]/10), int(image.shape[0] / 10))
    image = cv2.putText(image, logit, org1, font, fontScale, (255, 255, 255), thickness, cv2.LINE_8)
    return image


def wrapper(row):
    r = pd.Series(row.to_dict().copy())
    def drawer():
        try:
            path = r.image_path
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            color_1 = (225, 0, 0) if r.is_mistake_1 else (0, 225, 0)
            cv2.circle(image, (image.shape[1]-int(image.shape[1]/10), image.shape[0]-int(image.shape[0]/20)),
                       int(image.shape[0]/25), color=color_1, thickness=-1)
            image = logit_writer(image, r)
            return image
        except Exception as e:
            print(traceback.format_exc(), r.image)
            return np.zeros(shape=(5, 5, 3))
    return drawer


def count_f1(row, true_column_name, pred_column_name):
    return f1_score(row[true_column_name].astype(float),
                    row[pred_column_name].astype(float),
                    average='binary',
                    zero_division=0)


def get_labels(x, labels):
    """Used by advanced_analysis_generator()"""
    l = []
    for i, val in enumerate(x):
        if val == 1:
            l.append(labels[i])
    return l


def one_hot_tensor(tensor):
    one_hot = torch.zeros_like(tensor, dtype=torch.float32)[0]
    one_hot[torch.argmax(tensor)] = 1
    return one_hot


def f1_html_from_dict_generator(data: dict, output_name: str, head_name: dict):
    print('Run f1_html_from_dict_generator...')
    data_renamed = {head_name[key]: value for key, value in data.items()}
    df = pd.DataFrame.from_dict(data_renamed, orient='index').rename(columns={'1<len<4': '1>len>4'})
    html = df.reset_index().to_html(escape=False, index=False)
    directory = os.getcwd()
    file_path = os.path.join(directory, output_name)
    with open(file_path, 'w') as f:
        f.write(html)


def html_cm_exporter(df: pd.DataFrame, head_number: int, head_name: dict):
    """Exporting rendered html with confusion matrix"""
    conditions = ['len==1', '1<len<4', 'len>3']
    for cond in conditions:
        # filter df with the condition
        df_1 = df.query(cond).copy()
        html = pd\
            .crosstab(df_1[f'true_{head_number}_decoded'].astype(str),
                      df_1[f'pred_{head_number}_decoded'].astype(str)
                      ) \
            .reset_index() \
            .to_html(escape=False, index=False)
        file_name = f"head_{head_name}_{cond}.html"
        directory = os.getcwd()
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'w') as f:
            f.write(html)
        # export table with visualised
        html = pd \
            .crosstab(df_1[f'true_{head_number}_decoded'].astype(str),
                      df_1[f'pred_{head_number}_decoded'].astype(str),
                      values=df_1.visualized,
                      aggfunc=lambda x: ListOfImageLoaders(
                          x.sample(min(50, len(x))).values.tolist(), columns_per_row=10)
                      ) \
            .reset_index() \
            .to_html(escape=False, index=False)
        file_name = f"head_{head_name}_{cond}_visualized.html"
        directory = os.getcwd()
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'w') as f:
            f.write(html)


def advanced_analysis_generator(df: pd.DataFrame,
                                labels: list,
                                head_number: int,
                                head_len: int,
                                head_type: str,
                                head_name: str) -> pd.DataFrame:
    """
    Creates additional columns and adds confusion matrix with images
    for the dataframe with predicts for validation subset.

    "head_type" can be:
        - 'binary' for binary output with one neuron, like (1) or (0);
        - 'one-hot-single' for one-hot with argmax output, like [1,0,0,0,0];
        - 'one-hot-multiple' for one-hot with sigmoid output, like [1,0,1,1,0].
    """
    print(f'Running advanced_analysis_generator for HEAD_{head_number}..')

    labels = {k: v for k, v in enumerate(labels)}

    if head_type == 'binary':
        df[f'pred_{head_number}_decoded'] = df[f'pred_{head_number}'].apply(lambda x: labels[int(x)])
        df[f'true_{head_number}_decoded'] = df[f'true_{head_number}'].apply(lambda x: labels[int(x)])
    elif head_type == 'one-hot-single':
        df[f'pred_{head_number}_decoded'] = df[f'pred_{head_number}'].apply(lambda x: labels[int(np.argmax(x))])
        df[f'true_{head_number}_decoded'] = df[f'true_{head_number}'].apply(lambda x: labels[int(np.argmax(x))])
    elif head_type == 'one-hot-multiple':
        df[f'pred_{head_number}_decoded'] = df[f'pred_{head_number}'].apply(lambda x: get_labels(x, labels))
        df[f'true_{head_number}_decoded'] = df[f'true_{head_number}'].apply(lambda x: get_labels(x, labels))
    else:
        raise "Unsupported head_type in CONFIG.py! " \
              "Supported head_types: 'binary', 'one-hot-single', 'one-hot-multiple'"

    # adding "is_mistake" column
    if head_type == 'binary':
        df[f'is_mistake_{head_number}'] = df[f'pred_{head_number}_decoded'] != df[f'true_{head_number}_decoded']
    elif head_type == 'one-hot-single':
        df[f'is_mistake_{head_number}'] = df[f'pred_{head_number}_decoded'] != df[f'true_{head_number}_decoded']
    else:
        pass
        # TODO: add calculation of F1_score per label!

    df = df.rename(columns={'predicted_on': 'image'})
    df = df.explode('image')
    df['visualized'] = df.image.apply(lambda x: ImageLoader(x))
    if head_len < 20:
        html_cm_exporter(df, head_number, head_name)
    else:
        print(f"Skipping html and excel generation, because "
              f"head_{head_number} has more that 20 labels: {head_len} labels!")
    return df


def predictor(model,
              best_model_wts,
              data_transforms,
              df: pd.DataFrame,
              device: str,
              groups: int,
              label_column_dict: dict,
              head_type: dict,
              head_name: dict,
              height: int,
              width: int
              ):
    """Get validation dataframe and generate prediction dataframe for it"""
    print('-' * 10)
    print("\nRunning the in-line predictor module..")

    assert isinstance(df, pd.DataFrame), "The 'df' must be pandas.DataFrame"

    state_dict = torch.load(best_model_wts, map_location=torch.device(device))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy_input = torch.Tensor(1, 3 * groups, height, width).to(device)
    outputs = model(dummy_input)
    head_len = {head: len(label_column_dict[head]) for head in sorted(label_column_dict.keys())}
    assert len(outputs) == len(head_len), "Model output length doesn't equal number of heads in CONFIG.py"

    list_of_parameters = ['pred', 'true', 'pred_logit', 'component_id', 'predicted_on', 'len', 'width', 'height']

    parameters = {}
    for i in range(1, len(head_len) + 1):
        for param in list_of_parameters[:3]:
            parameters[param + f'_{i}'] = []
    for param in list_of_parameters[3:]:
        parameters[param] = []

    for component_id, rows in df.iterrows():  # len(df)
        images = rows.image
        parameters['predicted_on'].append(images)  # images_in_component.append(images)
        parameters['len'].append(len(images))          # length.append(len(images))

        if len(images) < groups:
            n = groups - len(images)
            images = np.append(images, np.random.choice(images, n))
        else:
            images = np.random.choice(images, groups)

        transform = lambda x: data_transforms(image=cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB))['image']
        tensor_image_group = torch.cat([transform(image) for image in images]).unsqueeze(0).to(device)

        outputs = model(tensor_image_group)

        parameters['component_id'].append(component_id)
        for i, head in enumerate(sorted(label_column_dict.keys())):
            parameters[f'true_{i + 1}'].append(np.asarray(rows[head]).max(axis=0))
            parameters[f'pred_{i + 1}'].append(one_hot_tensor(outputs[i]).float().detach().numpy())
            parameters[f'pred_logit_{i + 1}'].append(outputs[i].detach().cpu().numpy()[0])
        width, height = np.asarray([Image.open(image).size for image in images]).mean(axis=0)
        parameters['width'].append(round(width, 0))
        parameters['height'].append(round(height, 0))

    res_df = pd.DataFrame(parameters)

    # f1 scores calculation (f1_ij: f1_11, f1_12, f1_13, f1_14, f1_21, ...)
    filters = {
        'len': ['len>0', 'len==1', '1<len<4', 'len>3'],
        'imsize': ['width >= 75 and height >= 160', 'width < 75 and height < 160']}
    f1_dict = {}
    # late to use: sorted(head_type.keys()) instead of sorted(label_column_dict.keys())!
    for head in sorted(label_column_dict.keys()):
        f1_dict[head] = {}
        print(f'prediction of F1 score for {head}...')
        i = head.split('_')[-1]
        for filt in filters['len']:
            # # TODO: add head_type logic for calculation if the F1
            # # calculate for 'binary'
            # if head_type == 'binary':
            #     f1_dict[f'f1_{head}_{filt}'] = f1_score(
            #         res_df.query(filt)[f'true_{i}'].values,
            #         res_df.query(filt)[f'pred_{i}'].values.astype(float),
            #         average='binary')
            #     f1_dict[f'f1_{head}_big_img'] = f1_score(
            #         res_df.query(filters['imsize'][0]).true_1.values,
            #         res_df.query(filters['imsize'][0]).pred_1.values.astype(float),
            #         average='binary')
            #     f1_dict[f'f1_{head}_small_img'] = f1_score(
            #         res_df.query(filters['imsize'][1]).true_1.values,
            #         res_df.query(filters['imsize'][1]).pred_1.values.astype(float),
            #         average='binary')
            # else:
            # calculate for the other heads (with > 1 output logits)
            f1_dict[head][filt] = res_df\
                .query(filt)\
                .apply(lambda x: count_f1(x, f'true_{i}', f'pred_{i}'), axis=1)\
                .mean()
        f1_dict[head][f'f1_big_img'] = res_df \
            .query(filters['imsize'][0]) \
            .apply(lambda x: count_f1(x, f'true_{i}', f'pred_{i}'), axis=1) \
            .mean()
        f1_dict[head][f'f1_small_img'] = res_df \
            .query(filters['imsize'][1]) \
            .apply(lambda x: count_f1(x, f'true_{i}', f'pred_{i}'), axis=1) \
            .mean()

    # export rendered html with f1 for each head split by filters
    f1_html_from_dict_generator(data=f1_dict,
                                output_name="f1_table_visualized.html",
                                head_name=head_name)

    # export others html with cm tables along with visualized
    for head in sorted(label_column_dict.keys()):
        advanced_analysis_generator(df=res_df,
                                    labels=label_column_dict[head],
                                    head_number=head.split('_')[-1],
                                    head_len=head_len[head],
                                    head_type=head_type[head],
                                    head_name=head_name[head])
    today = str(date.today())
    res_df.to_pickle(f'./df_val_pred-{today}.pkl')
    print("In-line predictor module is finished!\n")
    return res_df, f1_dict
