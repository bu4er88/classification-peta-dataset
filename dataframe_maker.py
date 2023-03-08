import pandas as pd
import os
import numpy as np
from tqdm import tqdm


def get_intersection(*sets):
    intersected_labels = sets[0].intersection(*sets[1:])
    return intersected_labels


class DataFrameMakerListing:

    def __init__(self,
                 filename,
                 label_column_dict,
                 head_type,
                 idx,
                 condition,
                 dataset_path,
                 image_prunner_column,
                 label_pruner_column,
                 labels_intersection,
                 train=0.8,
                 val=0.2,
                 test=0.,
                 ):
        print("DataFrameMaker is making a dataframe..")
        self.df = self.df_reader(filename)
        self.input_df = self.df.copy()
        self.label_column_dict = label_column_dict
        self.head_type = head_type
        self.idx = idx
        self.condition = condition
        self.image_prunner_column = image_prunner_column
        self.label_pruner_column = label_pruner_column
        self.train = train
        self.val = val
        self.test = test
        self.labels_intersection = labels_intersection
        self.labels = sorted(self.label_column_dict.keys())
        self.dataset_path = dataset_path
        self._data_transformer()

    @staticmethod
    def df_reader(filename):
        """Read dataframe"""
        print('Run df_reader..')
        route = os.path.join(os.getcwd(), 'peta')
        if filename.split('.')[-1] == 'csv':
            return pd.read_csv(os.path.join(route, filename))
        if filename.split('.')[-1] == 'pkl':
            return pd.read_pickle(os.path.join(route, filename))

    def _component_label_intersector(self):
        print(f'Run _component_label_intersector..')
        labels = pd.read_pickle('labels.pkl').iloc[:, 0].to_list()
        shape_before = self.df.shape
        df_singletons = self.df[self.df['len'] == 1]
        df_other = self.df[self.df['len'] > 1]
        bad_components = []
        for id in tqdm(df_other['component_id'].unique()):
            all_component_labels_list = [l[1] for l in self.df[self.df['component_id'] == id][labels].iterrows()]
            component_labels = []
            for image_labels in all_component_labels_list:
                l = {v for v in image_labels.to_dict() if image_labels.to_dict()[v] == 1}
                component_labels.append(l)
            intersected_labels = get_intersection(*component_labels)
            if not intersected_labels:
                bad_components.append(id)
            else:
                good_labels = list(intersected_labels)
                df_other.loc[df_other['component_id'] == id, labels] = 0
                df_other.loc[df_other['component_id'] == id, good_labels] = 1
        if bad_components:
            print(f'Detected {len(bad_components)} bad components: {bad_components}')
            print(f'Bad components lengths: {[len(df_other[df_other.component_id == id]) for id in bad_components]}')
        df_other = df_other.query(f'~component_id.isin({bad_components})')
        self.df = pd.concat([df_other, df_singletons], ignore_index=True)
        assert self.df.query('len==1').shape == df_singletons.shape, \
            'Singletons were not properly processed with _component_label_prunner()'
        shape_after = self.df.shape
        print(f'Initial shape: {shape_before}. Pruned shape: {shape_after}')
        # self.df.to_csv('df_after_label_intersection.csv')

    def _component_label_prunner(self):
        print(f'Run _component_label_prunner with {self.label_pruner_column}..')
        labels = pd.read_pickle('labels.pkl').iloc[:, 0].to_list()
        shape_before = self.df.shape
        df_singletons = self.df[self.df['len'] == 1]
        df_other = self.df[self.df['len'] > 1]
        for id in df_other['component_id'].unique():
            good_labels = list(eval(df_other[df_other['component_id'] == id][self.label_pruner_column].iloc[0]).keys())
            df_other.loc[df_other['component_id'] == id, labels] = 0
            df_other.loc[df_other['component_id'] == id, good_labels] = 1
        self.df = pd.concat([df_other, df_singletons], ignore_index=True)
        assert self.df.query('len==1').shape == df_singletons.shape, \
            'Singletons were not properly processed with _component_label_prunner()'
        shape_after = self.df.shape
        print(f'Initial shape: {shape_before}. Pruned shape: {shape_after}')

    def _component_image_prunner(self):
        print(f'Run _component_image_prunner with {self.image_prunner_column}..')
        shape_before = self.df.shape
        self.df['is_good_image'] = self.df.apply(
            lambda x: 1 if (
                    len(x.image) > 1 and len(eval(x[self.image_prunner_column])) > 0 and
                    set(eval(x.attributes)).issubset(set(eval(x[self.image_prunner_column]).keys()))
            ) else 0, axis=1)
        self.df.loc[self.df.query('len==1').index.tolist(), 'is_good_image'] = True
        assert self.df.query('len==1').is_good_image.sum() == len(self.df.query('len==1')), \
            'Singletons were not properly processed with _component_image_filter()'
        self.df = self.df.query('is_good_image == 1')
        shape_after = self.df.shape
        print(f'Initial shape: {shape_before}. Pruned shape: {shape_after}')

    def _component_is_bad_prunner(self):
        print(f'Run _component_is_bad_prunner with {self.condition}..')
        shape_before = self.df.shape
        self.df = self.df.query(self.condition)
        shape_after = self.df.shape
        print(f'Initial shape: {shape_before}. Pruned shape: {shape_after}')

    # def _bad_age_detector(self):
    #     print('Run _bad_age_detector..')
    #     self.df['bad_age'] = self.df \
    #                              .apply(lambda x: sum(x[]),
    #                                     axis=1
    #                                     ) != 1
    #     print(f"{self.df.query('bad_age==True').shape[0]} rows with bad age attributes were deleted!")
    #     self.df = self.df.query('bad_age==False')

    def _dataframe_transformer(self):
        """
        Create new columns according to columns_1 and columns_2 names
        """
        print('Run _dataframe_transformer..')
        self.df = self.df.assign(
            image=lambda x: x.image_path.apply(lambda x: os.path.join(self.dataset_path, os.path.basename(x)))
        )
        for label in self.labels:
            self.df[label] = self.df \
                            .apply(lambda x: x[self.label_column_dict[label]].values, axis=1)
        agg_dict = {label: lambda x: x.tolist() for label in self.labels}
        agg_dict['image'] = 'unique'
        agg_dict['set'] = lambda x: x.unique()[0]
        self.df = self.df.groupby(self.idx).agg(agg_dict)

    def _data_set_splitter(self):
        """
        Splits data by 'train' and 'val' subsets.
        Adds new column 'set' to self.dataframe
        """
        print('Run _data_set_splitter..')
        pd.core.common.random_state(42)
        np.random.seed(42)
        self.df['len'] = self.df.groupby('component_id').agg({'image': 'count'})
        self.df['set'] = self.df.groupby(self.idx).image_path \
            .transform(lambda x: np.random.choice(['train', 'val', 'test'],
                                                  replace=True,
                                                  size=len(x),
                                                  p=[self.train, self.val, self.test]
                                                  )[0])
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 1000)

    def _data_transformer(self):
        # self._bad_age_detector()
        self._data_set_splitter()
        if self.condition:
            self._component_is_bad_prunner()
        if self.image_prunner_column:
            self._component_image_prunner()
        if self.label_pruner_column:
            self._component_label_prunner()
        if self.labels_intersection == 1:
            self._component_label_intersector()
        self._dataframe_transformer()
