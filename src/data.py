import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from collections import namedtuple
import argparse

class CreateDataset(Dataset):
    def __init__(self, xray_data, optical_data, target, columns_xray=None, columns_optical=None):
        self.xray_data = np.array(xray_data)
        self.optical_data = np.array(optical_data)
        self.target = np.array(target)
        self.columns_xray = columns_xray
        self.columns_optical = columns_optical

    @property
    def xray_features_low(self):
        return self.xray_data.min(axis=0)

    @property
    def xray_features_high(self):
        return self.xray_data.max(axis=0)

    @property
    def optical_features_low(self):
        return self.optical_data.min(axis=0)

    @property
    def optical_features_high(self):
        return self.optical_data.max(axis=0)

    @property
    def shape(self):
        return self.xray_data.shape, self.optical_data.shape

    def __getitem__(self, index):
        xray = torch.tensor(self.xray_data[index], dtype=torch.float)
        optical = torch.tensor(self.optical_data[index], dtype=torch.float)
        return xray, optical

    def __len__(self):
        return len(self.xray_data)  # Assuming xray_data and optical_data are of the same length

    def to_dataframe(self):
        xray_df = pd.DataFrame(self.xray_data, columns=self.columns_xray)
        optical_df = pd.DataFrame(self.optical_data, columns=self.columns_optical)
        df = pd.concat([xray_df, optical_df], axis=1)
        return df
    
def transform_features(df, feature_names, skip_log_features):
    transformed_features = [
        np.log10(1 + df[feature].values) if feature not in skip_log_features else df[feature].values
        for feature in feature_names
    ]
    return np.array(transformed_features).T

def preprocess_one(df_pos, feature_names, skip_log_features):
    # Transform features
    X_pos = transform_features(df_pos, feature_names, skip_log_features)

    X = X_pos
    Y = np.ones(len(X_pos))

    return X, Y

def handle_missing_values(X_train, X_test):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    X_train = imp_mean.fit_transform(X_train)
    X_test = imp_mean.transform(X_test)
    return X_train, X_test, imp_mean

Data = namedtuple(
    "Data",
    [
        "X_train_cat1",
        "X_train_cat2",
        "X_ev_cat1",
        "X_ev_cat2",
        "y_train",
        "y_ev",
    ],
)

def prepare_data(config: argparse.Namespace):
    # read the best and the worst match for a list of x-ray sources
    def_first_prob_df = pd.read_csv('../data/most_probable_new.csv')
    def_last_prob_df = pd.read_csv('../data/last_new.csv')

    def_first_prob_df_filtered = def_first_prob_df.dropna(subset=["name"])
    # extract chandra_source_id values from def_first_prob_df
    chandra_ids_in_first = def_first_prob_df_filtered['chandra_source_id'].unique()

    # filter def_last_prob_df based on the extracted chandra_source_id values
    filtered_last_prob_df = def_last_prob_df[def_last_prob_df['chandra_source_id'].isin(chandra_ids_in_first)]
    
    df_pos = def_first_prob_df_filtered.query(config.FILTERING).copy()
    df_pos.dropna(subset=['name'], inplace=True)
    chandra_ids_in_pos = df_pos['chandra_source_id'].unique()

    df_neg= filtered_last_prob_df[filtered_last_prob_df['chandra_source_id'].isin(chandra_ids_in_pos)].copy()

    df_pos.replace({'flux_aper_b': 0}, np.nan, inplace=True)
    df_neg.replace({'flux_aper_b': 0}, np.nan, inplace=True)
    #df_pos['gmag_logflux'] = df_pos['phot_g_mean_mag'] + np.log10(df_pos['flux_aper_b']/1e-13)*2.5
    #df_neg['gmag_logflux'] = df_neg['phot_g_mean_mag'] + np.log10(df_neg['flux_aper_b']/1e-13)*2.5

    feature_names_xray = [
        'hard_hs',
        'hard_hm',
        'hard_ms',
        'var_intra_prob_b',
        'var_inter_prob_b'
    ]

    feature_names_optical = [
        'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'bp_rp', 'bp_g', 'g_rp', 'parallax', 'parallax_over_error'
    ]

    # List of features to skip for the log transformation
    skip_log_features = [
        'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'bp_rp', 'bp_g', 'g_rp',
        'hard_hs',
        'hard_hm',
        'hard_ms',
        'var_intra_prob_b',
        'var_inter_prob_b'
    ]

    # Assuming df_pos contains both x-ray and optical data
    X_xray, Y = preprocess_one(df_pos, feature_names_xray, skip_log_features)
    X_optical, _ = preprocess_one(df_pos, feature_names_optical, skip_log_features)

    indices = np.arange(X_xray.shape[0])

    # Split indices
    indices_train, indices_test = train_test_split(indices, test_size=0.3, shuffle=True)

    # Use the same indices to split X-ray and Optical data
    X_train_xray, X_ev_xray = X_xray[indices_train], X_xray[indices_test]
    X_train_optical, X_ev_optical = X_optical[indices_train], X_optical[indices_test]
    Y_train, Y_ev = Y[indices_train], Y[indices_test]

    X_train_xray, X_ev_xray, imp_mean_xray = handle_missing_values(X_train_xray, X_ev_xray)
    X_train_optical, X_ev_optical, imp_mean_optical = handle_missing_values(X_train_optical, X_ev_optical)

    # torch everything
    X_train_xray, X_ev_xray = torch.tensor(X_train_xray), torch.tensor(X_ev_xray)
    X_train_optical, X_ev_optical = torch.tensor(X_train_optical), torch.tensor(X_ev_optical)
    Y_train, Y_ev = torch.tensor(Y_train), torch.tensor(Y_ev)

    data =  Data(
        X_train_xray.to(config.DEV),
        X_train_optical.to(config.DEV),
        X_ev_xray.to(config.DEV),
        X_ev_optical.to(config.DEV),
        Y_train.to(config.DEV),
        Y_ev.to(config.DEV),
    )

    return data