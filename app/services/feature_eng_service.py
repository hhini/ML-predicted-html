import pandas as pd
import numpy as np
# 必须先导入 experimental 才能导入 IterativeImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, KBinsDiscretizer

class FeatureEngService:
    """
    特征工程服务层：负责缺失值处理与特征变换
    """

    # ==========================
    # 1. 缺失值处理 (Imputation)
    # ==========================
    @staticmethod
    def drop_missing(df, threshold=0.5, axis=1):
        """
        删除法
        :param axis: 1=删除列, 0=删除行
        :param threshold: 缺失比例超过此值则删除 (仅对列有效)
        """
        df_clean = df.copy()
        if axis == 1:
            # 计算每列缺失比例
            miss_ratio = df_clean.isnull().mean()
            cols_to_drop = miss_ratio[miss_ratio > threshold].index.tolist()
            df_clean = df_clean.drop(columns=cols_to_drop)
            return df_clean, cols_to_drop
        else:
            # 删除含有缺失值的行 (慎用)
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna()
            dropped_count = initial_rows - len(df_clean)
            return df_clean, dropped_count

    @staticmethod
    def simple_impute(df, cols, method='median'):
        """
        统计学填充 (Simple Imputation)
        :param method: 'mean', 'median', 'most_frequent' (mode), 'constant' (Unknown)
        """
        df_new = df.copy()
        # 区分数值和分类
        if method in ['mean', 'median']:
            imputer = SimpleImputer(strategy=method)
        elif method == 'unknown':
            imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        else:
            imputer = SimpleImputer(strategy='most_frequent')
            
        df_new[cols] = imputer.fit_transform(df_new[cols])
        return df_new

    @staticmethod
    def advanced_impute(df, cols, method='knn', n_neighbors=5):
        """
        模型化填充 (Advanced Imputation) - 学术级方案
        :param method: 'knn' or 'mice' (IterativeImputer)
        """
        df_new = df.copy()
        
        # 注意：KNN 和 MICE 只能处理数值型数据。如果包含分类变量，需要先编码。
        # 这里为了简化，假设传入的 cols 都是数值型。
        
        if method == 'knn':
            # KNN 填充：寻找最近的 K 个样本
            imputer = KNNImputer(n_neighbors=n_neighbors)
        else:
            # MICE (链式方程多重插补)：学术界金标准
            # 它是通过回归模型预测缺失值，利用了其他特征的信息
            imputer = IterativeImputer(max_iter=10, random_state=0)
            
        df_new[cols] = imputer.fit_transform(df_new[cols])
        return df_new

    # ==========================
    # 2. 特征变换 (Transformation)
    # ==========================
    @staticmethod
    def scale_features(df, cols, method='standard'):
        """
        特征缩放
        :param method: 'standard' (Z-Score), 'minmax' (0-1)
        """
        df_new = df.copy()
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
            
        df_new[cols] = scaler.fit_transform(df_new[cols])
        return df_new

    @staticmethod
    def gaussian_transform(df, cols, method='log'):
        """
        非线性变换 (使其符合正态分布)
        :param method: 'log', 'box-cox', 'yeo-johnson'
        """
        df_new = df.copy()
        
        if method == 'log':
            # log1p = log(x + 1) 避免 x=0 时报错
            # 仅适用于非负数
            for col in cols:
                if (df_new[col] < 0).any():
                    continue # 跳过含负数的列
                df_new[col] = np.log1p(df_new[col])
                
        elif method in ['box-cox', 'yeo-johnson']:
            # Yeo-Johnson 支持负数，Box-Cox 仅支持正数
            # PowerTransformer 会自动标准化 (均值0方差1)
            pt = PowerTransformer(method=method, standardize=True)
            df_new[cols] = pt.fit_transform(df_new[cols])
            
        return df_new

    @staticmethod
    def discretize_features(df, cols, n_bins=5, strategy='quantile'):
        """
        离散化 (Binning)
        :param strategy: 'uniform' (等宽), 'quantile' (等频), 'kmeans' (聚类)
        """
        df_new = df.copy()
        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        
        # 结果会变成 0.0, 1.0, 2.0...
        df_new[cols] = est.fit_transform(df_new[cols])
        return df_new