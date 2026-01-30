import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression, RFE
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

class FeatureSelectionService:
    
    # ==========================================
    # 1. 数据划分 (Data Splitting)
    # ==========================================
    @staticmethod
    def split_data(df, target_col, test_size=0.2, task_type='classification'):
        """
        严谨的数据划分：分层抽样 (Stratified)
        """
        # 准备 X 和 y
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 简单处理：如果 X 中有非数值列，需要先编码才能 split 吗？
        # 通常 split 不关心内容，但为了后续算法方便，建议在这里做一次简单的 LabelEncoder
        # 但严谨的做法是：Split 后，在 Train 上 fit Encoder，Transform Test。
        # 为了简化 Demo，我们假设之前的步骤已经处理好了数值化，或者在这里临时处理。
        
        stratify = y if task_type == 'classification' else None
        
        # 分层抽样
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=stratify
        )
        
        return X_train, X_test, y_train, y_test

    @staticmethod
    def _prepare_data_for_selection(X):
        """
        内部辅助函数：自动处理非数值列 (Encoding)
        防止直接 drop 掉分类变量
        """
        X_encoded = X.copy()
        # 简单处理：对所有 object/category 列进行 LabelEncoding
        # 注意：这只是为了特征筛选的临时编码，不会改变原始数据
        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            # 强制转为 string 以避免混合类型报错
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            
        # 简单的缺失值填充 (填 0)，防止模型报错
        X_encoded = X_encoded.fillna(0)
        return X_encoded

    # ==========================================
    # 2. 过滤法 (Filter Methods)
    # ==========================================
    @staticmethod
    def filter_selection(X_train, y_train, task_type, method='auto', k=10):
        """
        根据任务类型自动选择统计检验方法
        """
        # 1. 预处理：编码 + 填补
        X_clean = FeatureSelectionService._prepare_data_for_selection(X_train)
        
        # 确保 y 也是干净的
        y_clean = y_train.fillna(y_train.mode()[0]) if task_type == 'classification' else y_train.fillna(y_train.mean())

        # 定义评分函数
        score_func = None
        
        if task_type == 'classification':
            if method == 'chi2': # 卡方 (要求非负)
                # 归一化到 0-1
                scaler = MinMaxScaler()
                X_clean = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns)
                score_func = chi2
            elif method == 'anova': # 方差分析
                score_func = f_classif
            elif method == 'mutual_info': # 互信息
                score_func = mutual_info_classif
            else: # auto
                score_func = f_classif
                
        else: # regression
            if method == 'pearson':
                score_func = f_regression
            elif method == 'mutual_info':
                score_func = mutual_info_regression
            else:
                score_func = f_regression

        # 运行筛选
        selector = SelectKBest(score_func=score_func, k='all') # 先算所有分
        selector.fit(X_clean, y_clean)
        
        results = pd.DataFrame({
            'Feature': X_clean.columns,
            'Score': selector.scores_,
            'P-Value': selector.pvalues_ if hasattr(selector, 'pvalues_') else [None]*len(X_clean.columns)
        }).sort_values('Score', ascending=False)
        
        return results

    # ==========================================
    # 3. 嵌入法 (Embedded Methods)
    # ==========================================
    @staticmethod
    def embedded_selection(X_train, y_train, task_type, method='xgboost', use_permutation=False):
        """
        Lasso, RF, XGBoost + Permutation Importance
        """
        # 1. 预处理
        X_clean = FeatureSelectionService._prepare_data_for_selection(X_train)
        y_clean = y_train.fillna(y_train.mode()[0]) if task_type == 'classification' else y_train.fillna(y_train.mean())

        # 2. 选择模型
        model = None
        if task_type == 'classification':
            if method == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif method == 'xgboost':
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            elif method == 'lasso':
                # Lasso 分类通常用 LogisticRegression + L1 penalty
                model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        else:
            if method == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif method == 'xgboost':
                model = XGBRegressor(random_state=42)
            elif method == 'lasso':
                model = LassoCV(random_state=42)
        
        # 3. 训练模型
        model.fit(X_clean, y_clean)
        
        # 4. 获取重要性
        importances = []
        
        if use_permutation:
            # 这里的 scoring 需要根据 task_type 调整，为了通用简单起见，不指定 scoring 让它自己选
            perm_importance = permutation_importance(model, X_clean, y_clean, n_repeats=5, random_state=42)
            importances = perm_importance.importances_mean
        else:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'): # Lasso / LR
                importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                importances = np.zeros(X_clean.shape[1])
                
        results = pd.DataFrame({
            'Feature': X_clean.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return results

    # ==========================================
    # 4. 包装法 (Wrapper Methods)
    # ==========================================
    @staticmethod
    def wrapper_selection(X_train, y_train, task_type, n_features_to_select=10):
        """
        RFE (递归特征消除)
        """
        # 1. 预处理
        X_clean = FeatureSelectionService._prepare_data_for_selection(X_train)
        y_clean = y_train.fillna(y_train.mode()[0]) if task_type == 'classification' else y_train.fillna(y_train.mean())
        
        # 2. 基模型 (通常用简单的 Tree 或 Linear)
        if task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            
        # 3. RFE
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
        selector.fit(X_clean, y_clean)
        
        results = pd.DataFrame({
            'Feature': X_clean.columns,
            'Selected': selector.support_,
            'Ranking': selector.ranking_
        }).sort_values('Ranking') # Ranking 1 means selected
        
        return results