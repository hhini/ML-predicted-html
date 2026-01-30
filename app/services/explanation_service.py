import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

class ExplanationService:
    """
    XAI 解释性服务与模型持久化
    """

    @staticmethod
    def get_shap_values(model, X_train, X_target, model_name, task_type='classification'):
        """
        计算 SHAP 值
        :param model: 训练好的模型对象
        :param X_train: 训练集背景数据 (用于 KernelExplainer)
        :param X_target: 需要解释的目标数据 (DataFrame)
        :return: explainer, shap_values
        """
        # 1. 识别模型类型，选择最优 Explainer
        model_type = 'tree'
        if any(x in model_name for x in ['Linear', 'Logistic', 'SVM', 'KNN', 'MLP']):
            model_type = 'kernel'
        
        # 针对 Pipeline 的处理 (如果模型被 StandardScaler 包裹)
        estimator = model
        if hasattr(model, 'steps'): 
            estimator = model.steps[-1][1] # 取出最后的模型
            # 注意：如果是 Pipeline，传入的 X_train 应该是转换后的，这里为了简化，
            # 假设 SHAP 主要用于 Tree 模型 (XGB/RF)，它们通常不需要 Pipeline。
            # 对于 SVM/LR，使用 KernelExplainer 计算会比较慢。

        try:
            if model_type == 'tree':
                # 树模型专用 (快)
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X_target)
            else:
                # 通用模型 (慢)，使用 K-Means 聚类减少背景数据量
                # 采样 50 个样本作为背景，否则算不动
                background = shap.kmeans(X_train, 50)
                explainer = shap.KernelExplainer(estimator.predict, background)
                shap_values = explainer.shap_values(X_target)

            # 兼容性处理：SHAP 对二分类有时候返回 list [values_class0, values_class1]
            # 或者返回 3D array (samples, features, classes)
            if task_type == 'classification':
                if isinstance(shap_values, list):
                    # list 形式：取 class 1
                    shap_values = shap_values[1]
                    if isinstance(explainer.expected_value, list) or (isinstance(explainer.expected_value, np.ndarray) and len(explainer.expected_value) > 1):
                        explainer.expected_value = explainer.expected_value[1]
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    # 3D array 形式：(samples, features, classes)，取 class 1
                    # 通常最后一维是 classes
                    if shap_values.shape[2] == 2:
                        shap_values = shap_values[:, :, 1]
                        if isinstance(explainer.expected_value, list) or (isinstance(explainer.expected_value, np.ndarray) and len(explainer.expected_value) > 1):
                             explainer.expected_value = explainer.expected_value[1]
                
            return explainer, shap_values

        except Exception as e:
            print(f"SHAP 计算失败: {e}")
            return None, None

    @staticmethod
    def save_model(model, features, filename="best_model.pkl"):
        """
        保存模型与特征列表
        """
        payload = {
            "model": model,
            "features": features
        }
        joblib.dump(payload, filename)
        return filename