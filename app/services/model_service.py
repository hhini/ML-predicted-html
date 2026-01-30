import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, KFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# 关闭 Optuna 的啰嗦日志
optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModelService:
    """
    AutoML 核心服务：贝叶斯调参 + 交叉验证竞技场
    """

    @staticmethod
    def get_model_factory(task_type, model_name, params={}):
        """
        工厂模式：根据名字和参数生成模型对象
        对于距离敏感模型(SVM, KNN, MLP, LR)，自动套用 StandardScaler
        """
        base_model = None
        
        # 动态实例化，避免一次性初始化所有模型导致参数不兼容报错
        if task_type == 'classification':
            if model_name == "Logistic Regression":
                base_model = LogisticRegression(**params, max_iter=1000)
            elif model_name == "Random Forest":
                base_model = RandomForestClassifier(**params, n_jobs=-1)
            elif model_name == "XGBoost":
                base_model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
            elif model_name == "LightGBM":
                base_model = LGBMClassifier(**params, n_jobs=-1, verbose=-1)
            elif model_name == "SVM":
                base_model = SVC(**params, probability=True)
            elif model_name == "KNN":
                base_model = KNeighborsClassifier(**params, n_jobs=-1)
            elif model_name == "MLP (Neural Net)":
                base_model = MLPClassifier(**params, max_iter=1000)
        else:
            if model_name == "Linear Regression":
                base_model = LinearRegression(**params)
            elif model_name == "Random Forest":
                base_model = RandomForestRegressor(**params, n_jobs=-1)
            elif model_name == "XGBoost":
                base_model = XGBRegressor(**params, n_jobs=-1)
            elif model_name == "LightGBM":
                base_model = LGBMRegressor(**params, n_jobs=-1, verbose=-1)
            elif model_name == "SVM":
                base_model = SVR(**params)
            elif model_name == "KNN":
                base_model = KNeighborsRegressor(**params, n_jobs=-1)
            elif model_name == "MLP (Neural Net)":
                base_model = MLPRegressor(**params, max_iter=1000)
        
        if base_model is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        # 管道处理：除了树模型，其他都需要标准化
        tree_models = ["Random Forest", "XGBoost", "LightGBM"]
        if model_name not in tree_models:
            return make_pipeline(StandardScaler(), base_model)
        else:
            return base_model

    @staticmethod
    def optimize_hyperparameters(X, y, task_type, model_name, n_trials=10):
        """
        贝叶斯优化 (Optuna)
        """
        def objective(trial):
            # 1. 定义超参数搜索空间 (Search Space)
            params = {}
            if model_name == "Random Forest":
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
                params['max_depth'] = trial.suggest_int('max_depth', 3, 20)
            elif model_name == "XGBoost" or model_name == "LightGBM":
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
                params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            elif model_name == "SVM":
                params['C'] = trial.suggest_float('C', 0.1, 10, log=True)
                if task_type == 'classification': params['kernel'] = 'rbf' # 简化
            elif model_name == "KNN":
                params['n_neighbors'] = trial.suggest_int('n_neighbors', 3, 15)
            elif model_name == "Logistic Regression":
                params['C'] = trial.suggest_float('C', 0.1, 10, log=True)
            # ... 其他模型暂用默认或简化
            
            # 2. 交叉验证评估
            model = ModelService.get_model_factory(task_type, model_name, params)
            cv = StratifiedKFold(n_splits=3) if task_type == 'classification' else KFold(n_splits=3)
            
            scores = []
            # 简单的 CV 用于调参 (速度优先)
            from sklearn.model_selection import cross_val_score
            # accuracy 或 r2
            scoring = 'accuracy' if task_type == 'classification' else 'r2'
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    @staticmethod
    def train_and_evaluate(X, y, task_type, model_list, n_splits=5, use_optuna=False, n_trials=5):
        """
        主函数：在训练集上进行 K-Fold 交叉验证，并收集所有绘图所需数据
        """
        results = {}
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if task_type == 'classification' else KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for name in model_list:
            # 1. 获取最佳参数
            best_params = {}
            if use_optuna:
                best_params = ModelService.optimize_hyperparameters(X, y, task_type, name, n_trials)
            
            model = ModelService.get_model_factory(task_type, name, best_params)
            
            # 2. 运行 K-Fold 并收集详细数据
            fold_metrics = []
            y_tests_all = [] # 真实标签
            y_probs_all = [] # 预测概率 (用于 ROC/校准)
            y_preds_all = [] # 预测类别 (用于混淆矩阵)
            
            # 由于 Pipeline 无法直接用于 cross_validate 的 return_estimator，我们手动循环
            # 这样可以手动处理 Scaling
            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # 训练
                model.fit(X_train_fold, y_train_fold)
                
                # 预测
                y_pred = model.predict(X_val_fold)
                if task_type == 'classification':
                    # 注意：有些模型(SVM)需要 probability=True
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_val_fold)[:, 1]
                    else:
                        y_prob = y_pred # Fallback
                        
                    # 计算单折指标
                    fold_metrics.append({
                        "Accuracy": accuracy_score(y_val_fold, y_pred),
                        "Precision": precision_score(y_val_fold, y_pred, zero_division=0),
                        "Recall": recall_score(y_val_fold, y_pred, zero_division=0),
                        "F1": f1_score(y_val_fold, y_pred, zero_division=0),
                        "AUC": 0 if not hasattr(model, "predict_proba") else 
                               roc_auc_score(y_val_fold, y_prob)
                    })
                    y_tests_all.extend(y_val_fold)
                    y_probs_all.extend(y_prob)
                    y_preds_all.extend(y_pred)
                else:
                    # 回归指标
                    fold_metrics.append({
                        "R2": r2_score(y_val_fold, y_pred),
                        "MAE": mean_absolute_error(y_val_fold, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_val_fold, y_pred))
                    })
                    y_tests_all.extend(y_val_fold)
                    y_probs_all.extend(y_pred) # 回归中 prob 就是 pred

            # 3. 汇总结果
            metrics_df = pd.DataFrame(fold_metrics)
            mean_metrics = metrics_df.mean().to_dict()
            
            # 分类任务还需要计算整体的 ROC 数据
            roc_data = {}
            if task_type == 'classification':
                fpr, tpr, _ = roc_curve(y_tests_all, y_probs_all)
                roc_auc = auc(fpr, tpr)
                mean_metrics['AUC'] = roc_auc # 更新为整体 AUC
                roc_data = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
            
            # 4. 全量训练 (用于导出模型和特征重要性分析)
            # 注意：必须重新实例化一个新模型，或者 clone 之前的 model，这里直接复用工厂
            final_model = ModelService.get_model_factory(task_type, name, best_params)
            final_model.fit(X, y)
            
            # 提取特征重要性
            feature_importance = None
            # 如果是 Pipeline，需要取出步骤
            estimator = final_model
            if isinstance(final_model, Pipeline):
                estimator = final_model.steps[-1][1]
            
            if hasattr(estimator, 'feature_importances_'):
                feature_importance = estimator.feature_importances_
            elif hasattr(estimator, 'coef_'):
                feature_importance = np.abs(estimator.coef_[0]) if estimator.coef_.ndim > 1 else np.abs(estimator.coef_)

            results[name] = {
                "metrics": mean_metrics,      # 平均指标
                "fold_metrics": metrics_df,   # 每一折的指标 (用于统计检验)
                "y_true": y_tests_all,        # 拼接后的真实值
                "y_pred": y_probs_all,        # 拼接后的预测值 (概率)
                "y_pred_class": y_preds_all,  # 拼接后的预测类别
                "roc_data": roc_data,         # ROC 绘图数据
                "best_params": best_params,   # 调参结果
                "final_model": final_model,   # 全量训练后的模型对象
                "feature_importance": feature_importance # 特征重要性
            }
            
        return results

    @staticmethod
    def compute_learning_curve(model, X, y, cv=5):
        """
        计算学习曲线数据
        """
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring='accuracy' # 简化，默认用 accuracy 或 r2
        )
        return train_sizes, train_scores, test_scores
        