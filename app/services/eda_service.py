import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, chi2


class EdaService:
    """
    EDA 深度分析服务层
    提供基础统计、分布形态、异常值检测、相关性分析
    """

    @staticmethod
    def get_column_types(df):
        """
        初步自动推断列类型 (Numeric / Categorical)
        """
        col_types = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # 如果是数字，但唯一值非常少（比如 < 5% 且绝对数量 < 20），可能是分类变量（如 Gender: 0,1）
                if df[col].nunique() < 20 and df[col].nunique() / len(df) < 0.05:
                    col_types[col] = 'Categorical'
                else:
                    col_types[col] = 'Numeric'
            else:
                col_types[col] = 'Categorical'
        return col_types

    @staticmethod
    def get_detailed_stats(df, col_name, dtype):
        """
        计算单个特征的深度统计指标
        """
        series = df[col_name]
        stats = {}

        # 1. 通用指标
        stats['Missing Values'] = series.isnull().sum()
        stats['Missing Ratio'] = f"{series.isnull().mean() * 100:.2f}%"
        stats['Unique Values (Cardinality)'] = series.nunique()
        
        # 2. 数值型特有指标
        if dtype == 'Numeric':
            stats['Mean'] = round(series.mean(), 2)
            stats['Std Dev'] = round(series.std(), 2)
            stats['Min'] = round(series.min(), 2)
            stats['Median (50%)'] = round(series.median(), 2)
            stats['Max'] = round(series.max(), 2)
            
            # 分布形态
            stats['Skewness (偏度)'] = round(series.skew(), 2)
            stats['Kurtosis (峰度)'] = round(series.kurt(), 2)
            
            # IQR 异常值检测
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            stats['Outliers Count (IQR Method)'] = len(outliers)
            stats['Outlier Ratio'] = f"{len(outliers)/len(series)*100:.2f}%"

        # 3. 分类型特有指标
        else:
            stats['Mode (众数)'] = series.mode()[0] if not series.mode().empty else "N/A"
            top_val = series.value_counts().idxmax()
            top_freq = series.value_counts().max()
            stats['Top Class'] = f"{top_val} (Freq: {top_freq})"
        
        return stats

    @staticmethod
    def get_correlation_matrix(df, col_types):
        """
        计算数值型特征的相关系数矩阵 (Pearson)
        """
        # 只筛选 Numeric 类型的列
        numeric_cols = [col for col, dtype in col_types.items() if dtype == 'Numeric']
        if len(numeric_cols) < 2:
            return None
        return df[numeric_cols].corr(method='pearson')

    @staticmethod
    def get_missing_correlation(df):
        """
        计算缺失值相关性矩阵
        (分析：如果 A 缺失，B 是否也倾向于缺失？)
        """
        # 生成一个 True/False 的矩阵，代表是否缺失
        nullity = df.isnull()
        # 如果没有缺失值，或者全都是缺失值，相关性计算会报错或无意义
        if nullity.sum().sum() == 0:
            return None
        # 计算布尔值的相关性
        return nullity.corr()


    # ... (之前的 get_column_types, get_missing_correlation 等方法保留) ...

    # ==========================================
    # 1. 数据质量断裂点 (Data Integrity)
    # ==========================================
    @staticmethod
    def detect_special_values(df, special_vals=[0, -1, -999, 999]):
        """
        审计特殊值：检查每一列中是否包含可能是缺失值掩码的特殊数字
        """
        audit_results = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_res = {"Feature": col}
                for val in special_vals:
                    count = (df[col] == val).sum()
                    if count > 0:
                        col_res[f"Count ({val})"] = count
                        col_res[f"Ratio ({val})"] = f"{count/len(df):.2%}"
                
                # 只有当发现了特殊值才记录
                if len(col_res) > 1:
                    audit_results.append(col_res)
        return pd.DataFrame(audit_results)

    # ==========================================
    # 2. 分布动力学 (Distribution Dynamics)
    # ==========================================
    @staticmethod
    def normality_test(series):
        """
        正态性检验：Shapiro-Wilk
        :return: (statistic, p-value, is_normal)
        """
        # Shapiro 对大数据集太敏感，通常采样 5000 个点
        data = series.dropna()
        if len(data) > 5000:
            data = data.sample(5000, random_state=42)
        
        stat, p_value = stats.shapiro(data)
        # 常用 alpha = 0.05, p < 0.05 拒绝原假设（即不服从正态分布）
        return stat, p_value, p_value > 0.05

    @staticmethod
    def detect_outliers_isolation_forest(df, col_name, contamination=0.05):
        """
        使用孤立森林检测异常值 (适合多维或单维，这里演示单维)
        """
        data = df[[col_name]].dropna()
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(data)
        # -1 为异常值，1 为正常
        outliers = data[preds == -1]
        return outliers

    # ==========================================
    # 3. 特征间关系 (Relationships)
    # ==========================================
    @staticmethod
    def calculate_vif(df, numeric_cols):
        """
        计算 VIF (方差膨胀因子) 检查多重共线性
        """
        # 必须处理缺失值才能算 VIF，这里简单丢弃
        clean_df = df[numeric_cols].dropna()
        
        # 添加截距项 (const) 是 statsmodels 的要求，否则 VIF 计算不准
        clean_df_with_const = clean_df.copy()
        clean_df_with_const['const'] = 1
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_cols
        
        vif_vals = []
        for i in range(len(numeric_cols)):
            try:
                # 计算第 i 个特征的 VIF
                val = variance_inflation_factor(clean_df_with_const.values, i)
                vif_vals.append(round(val, 2))
            except:
                vif_vals.append(np.inf) # 发生除零错误通常意味着完全共线性
                
        vif_data["VIF"] = vif_vals
        return vif_data.sort_values(by="VIF", ascending=False)

    # ==========================================
    # 4. 目标与偏差 (Target & Bias)
    # ==========================================
    @staticmethod
    def analyze_target_relationship(df, target_col, col_types):
        """
        计算每个特征与目标变量的关联度
        分类目标 -> 卡方检验 / 互信息
        连续目标 -> 皮尔逊相关 / 互信息
        """
        results = []
        target_type = col_types[target_col]
        
        # 预处理：删除缺失值，简单编码
        data = df.copy().dropna()
        y = data[target_col]
        
        # 如果目标是分类，使用 LabelEncoder
        if target_type == 'Categorical' and y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y)

        for col in df.columns:
            if col == target_col: continue
            
            feat_type = col_types[col]
            score = 0
            method = ""
            
            try:
                # Case A: 目标是分类 (Classification)
                if target_type == 'Categorical':
                    if feat_type == 'Numeric':
                        # 连续转分类：ANOVA 或 互信息 (这里用互信息通用性强)
                        score = mutual_info_classif(data[[col]], y, random_state=42)[0]
                        method = "Mutual Info"
                    else:
                        # 分类转分类：卡方检验 (Chi-Square) 需要数字输入
                        # 这里简化处理，如果是 object 类型先编码
                        if data[col].dtype == 'object':
                             X_feat = pd.factorize(data[col])[0].reshape(-1, 1)
                        else:
                             X_feat = data[[col]]
                        
                        # 卡方检验返回 (chi2, p-value)，我们取 chi2
                        score = chi2(X_feat, y)[0] 
                        method = "Chi-Square"
                
                # Case B: 目标是连续 (Regression)
                else:
                    if feat_type == 'Numeric':
                        score = data[col].corr(pd.Series(y, index=data.index))
                        method = "Pearson Corr"
                    else:
                        # 分类转连续：ANOVA (略复杂，暂用互信息代替)
                        if data[col].dtype == 'object':
                             X_feat = pd.factorize(data[col])[0].reshape(-1, 1)
                        else:
                             X_feat = data[[col]]
                        score = mutual_info_regression(X_feat, y, random_state=42)[0]
                        method = "Mutual Info"

                results.append({
                    "Feature": col,
                    "Score": round(abs(score), 4), # 取绝对值
                    "Method": method
                })
            except Exception as e:
                results.append({"Feature": col, "Score": -1, "Method": "Error"})

        return pd.DataFrame(results).sort_values(by="Score", ascending=False)        