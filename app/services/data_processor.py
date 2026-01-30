import pandas as pd
import io

class DataProcessor:
    """
    数据处理核心服务类
    负责数据的读取、清洗、转换，不包含任何界面显示代码
    """
    
    @staticmethod
    def load_data(uploaded_file):
        """
        读取上传的文件并转换为 DataFrame
        
        :param uploaded_file: Streamlit 的上传文件对象
        :return: DataFrame (如果成功), None (如果失败)
        """
        if uploaded_file is None:
            return None
            
        try:
            # 获取文件名后缀
            filename = uploaded_file.name
            
            # 根据后缀判断读取方式
            if filename.endswith('.csv'):
                # encoding='utf-8' 是通用标准，如果是中文乱码可能需要 'gbk'
                df = pd.read_csv(uploaded_file)
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(uploaded_file)
            else:
                # 遇到不支持的格式
                return None
                
            return df
            
        except Exception as e:
            # 实际开发中这里应该记录日志
            print(f"读取文件出错: {e}")
            return None

    @staticmethod
    def get_data_summary(df):
        """
        获取数据的基本信息 (给分析师看的概览)
        """
        if df is None:
            return None
            
        summary = {
            "rows": df.shape[0],      # 行数
            "cols": df.shape[1],      # 列数
            "missing_values": df.isnull().sum().sum(), # 总缺失值
            "memory": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB", # 内存占用
            "columns": list(df.columns) # 列名列表
        }
        return summary