import streamlit as st
from services.data_processor import DataProcessor

def render_upload_view():
    """
    渲染数据上传页面的 UI
    """
    st.header("📂 数据接入中心")
    st.markdown("支持上传 **CSV** 或 **Excel** 文件，系统将自动进行解析。")

    # 1. 文件上传组件
    uploaded_file = st.file_uploader(
        "拖拽文件到此处或点击上传", 
        type=["csv", "xlsx", "xls"]
    )

    # 2. 如果用户上传了文件
    if uploaded_file is not None:
        # 调用 Service 层读取数据
        df = DataProcessor.load_data(uploaded_file)
        
        if df is not None:
            st.success("✅ 数据读取成功！")
            
            # --- 关键点：保存到 Session State ---
            # Streamlit 每次点击按钮都会从头运行脚本。
            # 为了记住“我已经上传过文件了”，我们需要把它存到 session_state 里。
            st.session_state['df'] = df
            st.session_state['original_df'] = df.copy() # 保存原始副本用于对比
            st.session_state['change_log'] = [] # 初始化操作日志
            st.session_state['file_name'] = uploaded_file.name
            
            # 3. 展示数据概览
            summary = DataProcessor.get_data_summary(df)
            
            # 使用列布局 (Columns) 让显示更紧凑
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("样本量 (Rows)", summary['rows'])
            col2.metric("特征数 (Columns)", summary['cols'])
            col3.metric("缺失值总数", summary['missing_values'])
            col4.metric("内存占用", summary['memory'])
            
            # 4. 展示前5行数据
            st.subheader("数据预览 (Top 5)")
            st.dataframe(df.head())
            
        else:
            st.error("❌ 文件读取失败，请检查文件格式是否正确。")
    
    # 5. 如果之前已经上传过数据，即使用户没在操作上传框，也要显示出来
    elif 'df' in st.session_state:
        st.info(f"当前正在使用文件: **{st.session_state.get('file_name', 'Unknown')}**")
        st.dataframe(st.session_state['df'].head())