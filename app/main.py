import streamlit as st
import os
import platform
import matplotlib.pyplot as plt

# --- 全局配置: 解决 Matplotlib 中文显示问题 ---
try:
    system_name = platform.system()
    if system_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    elif system_name == "Darwin": # Mac
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else: # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Font config error: {e}")

# --- UI 美化函数 ---
def load_assets():
    # 获取当前脚本的绝对路径目录 (app/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (app/ 的上一级)
    project_root = os.path.dirname(current_dir)
    
    # 1. 加载 CSS
    css_path = os.path.join(project_root, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # 2. 侧边栏 Logo (如果有)
    logo_path = os.path.join(project_root, "assets", "logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_container_width=True)

# --- 新增 import ---
from ui.upload_view import render_upload_view 
from ui.eda_view import render_eda_view
from ui.feature_eng_view import render_feature_eng_view
from ui.feature_selection_view import render_feature_selection_view
from ui.model_view import render_model_view
# 1. 设置页面基本配置 (浏览器标签页标题等)
st.set_page_config(
    page_title="AI Insight Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载自定义样式和素材
load_assets()

# 2. 页面标题
st.title("🏥 AI Insight Studio - 智能预测平台")

# 3. 写一段欢迎语
st.markdown("""
欢迎使用企业级机器学习预测平台。
目前支持领域：**医学、金融、商业**
""")

# 4. 做一个简单的侧边栏 (Sidebar)
with st.sidebar:
    st.header("功能导航")
    mode = st.radio(
        "选择模式:",
        ("数据上传", "数据分析 (EDA)", "特征工程 (清洗/变换)", "特征选择", "模型训练", "预测结果")
    )

# ... (之前的代码) ...



#--- 新增 import ---
from ui.prediction_view import render_prediction_view

# 5. 根据侧边栏的选择显示不同内容
if mode == "数据上传":
    render_upload_view()  # <--- 这里调用我们写好的模块
    
elif mode == "数据分析 (EDA)":
    render_eda_view()
    
elif mode == "特征工程 (清洗/变换)":
    render_feature_eng_view()
    
elif mode == "特征选择":
    render_feature_selection_view()
    
elif mode == "模型训练":
    render_model_view()

    # 6. 预测结果页面
elif mode == "预测结果":
    render_prediction_view()