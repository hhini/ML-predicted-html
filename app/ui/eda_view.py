import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy import stats # 用于 Q-Q 图
from services.eda_service import EdaService
def render_eda_view():
    st.header("🔍 数据深度透视 (Deep EDA)")

    if 'df' not in st.session_state:
        st.warning("请先在【数据上传】模块上传数据！")
        return
    
    df = st.session_state['df']

    # 使用 Tabs 分离关注点，避免页面太长
    tabs = st.tabs([
        "📋 概览 & 质量审计", 
        "📈 分布与正态性", 
        "🔗 多重共线性 (VIF)", 
        "🎯 目标关联分析",
        "🧩 缺失值侦探"
    ])

    # ==========================
    # Tab 1: 全局概览与类型修正
    # ==========================
    with tabs[0]:
        st.subheader("1. 数据字典与类型定义")
        st.markdown("""
        系统已根据唯一值数量自动推断类型。**请务必检查并修正**，这决定了后续统计和建模的方式。
        - **Numeric**: 连续数值（如年龄、收入）。
        - **Categorical**: 类别标签（如性别、疾病分级、ID）。
        """)

        # 初始化或同步类型
        # 核心修复：检查缓存的 col_types 键是否与当前 df 的列完全一致
        current_cols = set(df.columns)
        cached_cols = set(st.session_state['col_types'].keys()) if 'col_types' in st.session_state else set()
        
        if 'col_types' not in st.session_state or current_cols != cached_cols:
            st.session_state['col_types'] = EdaService.get_column_types(df)

        # 构造配置表供用户编辑
        type_df = pd.DataFrame({
            "Feature Name": st.session_state['col_types'].keys(),
            "Detected Type": st.session_state['col_types'].values(),
            "Unique Values": [df[col].nunique() for col in df.columns],
            "Sample Values": [str(list(df[col].unique()[:5])) for col in df.columns]
        })

        # 可编辑表格
        edited_df = st.data_editor(
            type_df,
            column_config={
                "Detected Type": st.column_config.SelectboxColumn(
                    "Data Type (User Define)",
                    options=["Numeric", "Categorical"],
                    required=True,
                )
            },
            hide_index=True,
            use_container_width=True,
            height=400,
            key="type_editor_v2"
        )

        # 更新 Session State
        new_types = dict(zip(edited_df['Feature Name'], edited_df['Detected Type']))
        st.session_state['col_types'] = new_types
        st.success("✅ 类型定义已生效！请点击其他 Tab 查看详细分析。")




    # ==========================
    # Tab 1: 质量审计 (Quality Audit)
    # ==========================

        st.subheader("1. 数据完整性检查 (Data Integrity)")
        
        # ... (这里保留原来的类型定义代码) ...
        # 请把原来的类型定义代码放在这里
        
        st.divider()
        st.markdown("#### 🕵️ 特殊值/占位符审计")
        st.caption("很多真实数据会用 `0`, `-1`, `-999` 代表缺失。如果这些值占比很高，直接算均值会严重失真。")
        
        # 用户可自定义要审计的特殊值
        user_vals = st.text_input("输入要检测的特殊值 (用逗号分隔)", "0, -1, -999, 999")
        try:
            special_vals = [float(x.strip()) for x in user_vals.split(',')]
            audit_df = EdaService.detect_special_values(df, special_vals)
            
            if not audit_df.empty:
                st.warning(f"⚠️ 发现 {len(audit_df)} 个特征包含潜在的占位符！")
                st.dataframe(audit_df, use_container_width=True)
            else:
                st.success("未检测到指定的特殊值占位符。")
        except:
            st.error("输入的格式不正确，请输入数字，用逗号分隔。")

    # ==========================
    # Tab 2: 分布与正态性 (Distribution)
    # ==========================
    with tabs[1]:
        st.subheader("2. 分布动力学检测")
        col_sel = st.selectbox("选择数值特征进行检验:", 
                             [c for c, t in st.session_state['col_types'].items() if t == 'Numeric'])
        
        col_l, col_r = st.columns([1, 1])
        
        # 1. 正态性检验 (Normality Test)
        with col_l:
            st.markdown("##### 🧪 Shapiro-Wilk 正态性检验")
            stat, p_val, is_normal = EdaService.normality_test(df[col_sel])
            
            st.metric("P-Value", f"{p_val:.4f}")
            if p_val < 0.05:
                st.error(f"P < 0.05 拒绝原假设：数据 **不服从** 正态分布。\n建议：考虑 Log 变换或 Box-Cox 变换。")
            else:
                st.success("P > 0.05 无法拒绝原假设：数据 **近似服从** 正态分布。")
            
            # Q-Q Plot
            st.markdown("##### Q-Q 图 (Quantile-Quantile Plot)")
            # 绘制 Q-Q 图比较麻烦，这里用 scipy 计算点，用 plotly 画
            qq_data = df[col_sel].dropna()
            (osm, osr), (slope, intercept, r) = stats.probplot(qq_data, dist="norm")
            fig_qq = px.scatter(x=osm, y=osr, labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'})
            # 添加红线
            fig_qq.add_trace(go.Scatter(x=osm, y=slope*osm + intercept, mode='lines', name='Normal Line', line=dict(color='red')))
            st.plotly_chart(fig_qq, use_container_width=True)

        # 2. 高级异常值检测 (Outlier Detection)
        with col_r:
            st.markdown("##### 🚨 异常值判定")
            method = st.radio("选择检测方法:", ["IQR (箱线图法)", "Isolation Forest (孤立森林)"])
            
            if method == "IQR (箱线图法)":
                fig = px.box(df, y=col_sel, points="outliers", title="Boxplot (IQR)")
                st.plotly_chart(fig, use_container_width=True)
                # 简单计算
                Q1 = df[col_sel].quantile(0.25)
                Q3 = df[col_sel].quantile(0.75)
                IQR = Q3 - Q1
                n_outliers = ((df[col_sel] < (Q1 - 1.5 * IQR)) | (df[col_sel] > (Q3 + 1.5 * IQR))).sum()
                st.info(f"IQR法检测到 {n_outliers} 个异常点。")
                
            else:
                contamination = st.slider("异常比例阈值 (Contamination)", 0.01, 0.2, 0.05)
                outliers = EdaService.detect_outliers_isolation_forest(df, col_sel, contamination)
                
                # 可视化：散点图，红色为异常
                df_plot = df.copy()
                df_plot['Is Outlier'] = 'Normal'
                df_plot.loc[outliers.index, 'Is Outlier'] = 'Outlier'
                
                # 为了画图好看，加个随机抖动或者就在一维展示
                fig_iso = px.strip(df_plot, x=col_sel, color='Is Outlier', 
                                 color_discrete_map={'Normal':'blue', 'Outlier':'red'},
                                 title="Isolation Forest Detection")
                st.plotly_chart(fig_iso, use_container_width=True)
                st.warning(f"孤立森林检测到 {len(outliers)} 个潜在离群点 (可能是欺诈或罕见病例)。")

    # ==========================
    # Tab 3: 多重共线性 (VIF)
    # ==========================
    with tabs[2]:
        st.subheader("3. 多重共线性检查 (VIF)")
        st.markdown("""
        **VIF (方差膨胀因子) 解释**:
        - **VIF < 5**: 良好。
        - **5 < VIF < 10**: 值得关注，可能存在共线性。
        - **VIF > 10**: **严重共线性**。在线性模型(LR)中必须剔除，否则系数不可信；对树模型(XGBoost)影响较小。
        """)
        
        if st.button("🚀 开始计算 VIF (耗时操作)"):
            num_cols = [c for c, t in st.session_state['col_types'].items() if t == 'Numeric']
            if len(num_cols) < 2:
                st.error("数值型特征不足2个，无法计算 VIF。")
            else:
                with st.spinner("正在解方程组计算 VIF..."):
                    vif_df = EdaService.calculate_vif(df, num_cols)
                    
                    # 样式优化：高亮 VIF 大的行
                    def highlight_vif(val):
                        color = 'red' if val > 10 else 'orange' if val > 5 else 'green'
                        return f'color: {color}'
                    
                    st.dataframe(vif_df.style.applymap(highlight_vif, subset=['VIF']), use_container_width=True)

    # ==========================
    # Tab 4: 目标关联与偏差
    # ==========================
    with tabs[3]:
        st.subheader("4. 标签平衡与特征筛选")
        
        target_col = st.selectbox("🎯 请选择目标变量 (y):", df.columns, index=len(df.columns)-1)
        
        # 1. 类别不平衡检查
        if st.session_state['col_types'][target_col] == 'Categorical':
            st.markdown("#### ⚖️ 类别平衡性检查 (Class Balance)")
            counts = df[target_col].value_counts(normalize=True)
            counts_df = counts.reset_index()
            counts_df.columns = ['Class', 'Ratio']
            
            fig_bal = px.pie(counts_df, values='Ratio', names='Class', title=f"目标变量 {target_col} 分布")
            col_b1, col_b2 = st.columns([1, 2])
            with col_b1:
                st.dataframe(counts_df.style.format({'Ratio': '{:.2%}'}))
                min_class_ratio = counts.min()
                if min_class_ratio < 0.1:
                    st.error(f"⚠️ 检测到严重类别不平衡！最小类别占比仅 {min_class_ratio:.2%}。\n建议：后续建模使用 SMOTE 或 Class Weight。")
                else:
                    st.success("类别分布相对平衡。")
            with col_b2:
                st.plotly_chart(fig_bal, use_container_width=True)
        
        # 2. 特征关联度排行
        st.divider()
        st.markdown("#### 🏆 特征预测力排行")
        if st.button("计算特征关联度"):
            with st.spinner("正在进行卡方检验/互信息计算..."):
                rel_df = EdaService.analyze_target_relationship(df, target_col, st.session_state['col_types'])
                
                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    st.dataframe(rel_df, height=400)
                with col_r2:
                    fig_rel = px.bar(rel_df.head(10), x='Score', y='Feature', orientation='h', 
                                   title="Top 10 强相关特征", color='Score')
                    fig_rel.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_rel, use_container_width=True)

    # ==========================
    # Tab 5: 缺失值 (保留之前的代码)
    # ==========================
    with tabs[4]:

        st.subheader("4. 缺失值模式侦探")
        st.markdown("""
        **分析目的**:
        - **随机性判断**: 缺失是完全随机的(MCAR)，还是有规律的？(比如: '收入'缺失的人通常'职业'也缺失?)
        - **处理策略**: 如果某列缺失 > 50%，建议直接剔除；如果缺失 < 5%，可考虑均值/中位数填充。
        """)

        # 1. 缺失比例直方图
        missing_series = df.isnull().sum()
        missing_df = pd.DataFrame({
            "Feature": missing_series.index,
            "Missing Count": missing_series.values,
            "Missing Ratio": (missing_series.values / len(df)).round(4)
        }).sort_values(by="Missing Count", ascending=False)
        
        # 只显示有缺失值的列
        missing_df = missing_df[missing_df["Missing Count"] > 0]

        if not missing_df.empty:
            col_m1, col_m2 = st.columns([2, 1])
            with col_m1:
                fig_miss = px.bar(
                    missing_df, 
                    x="Feature", 
                    y="Missing Ratio",
                    text="Missing Count",
                    title="各特征缺失值比例",
                    labels={"Missing Ratio": "缺失比例 (0-1)"}
                )
                fig_miss.update_yaxes(range=[0, 1]) # 固定Y轴为0-100%
                st.plotly_chart(fig_miss, use_container_width=True)
            
            with col_m2:
                st.warning(f"共有 {len(missing_df)} 个特征存在缺失值。")
                st.dataframe(missing_df, hide_index=True)

            # 2. 缺失相关性 (Nullity Correlation)
            st.divider()
            st.markdown("### 缺失关联性热力图 (Nullity Correlation)")
            st.caption("颜色越深，说明两个特征倾向于**同时缺失** (正相关) 或 **一个缺失另一个就不缺失** (负相关)。")
            
            miss_corr = EdaService.get_missing_correlation(df)
            
            if miss_corr is not None:
                fig_null = px.imshow(
                    miss_corr,
                    text_auto=".2f",
                    color_continuous_scale="Viridis",
                    title="缺失模式相关性"
                )
                st.plotly_chart(fig_null, use_container_width=True)
            else:
                st.info("数据缺失模式过于简单或单一，无法计算缺失相关性。")
                
        else:
            st.success("🎉 太棒了！当前数据集中没有检测到任何缺失值。")

            