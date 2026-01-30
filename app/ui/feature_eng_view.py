import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from services.feature_eng_service import FeatureEngService

def _update_df_and_clear_cache(new_df, log_msg=None):
    """更新全局 dataframe 并清除下游任务（如特征选择）的缓存"""
    st.session_state['df'] = new_df
    
    # 记录操作日志
    if log_msg:
        st.session_state['change_log'].append(log_msg)
    
    # 清除特征选择产生的划分数据，强制重新划分
    keys_to_clear = ['X_train', 'X_test', 'y_train', 'y_test', 'suggestions']
    cleared = []
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
            cleared.append(k)
            
    if cleared:
        st.toast(f"⚠️ 数据已更新，重置了下游缓存: {cleared}", icon="🔄")

def render_feature_eng_view():
    st.header("⚙️ 特征工程工厂 (Feature Engineering)")

    if 'df' not in st.session_state:
        st.warning("请先上传数据")
        return
    
    # 懒加载初始化：确保有原始数据副本和日志
    if 'original_df' not in st.session_state:
        st.session_state['original_df'] = st.session_state['df'].copy()
    if 'change_log' not in st.session_state:
        st.session_state['change_log'] = []
    
    # 复制一份用于操作，避免直接修改原始 session 直到用户确认
    # 但为了简化流程，我们这里采用“直接应用并允许回滚”或“步步为营”的策略
    # 这里演示直接操作 Session 中的 df
    df = st.session_state['df']
    col_types = st.session_state.get('col_types', {})

    # 使用 Tabs 分离三大任务
    tab1, tab2, tab3 = st.tabs(["🧹 缺失值处理 (Cleaning)", "🧪 特征变换 (Transformation)", "🔍 最终数据验收 (Final Check)"])

    # ==========================
    # Tab 1: 缺失值处理
    # ==========================
    with tab1:
        st.subheader("1. 缺失值处理策略")
        
        # 自动检测含缺失值的列
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if not missing_cols:
            st.success("🎉 数据集非常干净，没有发现缺失值！")
        else:
            st.info(f"检测到以下列存在缺失值: {missing_cols}")
            
            # --- 区域 A: 删除策略 ---
            with st.expander("🗑️ 方案 A: 删除 (Deletion)", expanded=False):
                st.markdown("如果某列缺失率过高 (>50%)，建议直接删除。")
                del_mode = st.radio("删除维度:", ["删除列 (Feature)", "删除行 (Sample)"])
                
                if del_mode == "删除列 (Feature)":
                    threshold = st.slider("缺失率阈值 (超过此比例的列将被删除)", 0.1, 0.9, 0.5)
                    if st.button("执行列删除"):
                        new_df, dropped = FeatureEngService.drop_missing(df, threshold, axis=1)
                        _update_df_and_clear_cache(new_df, f"🗑️ 删除列 (Missing > {threshold*100}%): {dropped}")
                        st.success(f"已删除列: {dropped}")
                        st.rerun()
                else:
                    if st.button("执行行删除 (慎用)"):
                        new_df, count = FeatureEngService.drop_missing(df, axis=0)
                        _update_df_and_clear_cache(new_df, f"🗑️ 删除行: {count} 行样本")
                        st.success(f"已删除 {count} 行含有缺失值的样本。")
                        st.rerun()

            # --- 区域 B: 填充策略 ---
            with st.expander("💉 方案 B: 智能填充 (Imputation)", expanded=True):
                col_to_impute = st.multiselect("选择需要填充的特征:", missing_cols)
                
                method = st.selectbox(
                    "选择填充算法:",
                    [
                        "中位数 (Median) - 推荐用于偏态分布",
                        "均值 (Mean) - 仅适用于正态分布",
                        "众数 (Mode) - 适用于分类变量",
                        "KNN (K-近邻) - 基于相似样本填充 (慢但准)",
                        "MICE (多重插补) - 学术界金标准 (慢但最稳健)"
                    ]
                )
                
                # 映射选项到代码参数
                method_map = {
                    "中位数 (Median)": "median",
                    "均值 (Mean)": "mean",
                    "众数 (Mode)": "most_frequent",
                    "KNN (K-近邻)": "knn",
                    "MICE (多重插补)": "mice"
                }
                
                if st.button("开始填充"):
                    if not col_to_impute:
                        st.error("请先选择列！")
                    else:
                        selected_method = method_map[method.split(' - ')[0]]
                        
                        # 检查类型限制
                        is_advanced = selected_method in ['knn', 'mice']
                        # 如果是高级填充，确保选的都是数值型
                        if is_advanced:
                            non_numeric = [c for c in col_to_impute if col_types.get(c) != 'Numeric']
                            if non_numeric:
                                st.error(f"KNN/MICE 仅支持数值型特征。检测到非数值列: {non_numeric}，请改用众数填充。")
                                st.stop()

                        with st.spinner(f"正在使用 {selected_method} 进行填充..."):
                            if is_advanced:
                                new_df = FeatureEngService.advanced_impute(df, col_to_impute, selected_method)
                            else:
                                new_df = FeatureEngService.simple_impute(df, col_to_impute, selected_method)
                            
                            # 将结果暂存到 session_state 用于预览，而不是直接覆盖 'df'
                            st.session_state['preview_df'] = new_df
                            st.session_state['impute_cols'] = col_to_impute
                            st.success(f"✅ 运算完成！请查看下方对比，满意请点击底部的【确认应用】。")

                # --- 预览区域 ---
                if 'preview_df' in st.session_state:
                    new_df = st.session_state['preview_df']
                    impute_cols = st.session_state.get('impute_cols', [])
                    
                    st.divider()
                    st.subheader("📊 填充效果对比 (Before vs After)")
                    
                    # 1. 数据值对比 (Sample)
                    st.caption("数据样本对比 (前5行)")
                    col_b, col_a = st.columns(2)
                    with col_b:
                        st.markdown("**原始数据 (Original)**")
                        st.dataframe(df[impute_cols].head())
                    with col_a:
                        st.markdown("**填充后 (Imputed)**")
                        st.dataframe(new_df[impute_cols].head())
                    
                    # 2. 分布对比 (仅对数值型有效)
                    numeric_imputed = [c for c in impute_cols if col_types.get(c) == 'Numeric']
                    if numeric_imputed:
                        st.caption("分布形态变化检测")
                        target_viz = st.selectbox("选择查看分布的列:", numeric_imputed, key='viz_impute')
                        
                        fig = go.Figure()
                        # 原始分布
                        fig.add_trace(go.Histogram(
                            x=df[target_viz], 
                            name='Before (Original)', 
                            opacity=0.5, 
                            marker_color='gray'
                        ))
                        # 填充后分布
                        fig.add_trace(go.Histogram(
                            x=new_df[target_viz], 
                            name='After (Imputed)', 
                            opacity=0.5, 
                            marker_color='blue'
                        ))
                        fig.update_layout(barmode='overlay', title=f"{target_viz} 填充前后分布对比")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 确认按钮
                    col_confirm, col_cancel = st.columns([1, 4])
                    if col_confirm.button("✅ 确认应用修改", type="primary"):
                        _update_df_and_clear_cache(new_df, f"💉 缺失值填充: {impute_cols}")
                        del st.session_state['preview_df'] # 清除预览状态
                        st.success("修改已保存！")
                        st.rerun()
                    
                    if col_cancel.button("❌ 放弃修改"):
                        del st.session_state['preview_df']
                        st.rerun()

    # ==========================
    # Tab 2: 特征变换
    # ==========================
    with tab2:
        st.subheader("2. 特征变换实验室")
        
        # 筛选数值型特征
        num_cols = [c for c, t in col_types.items() if t == 'Numeric']
        
        target_col = st.selectbox("选择要变换的特征:", num_cols)
        
        # 展示变换前的分布
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔴 变换前 (Original)")
            fig_before = px.histogram(df, x=target_col, title="原始分布", marginal="box")
            st.plotly_chart(fig_before, use_container_width=True)
            st.metric("原始偏度 (Skew)", f"{df[target_col].skew():.2f}")

        # 变换工具箱
        st.divider()
        action = st.radio("选择变换操作:", ["数据标准化 (Scaling)", "非线性变换 (Gaussian)", "离散化 (Binning)"], horizontal=True)

        new_df_preview = df.copy() # 默认从原始数据开始
        
        # 尝试从 session_state 获取之前的预览（如果有且针对同一列）
        if 'trans_preview_df' in st.session_state:
            # 只有当预览的是当前选择的列时才使用，否则视为切换了列，重置
            if st.session_state.get('trans_target_col') == target_col:
                new_df_preview = st.session_state['trans_preview_df']
            else:
                # 切换了列，清除旧预览
                del st.session_state['trans_preview_df']
                if 'trans_target_col' in st.session_state:
                    del st.session_state['trans_target_col']

        if action == "数据标准化 (Scaling)":
            method = st.selectbox("方法:", ["Z-Score (StandardScaler)", "Min-Max (0-1)"])
            m_code = 'standard' if 'Z-Score' in method else 'minmax'
            if st.button("应用并预览"):
                new_df_preview = FeatureEngService.scale_features(df, [target_col], m_code)
                # 保存状态
                st.session_state['trans_preview_df'] = new_df_preview
                st.session_state['trans_target_col'] = target_col
                st.rerun() # 强制刷新以更新图表
        
        elif action == "非线性变换 (Gaussian)":
            method = st.selectbox("方法:", ["Log (对数变换)", "Box-Cox (需正数)", "Yeo-Johnson (通用)"])
            m_code = method.split(' ')[0].lower()
            if st.button("应用并预览"):
                try:
                    new_df_preview = FeatureEngService.gaussian_transform(df, [target_col], m_code)
                    # 保存状态
                    st.session_state['trans_preview_df'] = new_df_preview
                    st.session_state['trans_target_col'] = target_col
                    st.rerun()
                except Exception as e:
                    st.error(f"变换失败: {str(e)}")

        elif action == "离散化 (Binning)":
            n_bins = st.slider("分箱数量:", 2, 10, 5)
            strategy = st.selectbox("策略:", ["quantile (等频)", "uniform (等宽)"])
            if st.button("应用并预览"):
                new_df_preview = FeatureEngService.discretize_features(df, [target_col], n_bins, strategy)
                # 保存状态
                st.session_state['trans_preview_df'] = new_df_preview
                st.session_state['trans_target_col'] = target_col
                st.rerun()

        # 预览结果
        with col2:
            st.markdown("#### 🟢 变换后 (Transformed)")
            
            # 如果是 Binning，数据变成 0,1,2，可能需要作为 Categorical 画图
            if action == "离散化 (Binning)" and target_col in new_df_preview:
                fig_after = px.bar(new_df_preview[target_col].value_counts().reset_index(), x='index', y=target_col, title="分箱后分布")
            else:
                fig_after = px.histogram(new_df_preview, x=target_col, title="变换后分布", marginal="box")
                st.metric("新偏度 (New Skew)", f"{new_df_preview[target_col].skew():.2f}")
            
            st.plotly_chart(fig_after, use_container_width=True)

        # 数据值对比 (新增)
        if not new_df_preview.equals(df):
            st.divider()
            st.markdown("#### 🔢 数据值前后对比 (Values Comparison)")
            col_val_b, col_val_a = st.columns(2)
            with col_val_b:
                st.caption("变换前 (Before)")
                st.dataframe(df[[target_col]].head())
            with col_val_a:
                st.caption("变换后 (After)")
                st.dataframe(new_df_preview[[target_col]].head())

        # 确认保存按钮
        st.divider()
        col_save, col_discard = st.columns([1, 4])
        
        if col_save.button("💾 确认保存该变换"):
            if new_df_preview.equals(df):
                 st.warning("⚠️ 没有检测到变化，请先点击【应用并预览】")
            else:
                _update_df_and_clear_cache(new_df_preview, f"🧪 特征变换: {target_col} -> {action}")
                # 保存后清除预览状态
                if 'trans_preview_df' in st.session_state:
                    del st.session_state['trans_preview_df']
                if 'trans_target_col' in st.session_state:
                    del st.session_state['trans_target_col']
                    
                st.success(f"特征 {target_col} 已更新！")
                st.rerun()
                
        if col_discard.button("❌ 撤销预览"):
             if 'trans_preview_df' in st.session_state:
                del st.session_state['trans_preview_df']
                st.rerun()

    # ==========================
    # Tab 3: 最终数据验收
    # ==========================
    with tab3:
        st.subheader("🏁 数据最终体检 & 变更审计")
        
        current_df = st.session_state.get('df')
        original_df = st.session_state.get('original_df')
        changes = st.session_state.get('change_log', [])
        
        # 1. 变更日志 (Change Log)
        if changes:
            st.info(f"📝 累计检测到 {len(changes)} 项修改操作：")
            for idx, msg in enumerate(changes, 1):
                st.markdown(f"**{idx}.** {msg}")
        else:
            st.warning("⚠️ 尚未检测到任何数据修改操作（您使用的是原始数据）。")

        st.divider()

        # 2. 宏观对比 (Original vs Current)
        st.markdown("### 🔍 全局变化对比 (Original vs Current)")
        
        c1, c2, c3 = st.columns(3)
        # 行数变化
        delta_rows = current_df.shape[0] - original_df.shape[0]
        c1.metric("样本数 (Rows)", current_df.shape[0], delta=f"{delta_rows} 行" if delta_rows != 0 else None)
        
        # 列数变化
        delta_cols = current_df.shape[1] - original_df.shape[1]
        c2.metric("特征数 (Cols)", current_df.shape[1], delta=f"{delta_cols} 列" if delta_cols != 0 else None)
        
        # 缺失值变化
        orig_missing = original_df.isnull().sum().sum()
        curr_missing = current_df.isnull().sum().sum()
        delta_missing = curr_missing - orig_missing
        c3.metric("缺失值总数", curr_missing, delta=f"{delta_missing}" if delta_missing != 0 else "无变化", delta_color="inverse")

        # 3. 数据内容对比 (Side-by-Side)
        st.divider()
        st.markdown("### 🔢 数据内容快照 (Snapshot)")
        
        col_view_1, col_view_2 = st.columns(2)
        with col_view_1:
            st.markdown("**原始数据 (Original - Top 10)**")
            st.dataframe(original_df.head(10))
        with col_view_2:
            st.markdown("**当前数据 (Current - Top 10)**")
            st.dataframe(current_df.head(10))
            
        # 4. 缺失值残留检测
        if curr_missing > 0:
            st.error(f"❌ 警告：仍有 {curr_missing} 个缺失值未处理！建议返回 Tab 1。")
        else:
            st.success("✅ 数据完整性校验通过：无缺失值。")

        # 5. 心理确认
        st.divider()
        st.info("💡 提示：以上即为即将进入【特征筛选】模块的最终数据。")