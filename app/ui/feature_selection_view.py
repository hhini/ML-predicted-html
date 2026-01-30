import streamlit as st
import pandas as pd
import plotly.express as px
from services.feature_selection_service import FeatureSelectionService

def render_feature_selection_view():
    st.header("🧬 特征筛选漏斗 (Feature Selection Funnel)")

    if 'df' not in st.session_state:
        st.warning("请先上传数据")
        return

    df = st.session_state['df']
    col_types = st.session_state.get('col_types', {})

    # 1. 任务定义
    with st.expander("🎯 任务目标设置", expanded=True):
        target_col = st.selectbox("选择目标变量 (Target):", df.columns, index=len(df.columns)-1)
        task_type = "classification" if col_types.get(target_col) == 'Categorical' else "regression"
        st.info(f"当前任务被识别为: **{task_type}**")

    # 2. 必须先划分数据
    if 'X_train' not in st.session_state:
        st.warning("⚠️ 严谨警告：在进行特征选择前，必须冻结测试集，防止数据泄露！")
        
        split_ratio = st.slider("训练集占比 (Train Size)", 0.5, 0.9, 0.8)
        if st.button("🔒 执行分层抽样划分 (Stratified Split)"):
            with st.spinner("正在划分数据..."):
                X_train, X_test, y_train, y_test = FeatureSelectionService.split_data(
                    df, target_col, split_ratio, task_type
                )
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.success(f"划分完成！训练集: {X_train.shape[0]} 行, 测试集: {X_test.shape[0]} 行")
                st.rerun()
    else:
        st.success(f"✅ 数据已锁定 (Train: {len(st.session_state['X_train'])}, Test: {len(st.session_state['X_test'])})")
        if st.button("🔄 重置划分 (慎点)"):
            del st.session_state['X_train']
            st.rerun()

    # 只有划分了数据才能继续
    if 'X_train' in st.session_state:
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        
        # 增加测试集可视化确认
        if 'X_test' in st.session_state and 'y_test' in st.session_state:
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            with st.expander("📦 查看已冻结的测试集 (Test Set Preview)", expanded=False):
                st.caption(f"测试集包含 {len(X_test)} 个样本，将在模型评估阶段使用。")
                c_t1, c_t2 = st.columns(2)
                with c_t1:
                    st.markdown("**X_test (Features)**")
                    st.dataframe(X_test.head())
                with c_t2:
                    st.markdown("**y_test (Target)**")
                    st.dataframe(y_test.head())
                
                # 简单的分布确认
                if task_type == 'classification':
                    st.caption("测试集标签分布:")
                    st.dataframe(y_test.value_counts(normalize=True).rename("Proportion"))

        st.divider()
        st.subheader("🛠️ 特征筛选工具箱")
        
        # 初始化候选建议字典
        if 'suggestions' not in st.session_state:
            st.session_state['suggestions'] = {}
        
        # 初始化结果持久化存储 (防止刷新丢失)
        if 'filter_results' not in st.session_state:
            st.session_state['filter_results'] = {} # 存放已确认的候选集
        if 'embedded_results' not in st.session_state:
            st.session_state['embedded_results'] = {}
        if 'wrapper_results' not in st.session_state:
            st.session_state['wrapper_results'] = {}
        
        # 初始化中间分析结果 (用于展示报表供用户挑选)
        if 'analysis_filter' not in st.session_state:
            st.session_state['analysis_filter'] = {}
        if 'analysis_embedded' not in st.session_state:
            st.session_state['analysis_embedded'] = {}
        if 'analysis_wrapper' not in st.session_state:
            st.session_state['analysis_wrapper'] = {}

        tab1, tab2, tab3, tab4 = st.tabs([
            "1️⃣ 过滤法 (Filter)", 
            "2️⃣ 嵌入法 (Embedded)", 
            "3️⃣ 包装法 (Wrapper)", 
            "🏆 最终决策 (Decision)"
        ])

        # --- Tab 1: Filter ---
        with tab1:
            st.markdown("**Step 1: 统计分析** (计算特征与目标的相关性/显著性)")
            available_methods = ["chi2 (卡方)", "anova (方差分析)", "mutual_info (互信息)"] if task_type == 'classification' else ["pearson (皮尔逊)", "mutual_info (互信息)"]
            
            methods_f = st.multiselect("选择分析方法:", available_methods, default=[available_methods[0]])
            
            if st.button("🚀 开始分析 (Run Analysis)", key="btn_filter_analyze"):
                if not methods_f:
                    st.warning("请至少选择一种方法！")
                else:
                    st.session_state['analysis_filter'] = {} # 清空旧分析
                    for method_str in methods_f:
                        method_name = method_str.split(' ')[0]
                        with st.spinner(f"正在运行 {method_name}..."):
                            res_f = FeatureSelectionService.filter_selection(
                                X_train, y_train, task_type, method_name
                            )
                            st.session_state['analysis_filter'][method_str] = res_f
            
            # 展示分析结果并允许选择
            if st.session_state['analysis_filter']:
                st.divider()
                st.markdown("**Step 2: 查阅报告并筛选**")
                
                for method_str, res_f in st.session_state['analysis_filter'].items():
                    with st.expander(f"📊 分析报告: {method_str}", expanded=True):
                        # 格式化显示
                        format_dict = {"Score": "{:.4f}"}
                        if 'P-Value' in res_f.columns and res_f['P-Value'].notna().any():
                            format_dict["P-Value"] = "{:.4e}"
                        
                        c1, c2 = st.columns([3, 2])
                        with c1:
                            st.dataframe(res_f.style.format(format_dict, na_rep="N/A"), use_container_width=True, height=300)
                        with c2:
                            st.plotly_chart(px.bar(res_f.head(15), x='Score', y='Feature', orientation='h', title="Top 15 Features"), use_container_width=True)
                        
                        # 交互式筛选区
                        st.markdown("---")
                        c_sel1, c_sel2 = st.columns([3, 1])
                        with c_sel1:
                            # 默认选中 Top 20%
                            default_k = max(1, int(len(res_f) * 0.2))
                            default_feats = res_f.head(default_k)['Feature'].tolist()
                            
                            selected_feats = st.multiselect(
                                f"👉 请勾选要保留的特征 ({method_str}):", 
                                options=res_f['Feature'].tolist(),
                                default=default_feats,
                                key=f"sel_f_{method_str}"
                            )
                        with c_sel2:
                            st.write("") # Spacer
                            st.write("")
                            if st.button("📥 加入候选池", key=f"add_f_{method_str}"):
                                key_name = f"Filter ({method_str.split(' ')[0]})"
                                st.session_state['suggestions'][key_name] = selected_feats
                                # 同时也保存结果以便后续查看
                                st.session_state['filter_results'][key_name] = {'df': res_f, 'top_k': len(selected_feats)}
                                st.toast(f"已添加 {len(selected_feats)} 个特征到候选池！", icon="✅")

        # --- Tab 2: Embedded ---
        with tab2:
            st.markdown("**Step 1: 模型重要性评估** (基于模型训练)")
            available_models = ["random_forest (随机森林)", "xgboost (XGBoost)", "lasso (L1正则)"]
            
            c_e1, c_e2 = st.columns([2, 1])
            with c_e1:
                methods_e = st.multiselect("选择算法模型:", available_models, default=[available_models[0]])
            with c_e2:
                use_perm = st.checkbox("使用 Permutation Importance", value=False, help="更严谨但速度较慢")
            
            if st.button("🚀 开始评估 (Run Analysis)", key="btn_embed_analyze"):
                if not methods_e:
                    st.warning("请至少选择一种模型！")
                else:
                    st.session_state['analysis_embedded'] = {}
                    for method_str in methods_e:
                        method_name = method_str.split(' ')[0]
                        with st.spinner(f"正在训练 {method_name}..."):
                            res_e = FeatureSelectionService.embedded_selection(
                                X_train, y_train, task_type, method_name, use_perm
                            )
                            st.session_state['analysis_embedded'][method_str] = res_e

            # 展示分析结果并允许选择
            if st.session_state['analysis_embedded']:
                st.divider()
                st.markdown("**Step 2: 查阅报告并筛选**")
                
                for method_str, res_e in st.session_state['analysis_embedded'].items():
                    with st.expander(f"🌲 重要性报告: {method_str}", expanded=True):
                        c1, c2 = st.columns([3, 2])
                        with c1:
                            st.dataframe(res_e.style.format({"Importance": "{:.4f}"}), use_container_width=True, height=300)
                        with c2:
                            st.plotly_chart(px.bar(res_e.head(15), x='Importance', y='Feature', orientation='h', title="Top 15 Features"), use_container_width=True)
                        
                        # 交互式筛选区
                        st.markdown("---")
                        c_sel1, c_sel2 = st.columns([3, 1])
                        with c_sel1:
                            # 默认选中 Top 15
                            default_feats = res_e.head(15)['Feature'].tolist()
                            
                            selected_feats = st.multiselect(
                                f"👉 请勾选要保留的特征 ({method_str}):", 
                                options=res_e['Feature'].tolist(),
                                default=default_feats,
                                key=f"sel_e_{method_str}"
                            )
                        with c_sel2:
                            st.write("")
                            st.write("")
                            if st.button("📥 加入候选池", key=f"add_e_{method_str}"):
                                key_name = f"Embedded ({method_str.split(' ')[0]})"
                                st.session_state['suggestions'][key_name] = selected_feats
                                st.session_state['embedded_results'][key_name] = res_e
                                st.toast(f"已添加 {len(selected_feats)} 个特征到候选池！", icon="✅")

        # --- Tab 3: Wrapper ---
        with tab3:
            st.markdown("**Step 1: 递归消除 (RFE)** (按重要性排序)")
            
            n_features = st.number_input("RFE 目标特征数 (仅用于算法停止条件, 您可以在下方自由选择):", min_value=1, max_value=len(X_train.columns), value=10)
            
            if st.button("🚀 开始递归计算 (Run RFE)", key="btn_wrapper_analyze"):
                with st.spinner(f"正在递归训练模型..."):
                    res_w = FeatureSelectionService.wrapper_selection(
                        X_train, y_train, task_type, n_features_to_select=n_features
                    )
                    st.session_state['analysis_wrapper']['RFE'] = res_w
            
            if st.session_state['analysis_wrapper']:
                st.divider()
                st.markdown("**Step 2: 查阅排名并筛选**")
                
                res_w = st.session_state['analysis_wrapper']['RFE']
                with st.expander("🔄 RFE 排名报告", expanded=True):
                    st.dataframe(res_w, use_container_width=True, height=300)
                    
                    st.markdown("---")
                    c_sel1, c_sel2 = st.columns([3, 1])
                    with c_sel1:
                        # 默认选中 Ranking <= 10 的特征
                        default_feats = res_w[res_w['Ranking'] <= 10]['Feature'].tolist()
                        
                        selected_feats = st.multiselect(
                            "👉 请勾选要保留的特征:", 
                            options=res_w['Feature'].tolist(),
                            default=default_feats,
                            key="sel_w_rfe"
                        )
                    with c_sel2:
                        st.write("")
                        st.write("")
                        if st.button("📥 加入候选池", key="add_w_rfe"):
                            key_name = f"Wrapper (RFE)"
                            st.session_state['suggestions'][key_name] = selected_feats
                            st.session_state['wrapper_results'][key_name] = res_w
                            st.toast(f"已添加 {len(selected_feats)} 个特征到候选池！", icon="✅")

        # --- Tab 4: Final Decision ---
        with tab4:
            st.subheader("🏆 最终特征决策面板")
            
            suggestions = st.session_state.get('suggestions', {})
            
            if not suggestions:
                st.warning("请先在前三个 Tab 中运行至少一种筛选方法！")
            else:
                # 1. 展示各方法的推荐结果
                st.markdown("#### 📥 当前候选池 (已运行的方法)")
                
                # 增加一个清除按钮
                if st.button("🗑️ 清空所有候选池"):
                    st.session_state['suggestions'] = {}
                    st.session_state['filter_results'] = {}
                    st.session_state['embedded_results'] = {}
                    st.session_state['wrapper_results'] = {}
                    st.rerun()

                cols = st.columns(min(len(suggestions), 4)) # 最多显示4列，多了换行不好排，这里简化处理
                # 如果超过4个方法，可能需要改进显示方式，这里先简单罗列
                for idx, (method, feats) in enumerate(suggestions.items()):
                    with cols[idx % 4]: # 循环列
                        st.markdown(f"**{method}**")
                        st.caption(f"选出 {len(feats)} 个特征")
                        with st.expander("查看列表"):
                            st.write(feats)
                
                # 2. 投票机制 (Voting)
                st.divider()
                st.markdown("#### 🗳️ 智能投票 (Consensus Voting)")
                
                # 统计每个特征被几个方法推荐了
                all_candidates = []
                for feats in suggestions.values():
                    all_candidates.extend(feats)
                
                from collections import Counter
                vote_counts = Counter(all_candidates)
                
                vote_df = pd.DataFrame(vote_counts.items(), columns=['Feature', 'Votes']).sort_values('Votes', ascending=False)
                
                # 区分 强推荐 (3票) / 中等 (2票) / 弱 (1票)
                st.bar_chart(vote_df.set_index('Feature'))
                
                strong_feats = vote_df[vote_df['Votes'] == len(suggestions)]['Feature'].tolist()
                consensus_feats = vote_df[vote_df['Votes'] >= 2]['Feature'].tolist()
                
                if strong_feats:
                    st.success(f"🌟 **五星推荐 (被所有方法选中)**: {strong_feats}")
                if consensus_feats:
                    st.info(f"✅ **稳健推荐 (至少被2个方法选中)**: {consensus_feats}")
                
                # 3. 最终确认
                st.divider()
                final_choice = st.multiselect(
                    "请勾选最终要保留的特征:", 
                    options=X_train.columns,
                    default=consensus_feats if consensus_feats else vote_df.head(10)['Feature'].tolist()
                )
                
                if st.button("🔒 确认并保存最终特征集"):
                    # 这里可以将筛选后的 X_train 存回去，或者标记哪些列被 drop 了
                    # 为了简单，我们存一个 selected_features 列表到 session
                    st.session_state['selected_features'] = final_choice
                    st.success(f"已锁定 {len(final_choice)} 个特征！下一步可以进行模型训练了。")