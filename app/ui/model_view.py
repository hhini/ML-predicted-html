import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from services.model_service import ModelService
import io
import joblib

def render_model_view():
    st.header("⚔️ AutoML 竞技场 (Model Arena)")
    
    if 'X_train' not in st.session_state:
        st.error("请先在【特征筛选】模块中完成数据划分！")
        return

    X_train = st.session_state['X_train']
    y_train = st.session_state['y_train']
    
    # 获取之前筛选好的特征 (如果有)
    final_feats = st.session_state.get('final_features', list(X_train.columns))
    # 过滤 X_train
    X_train_final = X_train[final_feats]
    
    # 任务类型判断
    # 简单的判断逻辑，实际可复用 session 中的 task_type
    is_classification = y_train.nunique() < 20 
    task_type = 'classification' if is_classification else 'regression'

    # --- 配置区 ---
    with st.expander("⚙️ 实验配置 (Experimental Setup)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 1. 选择参赛模型")
            available_models = [
                "Logistic Regression" if is_classification else "Linear Regression",
                "Random Forest", "XGBoost", "LightGBM", "SVM", "KNN", "MLP (Neural Net)"
            ]
            selected_models = st.multiselect("Models:", available_models, default=["Random Forest", "XGBoost"])
        
        with col2:
            st.markdown("##### 2. 验证与调参")
            cv_folds = st.slider("K-Fold 折数:", 3, 10, 5)
            use_optuna = st.checkbox("开启贝叶斯调参 (Bayesian Opt)", value=False)
            n_trials = st.slider("调参迭代次数 (Trials):", 5, 200, 10, disabled=not use_optuna)
            st.caption("注：开启调参会显著增加运行时间，但能提升模型性能。")

    # --- 运行按钮 ---
    if st.button("🚀 开始竞技 (Run AutoML)"):
        with st.spinner("正在进行 K-Fold 交叉验证与贝叶斯调参... 请耐心等待"):
            # 调用 Service
            results = ModelService.train_and_evaluate(
                X_train_final, y_train, task_type, selected_models, 
                n_splits=cv_folds, use_optuna=use_optuna, n_trials=n_trials
            )
            st.session_state['model_results'] = results
            st.success("训练完成！")

    # --- 结果展示区 ---
    if 'model_results' in st.session_state:
        results = st.session_state['model_results']
        
        # 准备指标汇总表
        metrics_summary = []
        for name, res in results.items():
            row = res['metrics']
            row['Model'] = name
            metrics_summary.append(row)
        df_metrics = pd.DataFrame(metrics_summary).set_index('Model')
        
        # 1. 核心指标排行榜
        st.divider()
        st.subheader("🏆 模型排行榜 (Leaderboard)")
        # 高亮最优值
        st.dataframe(df_metrics.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

        # Tabs 展示高级可视化
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📈 ROC & 混淆矩阵", 
            "🕸️ 多维雷达图", 
            "📏 校准曲线", 
            "🔬 统计显著性",
            "🔧 最优参数",
            "✨ 特征重要性",
            "📚 学习曲线",
            "💾 模型导出"
        ])

        if is_classification:
            # === Tab 1: ROC & Confusion Matrix ===
            with tab1:
                col_roc, col_cm = st.columns([1, 1])
                
                # A. 组合 ROC 曲线
                with col_roc:
                    fig_roc = go.Figure()
                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    
                    for name, res in results.items():
                        roc = res['roc_data']
                        fig_roc.add_trace(go.Scatter(
                            x=roc['fpr'], y=roc['tpr'], 
                            mode='lines', name=f"{name} (AUC={roc['auc']:.3f})"
                        ))
                    
                    fig_roc.update_layout(title="组合 ROC 曲线 (Combined ROC)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", width=500, height=500)
                    st.plotly_chart(fig_roc, use_container_width=True)

                # B. 归一化混淆矩阵
                with col_cm:
                    st.markdown("##### 混淆矩阵 (Confusion Matrix)")
                    model_cm = st.selectbox("选择模型:", selected_models, key="cm_select")
                    
                    # 动态阈值调整
                    threshold = st.slider("判定阈值 (Decision Threshold)", 0.0, 1.0, 0.5, 0.01, key=f"thresh_{model_cm}")
                    
                    y_true = results[model_cm]['y_true']
                    y_probs = results[model_cm]['y_pred'] # 概率
                    # 根据阈值重新生成预测类别
                    y_pred_adj = [1 if p >= threshold else 0 for p in y_probs]
                    
                    cm = confusion_matrix(y_true, y_pred_adj, normalize='true') # 归一化！
                    
                    fig_cm = px.imshow(
                        cm, text_auto=".2%", aspect="equal",
                        color_continuous_scale="Blues",
                        title=f"{model_cm} Normalized Confusion Matrix (Thresh={threshold})",
                        labels=dict(x="Predicted", y="True")
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

            # === Tab 2: Radar Chart ===
            with tab2:
                st.markdown("##### 模型综合能力评估 (Radar Chart)")
                # 数据标准化到 0-1 之间以便画图 (虽然 metrics 都是 0-1，但以防万一)
                categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
                fig_radar = go.Figure()
                
                for name, res in results.items():
                    values = [res['metrics'][c] for c in categories]
                    # 闭环
                    values += [values[0]]
                    cats_closed = categories + [categories[0]]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values, theta=cats_closed, fill='toself', name=name
                    ))
                
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
                st.plotly_chart(fig_radar, use_container_width=True)

            # === Tab 3: Calibration Curve ===
            with tab3:
                st.markdown("##### 校准曲线 (Calibration Curve)")
                st.caption("越接近对角线，说明模型预测的概率越真实 (Reliability)。")
                fig_cal = go.Figure()
                fig_cal.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)

                for name, res in results.items():
                    if hasattr(res, 'best_params'): # 简单check
                        prob_true, prob_pred = calibration_curve(res['y_true'], res['y_pred'], n_bins=10)
                        fig_cal.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name=name))
                
                fig_cal.update_layout(xaxis_title="Mean Predicted Probability", yaxis_title="Fraction of Positives")
                st.plotly_chart(fig_cal, use_container_width=True)

        # === Tab 4: Statistical Significance ===
        with tab4:
            st.subheader("Wilcoxon Signed-Rank Test Heatmap")
            st.caption("对比各模型之间是否存在显著差异 (P-Value < 0.05 代表显著)。")
            
            p_values = pd.DataFrame(index=selected_models, columns=selected_models)
            for m1 in selected_models:
                for m2 in selected_models:
                    if m1 == m2:
                        p_values.loc[m1, m2] = np.nan
                    else:
                        # 获取每一折的 metric (例如 Accuracy)
                        scores1 = results[m1]['fold_metrics']['Accuracy' if is_classification else 'R2']
                        scores2 = results[m2]['fold_metrics']['Accuracy' if is_classification else 'R2']
                        # Wilcoxon 检验
                        try:
                            _, p = stats.wilcoxon(scores1, scores2)
                            p_values.loc[m1, m2] = p
                        except:
                            p_values.loc[m1, m2] = 1.0 # 样本完全一致时
            
            # 绘制 P-Value 热力图
            p_values = p_values.astype(float)
            fig_p = px.imshow(
                p_values, text_auto=".3f", 
                color_continuous_scale="Reds_r", # 红色越深P值越小(显著)
                title="P-Value Matrix (Green/White = Not Significant, Red = Significant)"
            )
            st.plotly_chart(fig_p, use_container_width=True)

        # === Tab 5: Best Params ===
        with tab5:
            st.json({name: res['best_params'] for name, res in results.items()})

        # === Tab 6: Feature Importance ===
        with tab6:
            st.markdown("##### 特征重要性 (Feature Importance)")
            st.caption("展示模型在全量训练集上学到的特征权重。")
            
            model_fi = st.selectbox("选择模型:", selected_models, key="fi_select")
            if model_fi in results:
                fi = results[model_fi].get('feature_importance')
                
                if fi is not None:
                    # 确保维度匹配 (防止 Pipeline 中特征数量变化)
                    if len(fi) == len(final_feats):
                        df_fi = pd.DataFrame({"Feature": final_feats, "Importance": fi})
                        df_fi = df_fi.sort_values("Importance", ascending=True)
                        
                        fig_fi = px.bar(
                            df_fi, x="Importance", y="Feature", orientation='h', 
                            title=f"{model_fi} Feature Importance",
                            height=max(400, len(final_feats) * 20)
                        )
                        st.plotly_chart(fig_fi, use_container_width=True)
                    else:
                        st.warning(f"特征数量不匹配 (Features: {len(final_feats)}, Importances: {len(fi)})。可能模型内部进行了特征处理。")
                else:
                    st.info(f"模型 {model_fi} 不支持直接提取特征重要性 (如 KNN, SVM-RBF)。")

        # === Tab 7: Learning Curve ===
        with tab7:
            st.markdown("##### 学习曲线 (Learning Curve)")
            st.caption("诊断模型是过拟合 (Overfitting) 还是欠拟合 (Underfitting)。")
            
            col_lc1, col_lc2 = st.columns([1, 3])
            with col_lc1:
                model_lc = st.selectbox("选择模型:", selected_models, key="lc_select")
                run_lc = st.button("生成曲线 (Run)", help="计算耗时较长，请耐心等待")
            
            with col_lc2:
                if run_lc:
                    with st.spinner(f"正在为 {model_lc} 计算学习曲线..."):
                        final_model = results[model_lc]['final_model']
                        # 必须使用未训练过的 clone 模型吗？learning_curve 会自动 clone
                        # 但我们需要原始参数。final_model 已经 fit 过了，但 learning_curve 会 clone 它并重新 fit
                        
                        try:
                            train_sizes, train_scores, test_scores = ModelService.compute_learning_curve(
                                final_model, X_train_final, y_train, cv=5
                            )
                            
                            train_mean = np.mean(train_scores, axis=1)
                            train_std = np.std(train_scores, axis=1)
                            test_mean = np.mean(test_scores, axis=1)
                            test_std = np.std(test_scores, axis=1)
                            
                            fig_lc = go.Figure()
                            
                            # Training Score
                            fig_lc.add_trace(go.Scatter(
                                x=train_sizes, y=train_mean, mode='lines+markers',
                                name='Training Score', line=dict(color='blue')
                            ))
                            # Validation Score
                            fig_lc.add_trace(go.Scatter(
                                x=train_sizes, y=test_mean, mode='lines+markers',
                                name='CV Score', line=dict(color='green')
                            ))
                            
                            fig_lc.update_layout(
                                title=f"Learning Curve ({model_lc})",
                                xaxis_title="Training Examples",
                                yaxis_title="Score",
                                yaxis_range=[0, 1.05]
                            )
                            st.plotly_chart(fig_lc, use_container_width=True)
                            
                            # 简要分析
                            gap = train_mean[-1] - test_mean[-1]
                            if gap > 0.1:
                                st.warning("⚠️ 存在较大的过拟合风险 (High Variance)。建议：增加数据量、减少特征、增加正则化。")
                            elif test_mean[-1] < 0.6: # 阈值仅供参考
                                st.warning("⚠️ 存在欠拟合风险 (High Bias)。建议：使用更复杂的模型、增加特征。")
                            else:
                                st.success("✅ 模型泛化能力良好。")
                                
                        except Exception as e:
                            st.error(f"计算失败: {str(e)}")

        # === Tab 8: Model Export ===
        with tab8:
            st.markdown("##### 导出模型 (Download Model)")
            st.caption("下载已在全量训练集上训练好的模型文件 (.pkl)。")
            
            c_ex1, c_ex2 = st.columns([1, 1])
            with c_ex1:
                model_ex = st.selectbox("选择要导出的模型:", selected_models, key="ex_select")
            
            with c_ex2:
                if model_ex in results:
                    final_model = results[model_ex]['final_model']
                    
                    # 序列化
                    buffer = io.BytesIO()
                    joblib.dump(final_model, buffer)
                    buffer.seek(0)
                    
                    st.download_button(
                        label=f"⬇️ 下载 {model_ex}.pkl",
                        data=buffer,
                        file_name=f"{model_ex}.pkl",
                        mime="application/octet-stream"
                    )