import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
from services.explanation_service import ExplanationService

def render_prediction_view():
    st.header("🔮 解释性分析与部署 (XAI & Deployment)")

    # 1. 检查前置条件
    if 'model_results' not in st.session_state:
        st.error("❌ 请先完成【模型训练】环节！")
        return
    if 'X_test' not in st.session_state:
        st.error("❌ 未检测到测试集，请返回【特征筛选】环节进行划分。")
        return

    results = st.session_state['model_results']
    model_names = list(results.keys())
    
    # 2. 冠军模型选择
    with st.container():
        st.markdown("### 1. 冠军模型遴选")
        col1, col2 = st.columns([1, 2])
        with col1:
            best_model_name = st.selectbox("选择要部署/解释的模型:", model_names)
            # 获取该模型在训练阶段的 CV 成绩
            cv_metrics = results[best_model_name]['metrics']
            st.caption(f"训练集 CV 成绩: {cv_metrics}")
        
        with col2:
            st.info("💡 提示：选择模型后，我们将首次解锁【测试集】进行终极验证。请确保这是你最终决定的模型，不要反复利用测试集调参，否则就是作弊（数据泄露）！")

    # 3. 终极考试 (The Final Exam)
    st.divider()
    st.markdown("### 2. 测试集终极验证 (Final Evaluation)")
    
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    # 确保特征列一致
    final_feats = st.session_state.get('final_features', list(st.session_state['X_train'].columns))
    X_test_final = X_test[final_feats]
    
    # 重新构建最佳模型（使用全量训练集重新 fit 也是一种策略，但为了严谨，我们通常复用 CV 中效果最好的参数重训，或者直接用最后一次的模型。
    # 这里为了演示简单，我们重新用最佳参数在全量 X_train 上训练一次）
    if st.button("🔓 解锁测试集并评估"):
        with st.spinner("正在全量训练集上重训模型，并预测测试集..."):
            # 获取数据
            X_train = st.session_state['X_train'][final_feats]
            y_train = st.session_state['y_train']
            
            # 获取工厂方法和参数
            from services.model_service import ModelService
            best_params = results[best_model_name]['best_params']
            
            # 判断任务类型
            task_type = 'classification' if y_train.nunique() < 20 else 'regression'
            
            # 重新训练
            final_model = ModelService.get_model_factory(task_type, best_model_name, best_params)
            final_model.fit(X_train, y_train)
            
            # 预测测试集
            y_pred_test = final_model.predict(X_test_final)
            
            # 保存到 session 以供后续解释使用
            st.session_state['final_model'] = final_model
            st.session_state['y_pred_test'] = y_pred_test
            
            # 展示成绩单
            col_res1, col_res2 = st.columns(2)
            if task_type == 'classification':
                acc = accuracy_score(y_test, y_pred_test)
                # 过拟合检测
                train_acc = cv_metrics['Accuracy']
                overfitting_gap = train_acc - acc
                
                with col_res1:
                    st.metric("测试集准确率 (Test Accuracy)", f"{acc:.4f}", delta=f"{-(overfitting_gap):.4f} vs Train")
                with col_res2:
                    if overfitting_gap > 0.05:
                        st.error(f"⚠️ 警告：检测到过拟合！测试集比训练集低 {(overfitting_gap*100):.2f}%。建议增加正则化或减少特征。")
                    else:
                        st.success("✅ 模型泛化能力良好 (无明显过拟合)。")
                
                st.text("详细分类报告:")
                st.text(classification_report(y_test, y_pred_test))
                
            else:
                r2 = r2_score(y_test, y_pred_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                with col_res1:
                    st.metric("测试集 R²", f"{r2:.4f}")
                with col_res2:
                    st.metric("测试集 RMSE", f"{rmse:.4f}")

    # 4. SHAP 解释 (XAI)
    if 'final_model' in st.session_state:
        final_model = st.session_state['final_model']
        X_train = st.session_state['X_train'][final_feats]
        
        st.divider()
        st.markdown("### 3. 模型可解释性 (XAI - SHAP)")
        st.caption("基于博弈论的 SHAP 值能告诉我们：每个特征对预测结果贡献了多少。")
        
        tab_global, tab_local = st.tabs(["🌍 全局特征决定力", "🔍 个体预测诊断"])
        
        # 为了速度，只计算测试集前 100 个样本的 SHAP
        # 实际生产中可以计算全部，但会慢
        X_explain = X_test_final.iloc[:100]
        
        # 懒加载：只有点击 Tab 才计算
        with tab_global:
            if st.button("⚡ 计算全局 SHAP (可能较慢)"):
                with st.spinner("正在计算 SHAP 值..."):
                    # 重新推断任务类型
                    y_train_temp = st.session_state.get('y_train')
                    task_type_shap = 'classification' if y_train_temp is not None and y_train_temp.nunique() < 20 else 'regression'
                    
                    explainer, shap_values = ExplanationService.get_shap_values(
                        final_model, X_train, X_explain, best_model_name, task_type=task_type_shap
                    )
                    
                    if shap_values is not None:
                        st.markdown("#### Beeswarm Plot (蜂群图)")
                        st.caption("点越红代表特征值越高，点越右代表对预测结果（如患病概率）的正向贡献越大。")
                        
                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, X_explain, show=False)
                        st.pyplot(fig)
                        
                        # 保存 explainer 到 session 供局部解释用
                        st.session_state['explainer'] = explainer
                        st.session_state['shap_values'] = shap_values
                        st.session_state['X_explain'] = X_explain

        with tab_local:
            if 'explainer' in st.session_state:
                st.markdown("#### 🔍 单样本微观诊断")
                
                # 选择样本
                sample_ids = X_explain.index.tolist()
                selected_idx = st.slider("选择样本索引 (0-99):", 0, len(X_explain)-1, 0)
                
                # 获取当前样本数据
                current_sample = X_explain.iloc[selected_idx]
                
                # --- Robust SHAP Value Extraction ---
                # 处理可能存在的格式问题 (List 或 3D Array)
                raw_shap = st.session_state['shap_values']
                final_shap_vals = raw_shap
                
                # 1. 如果是 list (多分类/二分类旧版API)，取 Class 1
                if isinstance(raw_shap, list):
                    final_shap_vals = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
                
                # 2. 如果是 3D array (samples, features, classes)，取 Class 1
                elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
                    # 假设最后一维是 classes
                    final_shap_vals = raw_shap[:, :, 1] if raw_shap.shape[2] > 1 else raw_shap[:, :, 0]
                
                # 确保现在是 2D array (samples, features)
                # 获取当前样本的 SHAP 值 (1D array)
                current_shap_values = final_shap_vals[selected_idx]
                # ------------------------------------

                # 1. 基础信息展示
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown("**原始特征值**")
                    st.dataframe(current_sample, height=300)
                with c2:
                    st.markdown("**为何做出此预测？ (Waterfall Plot)**")
                    try:
                        explainer = st.session_state['explainer']
                        
                        # 确保 base_value 是标量
                        base_val = explainer.expected_value
                        if isinstance(base_val, (list, np.ndarray)):
                             if len(base_val) > 1: base_val = base_val[1] # 默认取正类
                             else: base_val = base_val[0]
                        
                        # 构造 Explanation 对象
                        explanation = shap.Explanation(
                            values=current_shap_values, # 使用清洗后的 1D 数据
                            base_values=base_val,
                            data=current_sample.values,
                            feature_names=X_explain.columns
                        )
                        
                        fig_local, ax_local = plt.subplots(figsize=(8, 6))
                        shap.plots.waterfall(explanation, show=False, max_display=10)
                        st.pyplot(fig_local)
                    except Exception as e:
                        st.error(f"绘图失败: {str(e)}")
                
                # 2. What-If 分析
                st.divider()
                st.subheader("🎲 What-If 假设分析 (模拟实验)")
                st.caption("如果改变某些关键特征的值，预测结果会如何变化？")
                
                # 找出 Top 5 重要特征供修改
                # 这里的 current_shap_values 已经是 1D array，argsort 也是 1D
                top_indices = np.argsort(np.abs(current_shap_values))[::-1][:5]
                top_features = X_explain.columns[top_indices]
                
                col_inputs = st.columns(5)
                new_values = {}
                
                for i, feat in enumerate(top_features):
                    val = current_sample[feat]
                    with col_inputs[i]:
                        # 尝试推断步长
                        step = 1.0 if isinstance(val, (int, np.integer)) else 0.1
                        new_val = st.number_input(f"{feat}", value=float(val), step=step, key=f"whatif_{feat}")
                        new_values[feat] = new_val
                
                if st.button("🔮 重新预测 (Simulate)"):
                    # 构造新样本
                    simulated_sample = current_sample.copy()
                    for f, v in new_values.items():
                        simulated_sample[f] = v
                    
                    # 预测
                    model = st.session_state['final_model']
                    # 需要转为 DataFrame 并保持列顺序
                    input_df = pd.DataFrame([simulated_sample], columns=X_explain.columns)
                    
                    pred = model.predict(input_df)[0]
                    prob = None
                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(input_df)[0][1] # 正类概率
                        
                    # 展示结果对比
                    st.markdown("#### 模拟结果对比")
                    res_c1, res_c2 = st.columns(2)
                    with res_c1:
                        # 原始预测
                        orig_prob = model.predict_proba(pd.DataFrame([current_sample], columns=X_explain.columns))[0][1] if hasattr(model, "predict_proba") else 0
                        st.metric("原始概率 (Original)", f"{orig_prob:.4f}")
                        
                    with res_c2:
                        st.metric("模拟后概率 (Simulated)", f"{prob:.4f}", delta=f"{prob - orig_prob:.4f}")
            else:
                st.info("请先在【🌍 全局特征决定力】Tab 中点击计算。")

    # 5. 部署交付
    if 'final_model' in st.session_state:
        st.divider()
        st.markdown("### 4. 模型交付 (Deployment)")
        
        # 保存模型
        filename = "final_model.pkl"
        joblib.dump(st.session_state['final_model'], filename)
        
        with open(filename, "rb") as f:
            st.download_button(
                label="📦 下载模型文件 (.pkl)",
                data=f,
                file_name="ai_insight_model.pkl",
                mime="application/octet-stream"
            )
            
        st.markdown("#### 🚀 如何在 Python 中使用此模型？")
        code_snippet = f"""
import joblib
import pandas as pd

# 1. 加载模型
model = joblib.load('ai_insight_model.pkl')

# 2. 准备新数据 (确保特征顺序一致)
new_data = pd.DataFrame({{
    {', '.join([f"'{c}': [value]" for c in final_feats[:3]])}, ...
}})

# 3. 预测
prediction = model.predict(new_data)
print(f"预测结果: {{prediction}}")
        """
        st.code(code_snippet, language='python')