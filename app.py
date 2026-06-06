import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Page configuration
st.set_page_config(
    page_title="Order Return & Fraud Detection Hub",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS) for a premium dark/glassmorphism feel
st.markdown("""
<style>
    .main {
        background-color: #0f111a;
        color: #e6e6e6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a1d29;
        border-radius: 8px;
        color: #8c92ac;
        font-weight: 600;
        font-size: 16px;
        padding: 0px 24px;
        border: 1px solid #2d3142;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        border-color: #4f46e5;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: white !important;
        border-color: #4f46e5 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #a0aec0;
    }
    .metric-card {
        background-color: #161925;
        border: 1px solid #2d3142;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .risk-high {
        color: #ef4444;
        font-weight: bold;
    }
    .risk-low {
        color: #10b981;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load artifacts
@st.cache_resource
def load_models_and_metadata():
    models_dir = "models"
    
    # Check if files exist
    required_files = [
        "preprocessor.joblib",
        "feature_names.joblib",
        "evaluation_results.joblib"
    ]
    for f in required_files:
        if not os.path.exists(os.path.join(models_dir, f)):
            return None
            
    try:
        preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
        feature_names = joblib.load(os.path.join(models_dir, "feature_names.joblib"))
        eval_results = joblib.load(os.path.join(models_dir, "evaluation_results.joblib"))
        
        # Load models
        models = {
            'fraud': {
                'lr_no_smote': joblib.load(os.path.join(models_dir, "fraud_lr_no_smote.joblib")),
                'rf_no_smote': joblib.load(os.path.join(models_dir, "fraud_rf_no_smote.joblib")),
                'lr_smote': joblib.load(os.path.join(models_dir, "fraud_lr_smote.joblib")),
                'rf_smote': joblib.load(os.path.join(models_dir, "fraud_rf_smote.joblib")),
                'isolation_forest': joblib.load(os.path.join(models_dir, "fraud_isolation_forest.joblib")),
                'one_class_svm': joblib.load(os.path.join(models_dir, "fraud_one_class_svm.joblib")),
            },
            'returned': {
                'lr_no_smote': joblib.load(os.path.join(models_dir, "returned_lr_no_smote.joblib")),
                'rf_no_smote': joblib.load(os.path.join(models_dir, "returned_rf_no_smote.joblib")),
                'lr_smote': joblib.load(os.path.join(models_dir, "returned_lr_smote.joblib")),
                'rf_smote': joblib.load(os.path.join(models_dir, "returned_rf_smote.joblib")),
                'isolation_forest': joblib.load(os.path.join(models_dir, "returned_isolation_forest.joblib")),
                'one_class_svm': joblib.load(os.path.join(models_dir, "returned_one_class_svm.joblib")),
            }
        }
        return preprocessor, feature_names, eval_results, models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Load models
loaded_data = load_models_and_metadata()

# Header Section
st.title("🛡️ Return Prediction & Fraud Detection Hub")
st.markdown("""
Predict whether placed orders are likely to be returned or flagged as fraudulent using Machine Learning techniques.
Compare traditional supervised classifiers (with and without SMOTE resampling) side-by-side with unsupervised anomaly detection models (Isolation Forest, One-Class SVM).
""")

if loaded_data is None:
    st.warning("⚠️ Models and evaluation data were not found. Please train the models by running the training pipeline first.")
    
    # Train button for convenience if running locally
    if st.button("🚀 Train Models Now (This will run src/train.py)"):
        with st.spinner("Training models... Please wait (this samples ~20K entries and might take 1-2 minutes)..."):
            try:
                import subprocess
                result = subprocess.run(["python", "src/train.py"], capture_output=True, text=True)
                st.code(result.stdout)
                if result.returncode == 0:
                    st.success("Training completed successfully! Refreshing app...")
                    st.rerun()
                else:
                    st.error(f"Training failed:\n{result.stderr}")
            except Exception as ex:
                st.error(f"Failed to trigger script: {ex}")
    st.stop()

# Unpack
preprocessor, feature_names, eval_results, models = loaded_data

# Set up sidebar for real-time inference inputs
st.sidebar.header("🕹️ Simulation Panel")
st.sidebar.markdown("Modify order features to test real-time predictions:")

# Inputs mapping
# num_cols = ['TransactionAmt', 'dist1', 'C1', 'C2', 'C11', 'C13', 'D1', 'D2']
# cat_cols = ['ProductCD', 'card4', 'card6', 'DeviceType']
st.sidebar.subheader("Numeric Features")
trans_amt = st.sidebar.slider("Transaction Amount ($)", min_value=1.0, max_value=2500.0, value=150.0, step=0.1)
dist = st.sidebar.slider("Distance Feature (dist1)", min_value=0.0, max_value=1000.0, value=25.0)
c1 = st.sidebar.number_input("C1 Count (email count proxy)", min_value=0, max_value=500, value=2)
c2 = st.sidebar.number_input("C2 Count", min_value=0, max_value=500, value=2)
c11 = st.sidebar.number_input("C11 Count", min_value=0, max_value=500, value=1)
c13 = st.sidebar.number_input("C13 Count", min_value=0, max_value=500, value=1)
d1 = st.sidebar.slider("D1 (Days since registration)", min_value=0.0, max_value=600.0, value=15.0)
d2 = st.sidebar.slider("D2 (Days since last activity)", min_value=0.0, max_value=600.0, value=10.0)

st.sidebar.subheader("Categorical Features")
product_cd = st.sidebar.selectbox("Product Code (ProductCD)", options=['W', 'C', 'H', 'R', 'S'], index=0)
card4 = st.sidebar.selectbox("Card Brand (card4)", options=['visa', 'mastercard', 'american express', 'discover'], index=0)
card6 = st.sidebar.selectbox("Card Category (card6)", options=['debit', 'credit'], index=0)
device_type = st.sidebar.selectbox("Device Type", options=['desktop', 'mobile', 'missing'], index=0)

# Create input DataFrame
input_dict = {
    'TransactionAmt': [trans_amt],
    'dist1': [dist],
    'C1': [c1],
    'C2': [c2],
    'C11': [c11],
    'C13': [c13],
    'D1': [d1],
    'D2': [d2],
    'ProductCD': [product_cd],
    'card4': [card4],
    'card6': [card6],
    'DeviceType': [device_type]
}
df_input = pd.DataFrame(input_dict)

# Tabs
tab1, tab2 = st.tabs(["📊 Performance Dashboard", "🔮 Live Inference Engine"])

with tab1:
    st.header("📈 Model Evaluation & Method Comparison")
    st.markdown("""
    This panel displays the performance of the trained models on the test set. 
    Compare how classifiers perform with/without class balancing (**SMOTE**), and how they measure up against unsupervised **Anomaly Detectors**.
    """)
    
    target_tab = st.radio("Select Target to Inspect:", ["Fraud Detection (isFraud)", "Return Prediction (isReturned)"], horizontal=True)
    target_key = 'fraud' if "Fraud" in target_tab else 'returned'
    
    # Fetch results for this target
    target_results = eval_results['results'][target_key]
    
    # 1. Metrics Comparison Table
    st.subheader("📋 Core Evaluation Metrics")
    
    model_name_map = {
        'lr_no_smote': 'Logistic Regression (No SMOTE)',
        'rf_no_smote': 'Random Forest (No SMOTE)',
        'lr_smote': 'Logistic Regression + SMOTE',
        'rf_smote': 'Random Forest + SMOTE',
        'isolation_forest': 'Isolation Forest (Anomaly Det.)',
        'one_class_svm': 'One-Class SVM (Anomaly Det.)'
    }
    
    metrics_list = []
    for m_id, eval_pkg in target_results.items():
        report = eval_pkg['report']
        roc_auc = eval_pkg['roc_auc']
        
        # We focus on performance on the positive class '1' (Fraud/Return)
        precision_pos = report['1']['precision']
        recall_pos = report['1']['recall']
        f1_pos = report['1']['f1-score']
        accuracy = report['accuracy']
        
        metrics_list.append({
            "Model Name": model_name_map[m_id],
            "Accuracy": f"{accuracy*100:.2f}%",
            "Precision (Class 1)": f"{precision_pos*100:.2f}%",
            "Recall / Sensitivity (Class 1)": f"{recall_pos*100:.2f}%",
            "F1-Score (Class 1)": f"{f1_pos*100:.2f}%",
            "ROC-AUC": f"{roc_auc:.4f}"
        })
        
    df_metrics = pd.DataFrame(metrics_list)
    st.dataframe(df_metrics, use_container_width=True)
    
    # 2. Key Insights
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        > [!NOTE]
        > **SMOTE Impact**: Classifiers trained with SMOTE show a significant increase in **Recall** (getting fewer False Negatives) but a drop in **Precision** (generating more False Positives). This tradeoff is common when costs associated with missing an anomaly are high.
        """)
    with col2:
        st.markdown(f"""
        > [!TIP]
        > **Anomaly Detection models** (Isolation Forest, One-Class SVM) run in an unsupervised or semi-supervised fashion. They don't explicitly rely on labeled anomalous training data, making them excellent at finding "unknown unknowns" or zero-day anomalies, although they often yield lower overall precision than supervised classifiers.
        """)
        
    # 3. Plots: ROC Curves and Confusion Matrices
    st.subheader("📈 Visualization of Performance")
    
    col_plot1, col_plot2 = st.columns([2, 1])
    
    y_test_true = eval_results['y_test_fraud'] if target_key == 'fraud' else eval_results['y_test_returned']
    
    with col_plot1:
        # Plot ROC Curves
        fig_roc, ax_roc = plt.subplots(figsize=(10, 6.5))
        fig_roc.patch.set_facecolor('#0f111a')
        ax_roc.set_facecolor('#1a1d29')
        
        colors = {
            'lr_no_smote': '#ef4444',
            'rf_no_smote': '#f59e0b',
            'lr_smote': '#10b981',
            'rf_smote': '#3b82f6',
            'isolation_forest': '#8b5cf6',
            'one_class_svm': '#ec4899'
        }
        
        for m_id, eval_pkg in target_results.items():
            probs = eval_pkg['y_prob']
            fpr, tpr, _ = roc_curve(y_test_true, probs)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color=colors[m_id], lw=2, label=f"{model_name_map[m_id]} (AUC = {roc_auc:.3f})")
            
        ax_roc.plot([0, 1], [0, 1], color='#4b5563', lw=1.5, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate', color='#a0aec0', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate', color='#a0aec0', fontsize=12)
        ax_roc.set_title(f'ROC Curves Comparison ({target_tab})', color='#ffffff', fontsize=14, pad=15)
        
        # Style legend and ticks
        legend = ax_roc.legend(loc="lower right", facecolor='#161925', edgecolor='#2d3142')
        for text in legend.get_texts():
            text.set_color('#e6e6e6')
        ax_roc.tick_params(colors='#a0aec0')
        ax_roc.spines['bottom'].set_color('#2d3142')
        ax_roc.spines['top'].set_color('#2d3142')
        ax_roc.spines['left'].set_color('#2d3142')
        ax_roc.spines['right'].set_color('#2d3142')
        
        plt.tight_layout()
        st.pyplot(fig_roc)
        
    with col_plot2:
        # Show Confusion Matrix for selected model
        sel_model_id = st.selectbox("Select Model for Confusion Matrix:", options=list(model_name_map.keys()), format_func=lambda x: model_name_map[x], key="cm_select")
        
        cm = target_results[sel_model_id]['cm']
        
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 5.5))
        fig_cm.patch.set_facecolor('#0f111a')
        ax_cm.set_facecolor('#1a1d29')
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                    annot_kws={"size": 14, "weight": "bold"},
                    xticklabels=['Normal (0)', 'Anomaly (1)'],
                    yticklabels=['Normal (0)', 'Anomaly (1)'])
        
        ax_cm.set_xlabel('Predicted Label', color='#a0aec0', fontsize=12, labelpad=10)
        ax_cm.set_ylabel('True Label', color='#a0aec0', fontsize=12, labelpad=10)
        ax_cm.set_title(f'Confusion Matrix\n{model_name_map[sel_model_id]}', color='#ffffff', fontsize=13, pad=15)
        ax_cm.tick_params(colors='#a0aec0')
        
        plt.tight_layout()
        st.pyplot(fig_cm)

with tab2:
    st.header("🔮 Real-Time Inference Panel")
    st.markdown("""
    Submit a simulated transaction using the left sidebar. The system will pre-process the values and run predictions across all models simultaneously.
    """)
    
    # Run prediction
    try:
        # Preprocess input data
        X_input_prep = preprocessor.transform(df_input)
    except Exception as preprocess_err:
        st.error(f"Preprocessing error: {preprocess_err}")
        st.stop()
        
    # Predictions for Fraud and Return
    pred_data = {}
    for t_name in ['fraud', 'returned']:
        pred_data[t_name] = {}
        for m_id, model_obj in models[t_name].items():
            if 'isolation_forest' in m_id or 'one_class_svm' in m_id:
                # Anomaly detectors
                raw_pred = model_obj.predict(X_input_prep)[0]
                pred = 1 if raw_pred == -1 else 0
                
                # Probability proxy
                decision_val = model_obj.decision_function(X_input_prep)[0]
                # Map decision val to pseudo-probability (negative values are anomalous)
                # Lower values are anomalous
                prob = 1.0 / (1.0 + np.exp(decision_val)) # Sigmoid mapping
            else:
                # Classifiers
                pred = model_obj.predict(X_input_prep)[0]
                prob = model_obj.predict_proba(X_input_prep)[0][1]
                
            pred_data[t_name][m_id] = {
                'prediction': pred,
                'probability': prob
            }
            
    # Display prediction results
    col_out1, col_out2 = st.columns(2)
    
    # Card templates
    def display_results_column(col_obj, target_label, predictions):
        with col_obj:
            st.subheader(f"🔍 {target_label} Predictions")
            
            # Aggregate risk score based on predictions (unweighted vote)
            votes = [val['prediction'] for val in predictions.values()]
            risk_pct = sum(votes) / len(votes) * 100
            
            risk_class = "risk-high" if risk_pct >= 50 else "risk-low"
            risk_text = "HIGH RISK" if risk_pct >= 50 else "LOW RISK"
            
            st.markdown(f"""
            <div style="background-color: #1a1d29; border: 1px solid #2d3142; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="margin: 0; color: #a0aec0;">Consensus Risk Level</h4>
                <p style="margin: 10px 0; font-size: 32px; font-weight: 800;" class="{risk_class}">
                    {risk_text} ({risk_pct:.0f}%)
                </p>
                <p style="margin: 0; font-size: 13px; color: #a0aec0;">
                    {sum(votes)} out of {len(votes)} models flagged this transaction.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show individual model outputs
            st.markdown("##### Model Breakdown")
            for m_id, model_name in model_name_map.items():
                p_info = predictions[m_id]
                pred = p_info['prediction']
                prob = p_info['probability']
                
                icon = "🚨 Flagged" if pred == 1 else "✅ Clean"
                color = "#ef4444" if pred == 1 else "#10b981"
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; background-color: #161925; border: 1px solid #2d3142; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
                    <div>
                        <strong style="color: #ffffff;">{model_name}</strong><br/>
                        <span style="font-size: 12px; color: #a0aec0;">Anomaly Score/Prob: {prob:.2%}</span>
                    </div>
                    <div style="background-color: {color}22; color: {color}; border: 1px solid {color}; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 14px;">
                        {icon}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    display_results_column(col_out1, "Fraud Detection (isFraud)", pred_data['fraud'])
    display_results_column(col_out2, "Return Prediction (isReturned)", pred_data['returned'])
    
    # Feature Input Summary
    st.subheader("📝 Preprocessed Feature Matrix representation")
    st.markdown("This is how your sidebar entries map to numerical vectors after fitting the pipeline:")
    st.write(pd.DataFrame(X_input_prep, columns=feature_names))
