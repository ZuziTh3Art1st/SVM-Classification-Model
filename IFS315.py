import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
import streamlit as st
import joblib
import io

# 1. Page Setup & Configuration
# Let's make the app wide so our charts have plenty of room to breathe.
st.set_page_config(page_title="SVM Classification Model", layout="wide")

# 2. Data Loading & Caching
# We use Streamlit's cache decorator here. This ensures that the app doesn't re-read 
# the Excel or CSV file from scratch every time the user clicks a button or moves a slider.
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    else:
        # Fallback to our default dating dataset if the user hasn't uploaded anything.
        return pd.read_excel('dating.xls')

# 3. Sidebar: User Controls & File Upload
# This section builds the left-hand menu where users can upload data and tweak model math.
st.sidebar.header("1. Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your data (CSV or Excel)", type=["csv", "xls", "xlsx"])
st.sidebar.markdown("*Note: Ensure your data has 3 numerical feature columns followed by 1 category column.*")

st.sidebar.header("2. Model Settings")
# The C-parameter controls how strict the model is about making mistakes during training.
c_param = st.sidebar.slider("SVM Regularization (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
# The kernel dictates the mathematical shape the model uses to slice through the data.
kernel_type = st.sidebar.selectbox("SVM Kernel", ["rbf", "linear", "poly", "sigmoid"])
# How much data do we hold back to test the model's accuracy? (20% is standard).
test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5) / 100

st.sidebar.divider()
st.sidebar.header("3. System Actions")

# This button lets users completely flush the app's memory and start fresh.
if st.sidebar.button("Clear Cache & Reset App", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# 4. Main Dashboard Header
st.title("Interactive SVM Classification Model")
st.markdown("Upload your own data or use the default dataset to visualize Support Vector Machine decision boundaries in real-time.")
st.divider()

# We wrap all the main logic in a try-except block. That way, if a user uploads a completely 
# broken file, the app gracefully shows a friendly error instead of crashing.
try:
    # Load the data into a dataframe
    df_raw = load_data(uploaded_file)
    
    # Safety check: Do we actually have the 4 columns we need?
    if df_raw.shape[1] < 4:
        st.error("Uploaded dataset must have at least 4 columns (3 features + 1 target).")
        st.stop()

    # Dynamically grab the column names so this script works with ANY uploaded file.
    feature_cols = df_raw.columns[0:3].tolist()
    target_col = df_raw.columns[3]

    # 5. Data Preprocessing
    # Machine learning models struggle if one column has huge numbers (like 50,000 miles) 
    # and another has tiny numbers (like 2 liters). We scale everything nicely between 0 and 1.
    scaler = MinMaxScaler()
    df = df_raw.copy()
    df[feature_cols] = scaler.fit_transform(df_raw[feature_cols])

    # Isolate our clues (X) and the answer key (y)
    X = df.iloc[:, 0:3]
    y = df.iloc[:, 3]

    # Split the data based on the user's chosen test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # 6. Training the SVM Model
    # Here we actually build and train the brain of the app.
    svm_model = SVC(kernel=kernel_type, C=c_param)
    svm_model.fit(X_train, y_train)
    predictions = svm_model.predict(X_test)
    
    # 7. Exporting the Model (Sidebar feature)
    # We package both the model AND the scaler together. Without the scaler, 
    # the user wouldn't be able to properly process new data in the future.
    st.sidebar.header("4. Export Model")
    st.sidebar.markdown("Download the trained model to use in other Python environments.")
    
    export_package = {
        'model': svm_model,
        'scaler': scaler,
        'feature_names': feature_cols
    }
    
    # Save the package to an invisible memory buffer so Streamlit can offer it as a download
    buffer = io.BytesIO()
    joblib.dump(export_package, buffer)
    buffer.seek(0)
    
    st.sidebar.download_button(
        label="Download Model Package (.joblib)",
        data=buffer,
        file_name="svm_model_package.joblib",
        mime="application/octet-stream",
        use_container_width=True
    )

    # 8. Displaying Metrics & Confusion Matrix
    st.markdown("### Model Performance Overview")
    col_metric, col_matrix = st.columns([1, 2])

    with col_metric:
        accuracy = accuracy_score(y_test, predictions)
        st.metric(label="Testing Accuracy", value=f"{accuracy * 100:.2f}%")
        st.markdown("The **Confusion Matrix** (right) breaks down this accuracy score. The dark diagonal boxes represent correct predictions. Numbers outside the diagonal are mistakes the model made.")

    with col_matrix:
        # Map out exactly where the model succeeded and where it got confused
        cm = confusion_matrix(y_test, predictions, labels=svm_model.classes_)
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                    xticklabels=svm_model.classes_, 
                    yticklabels=svm_model.classes_, ax=ax_cm)
        ax_cm.set_ylabel('Actual Category', fontweight='bold')
        ax_cm.set_xlabel('Predicted Category', fontweight='bold')
        ax_cm.set_title('Confusion Matrix')
        
        st.pyplot(fig_cm)
        
        # Save the confusion matrix image to memory for downloading
        buf_cm = io.BytesIO()
        fig_cm.savefig(buf_cm, format="png", bbox_inches='tight')
        buf_cm.seek(0)
        st.download_button(
            label="Download Confusion Matrix (PNG)",
            data=buf_cm,
            file_name="confusion_matrix.png",
            mime="image/png"
        )

    st.divider()

    # 9. Tabs for Visualizations & Predictions
    # Grouping features into tabs keeps the dashboard from looking messy or overwhelming.
    tab1, tab2, tab3, tab4 = st.tabs([
        "3D Feature Space", 
        "2D Decision Boundaries", 
        "Predict New Data", 
        "Dataset & Correlation"
    ])

    # Tab 1: 3D Visualization
    with tab1:
        st.subheader("3D Scatter Plot")
        st.markdown("Visualize how your three features cluster in a three-dimensional space.")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Grab a nice color palette based on how many categories we actually have
        categories = df.iloc[:, 3].unique()
        colors = mpl.colormaps['viridis'].resampled(len(categories))

        for i, category in enumerate(categories):
            subset = df[df.iloc[:, 3] == category]
            ax.scatter(subset.iloc[:, 0], subset.iloc[:, 1], subset.iloc[:, 2],
                       color=colors(i), label=str(category), alpha=0.8, s=50, edgecolors='w')

        ax.set_xlabel(feature_cols[0])
        ax.set_ylabel(feature_cols[1])
        ax.set_zlabel(feature_cols[2])
        ax.set_title(f'Data Distribution (Scaled)')
        ax.legend(title=target_col)
        
        st.pyplot(fig)
        
        buf_3d = io.BytesIO()
        fig.savefig(buf_3d, format="png", bbox_inches='tight')
        buf_3d.seek(0)
        st.download_button(
            label="Download 3D Plot (PNG)",
            data=buf_3d,
            file_name="3d_scatter_plot.png",
            mime="image/png"
        )

    # Tab 2: 2D Decision Boundaries
    with tab2:
        st.subheader("Decision Boundaries by Feature Pair")
        st.markdown("These plots show exactly where the SVM algorithm drew lines to separate the categories.")
        
        # Turn our text categories into numbers so matplotlib can draw them
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        target_names = le.classes_

        fig2, axes = plt.subplots(1, 3, figsize=(20, 5))
        feature_pairs = [(0, 1), (0, 2), (1, 2)]
        X_array = X.to_numpy()

        for ax, (feat1, feat2) in zip(axes, feature_pairs):
            X_pair = X_array[:, [feat1, feat2]]
            
            # We train a tiny, temporary SVM here just for these two specific features 
            # so we can accurately draw the background boundary colors.
            clf = SVC(kernel=kernel_type, C=c_param)
            clf.fit(X_pair, y_encoded)

            DecisionBoundaryDisplay.from_estimator(
                clf, X_pair, response_method="predict",
                cmap=plt.cm.coolwarm, alpha=0.5, ax=ax, 
                xlabel=feature_cols[feat1], ylabel=feature_cols[feat2]
            )

            # Plop the actual data points over the colored background regions
            scatter = ax.scatter(X_pair[:, 0], X_pair[:, 1], c=y_encoded, 
                                 cmap=plt.cm.coolwarm, edgecolors='k', s=35)
            ax.set_title(f"{feature_cols[feat1]} vs \n{feature_cols[feat2]}")

        # Add one clean master legend at the top instead of cluttering each individual chart
        handles, _ = scatter.legend_elements()
        fig2.legend(handles, target_names, loc='upper center', ncol=3, title=target_col, bbox_to_anchor=(0.5, 1.05))
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        
        st.pyplot(fig2)
        
        buf_2d = io.BytesIO()
        fig2.savefig(buf_2d, format="png", bbox_inches='tight')
        buf_2d.seek(0)
        st.download_button(
            label="Download 2D Boundaries (PNG)",
            data=buf_2d,
            file_name="2d_decision_boundaries.png",
            mime="image/png"
        )

    # Tab 3: Predict New Data
    with tab3:
        st.subheader("Predict Categorization for New Data")
        st.markdown("Manually input new values to see how the trained model will classify them.")
        
        col_in1, col_in2, col_in3 = st.columns(3)
        
        # Default the input boxes to the mathematical mean of the raw data 
        val1 = col_in1.number_input(f"Enter {feature_cols[0]}:", value=float(df_raw[feature_cols[0]].mean()))
        val2 = col_in2.number_input(f"Enter {feature_cols[1]}:", value=float(df_raw[feature_cols[1]].mean()))
        val3 = col_in3.number_input(f"Enter {feature_cols[2]}:", value=float(df_raw[feature_cols[2]].mean()))
        
        if st.button("Generate Prediction", type="primary"):
            # Structure the user input, scale it, and ask the model for an answer
            input_df = pd.DataFrame([[val1, val2, val3]], columns=feature_cols)
            input_scaled = scaler.transform(input_df)
            pred = svm_model.predict(input_scaled)[0]
            st.success(f"**Prediction Result:** Based on your inputs, the model classifies this as: **{pred}**")

    # Tab 4: Raw Data & Heatmap
    with tab4:
        col_data, col_heat = st.columns([1, 1])
        
        with col_data:
            st.subheader("Dataset Snapshot (First 50 Rows)")
            st.dataframe(df_raw.head(50), use_container_width=True)
            
        with col_heat:
            st.subheader("Feature Correlation Heatmap")
            st.markdown("Values closer to 1 or -1 show a strong relationship, while values near 0 show little to no relationship.")
            
            # Check how closely our numerical features are tied to one another
            corr_matrix = df_raw[feature_cols].corr()
            
            fig_heat, ax_heat = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax_heat)
            
            st.pyplot(fig_heat)
            
            buf_heat = io.BytesIO()
            fig_heat.savefig(buf_heat, format="png", bbox_inches='tight')
            buf_heat.seek(0)
            st.download_button(
                label="Download Heatmap (PNG)",
                data=buf_heat,
                file_name="correlation_heatmap.png",
                mime="image/png"
            )

    # 10. Explanation Footer
    # A helpful drop-down guide so non-technical users understand what they are looking at.
    st.divider()
    with st.expander("How to read these charts (Click to expand)"):
        st.markdown("""
        **Support Vector Machine (SVM):**  
        This algorithm classifies data by finding the optimal *decision boundaries* (lines or planes) that separate different categories. 
        
        Before the data is visualized or trained, it is passed through a **Min-Max Scaler**. This shrinks all values down to a proportion between 0.0 and 1.0. This prevents columns with huge numbers (like miles flown) from completely overpowering columns with small numbers (like ice cream liters).

        * **Confusion Matrix:** Shows you exactly where the model succeeded and failed. The darker the diagonal boxes, the better the model is performing. 
        * **3D Plot:** Represents the relationship of all three features simultaneously.
        * **2D Plots:** Shows where the model mathematically divides the data when looking at just two features at a time.
        * **Correlation Heatmap:** Reveals hidden patterns in your raw data. If two variables are highly correlated, they move together (positive) or in opposite directions (negative).
        * **Hyperparameters:** Use the sidebar to change the **Kernel** (the mathematical shape of the boundary) and the **C-Parameter** (how hard the model tries to avoid misclassifying training points).
        """)

# Catch-all error handling so the user interface never fully breaks
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.info("Ensure your uploaded file is formatted with 3 numerical feature columns and 1 target category column.")
