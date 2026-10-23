import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import DecisionBoundaryDisplay
import streamlit as st

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Dating Profile Matcher", layout="wide", page_icon="💖")

# 2. Cache the data loading so the app runs much faster on interactions
@st.cache_data
def load_data():
    return pd.read_excel('dating.xls')

# --- SIDEBAR: Make the app interactive ---
st.sidebar.header("⚙️ Model Settings")
c_param = st.sidebar.slider("SVM C-parameter (Regularization)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
kernel_type = st.sidebar.selectbox("SVM Kernel", ["rbf", "linear", "poly", "sigmoid"])
test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5) / 100

st.title("💖 Dating Profile Compatibility Model")
st.markdown("Use Support Vector Machines (SVM) to classify dating profiles and find the perfect match based on lifestyle habits.")

# Load and prepare data
df_raw = load_data()
scaler = MinMaxScaler()
cols_to_scale = ['MILES', '% TIME GAMING', 'L OF ICE CREAM/WEEK']
df = df_raw.copy()
df[cols_to_scale] = scaler.fit_transform(df_raw[cols_to_scale])

# Isolate features (X) and target (y)
X = df.iloc[:, 0:3]
y = df.iloc[:, 3]

# Split the data based on sidebar slider
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Train the SVM model with dynamic sidebar parameters
svm_model = SVC(kernel=kernel_type, C=c_param)
svm_model.fit(X_train, y_train)
predictions = svm_model.predict(X_test)

# --- RESULTS SECTION ---
st.markdown("### 📊 Model Performance")
col1, col2 = st.columns([1, 2])

with col1:
    accuracy = accuracy_score(y_test, predictions)
    st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")

with col2:
    with st.expander("Show Detailed Classification Report"):
        st.text(classification_report(y_test, predictions))

st.divider()

# --- VISUALIZATIONS SECTION (Using Tabs for better UI) ---
tab1, tab2, tab3 = st.tabs(["🌐 3D Visualization", "📐 2D Decision Boundaries", "📁 Raw Data Snapshot"])

with tab1:
    st.subheader("3D Scatter Plot of Features")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    categories = df.iloc[:, 3].unique()
    # FIXED: Replaced get_cmap with the updated Matplotlib 3.9+ standard
    colors = mpl.colormaps['viridis'].resampled(len(categories))

    for i, category in enumerate(categories):
        subset = df[df.iloc[:, 3] == category]
        ax.scatter(subset.iloc[:, 0], subset.iloc[:, 1], subset.iloc[:, 2],
                   color=colors(i), label=str(category), alpha=0.8, s=50, edgecolors='w')

    ax.set_xlabel('Frequent Flyer Miles')
    ax.set_ylabel('Gaming (Hours)')
    ax.set_zlabel('Ice Cream (Liters)')
    ax.set_title(f'3D Profile Compatibility (Kernel: {kernel_type.upper()})')
    ax.legend(title="Category")
    st.pyplot(fig)

with tab2:
    st.subheader("2D Decision Boundaries")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    target_names = le.classes_

    fig2, axes = plt.subplots(1, 3, figsize=(20, 5))
    feature_names = ['Frequent Flyer Miles', 'Video Game Hours', 'Ice Cream (Liters)']
    feature_pairs = [(0, 1), (0, 2), (1, 2)]
    X_array = X.to_numpy()

    for ax, (feat1, feat2) in zip(axes, feature_pairs):
        X_pair = X_array[:, [feat1, feat2]]
        
        # Train a mini SVM to draw the boundary lines for these specific 2 features
        clf = SVC(kernel=kernel_type, C=c_param)
        clf.fit(X_pair, y_encoded)

        # Draw colored backgrounds
        DecisionBoundaryDisplay.from_estimator(
            clf, X_pair, response_method="predict",
            cmap=plt.cm.coolwarm, alpha=0.5, ax=ax, 
            xlabel=feature_names[feat1], ylabel=feature_names[feat2]
        )

        # Draw actual data points
        scatter = ax.scatter(X_pair[:, 0], X_pair[:, 1], c=y_encoded, 
                             cmap=plt.cm.coolwarm, edgecolors='k', s=35)
        ax.set_title(f"{feature_names[feat1]} vs \n{feature_names[feat2]}")

    # Add a master legend
    handles, _ = scatter.legend_elements()
    fig2.legend(handles, target_names, loc='upper center', ncol=3, title="Compatibility", bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    st.pyplot(fig2)

with tab3:
    st.subheader("Dataset Snapshot")
    st.dataframe(df.head(50), use_container_width=True)

# --- EXPLANATION SECTION ---
with st.expander("📖 What am I looking at? (Click to expand)"):
    st.markdown("""
    **Support Vector Machine (SVM):**  
    This machine learning model helps classify dating profiles by finding the best *decision boundaries* to separate different types of matches. The data has been scaled down to fit nicely between 0 and 1 (Min-Max Scaler).

    **The Filters Observed:**
    1. **Frequent Flyer Miles:** The amount of traveling they do.
    2. **Time Gaming:** The amount of time spent playing video games.
    3. **Ice Cream:** Liters of ice cream consumed weekly.

    **The Visuals:**  
    * **3D Plot:** Shows the actual, real-world relationship between all three variables at the same time.
    * **2D Plots:** Show the exact lines (decision boundaries) the SVM algorithm drew to separate the categories when comparing two variables at a time. Play with the **Kernel** and **C-Parameter** in the sidebar to watch how the model changes its math!

    **Results Summary:**
    * Ice Cream generally has very little effect on making a perfect match.
    * Frequent flying combined with a moderate, balanced amount of gaming is usually the sweet spot for a match!
    """)
