# Get all relevant libraries
import pandas as PANDAS
import numpy as NUMPY7
import matplotlib.pyplot as plottt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.cm as cm
import streamlit as st

st.title("Dating Profile Compatibility Model")

# Load the raw data and normalize it first
file_path = 'dating.xls'
df_raw = PANDAS.read_excel(file_path)

# Scale the data to have patterns seen more clearly
scaler = MinMaxScaler()
cols_to_scale = ['MILES', '% TIME GAMING', 'L OF ICE CREAM/WEEK']
df_normalized = df_raw.copy()
df_normalized[cols_to_scale] = scaler.fit_transform(df_raw[cols_to_scale])

# Get the normalized data into the code
df = df_normalized

# Isolate the data into their categories
X = df.iloc[:,0:3]
y = df.iloc[:, 3]

# Split into groups to be trained on
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Get the Support Vector Machine initialized
svm_model = SVC(kernel='rbf', C=1.0)

# The Support Vector Machine is trained
svm_model.fit(X_train, y_train)

# Support Vector Machine makes predictions
predictions = svm_model.predict(X_test)

# Get model evaluations
accuracy = accuracy_score(y_test, predictions)
st.write(f"### Accuracy: {accuracy * 100:.2f}%")
st.text("Detailed Breakdown:")
st.text(classification_report(y_test, predictions))

# Obtain three-dimensional visualizations
fig = plottt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

categories = df.iloc[:, 3].unique()
colors = cm.get_cmap('viridis', len(categories))

for i, category in enumerate(categories):
    # Note the 4-space indent here to prevent Python errors
    subset = df[df.iloc[:,3] == category]
    ax.scatter(subset.iloc[:,0], subset.iloc[:,1], subset.iloc[:,2],
               color=colors(i), label=str(category),
               alpha=0.7, s=40)

ax.set_xlabel('Frequent Flyer Miles')
ax.set_ylabel('Hours of Video Game Played')
ax.set_zlabel('Consumption of Ice Cream weekly (Liters)')
ax.set_title('Compatability of Dating Profiles (Values Scaled) - [SVM]')
ax.legend()

st.pyplot(fig)

# Generate two-dimensional graphs
# Get colouring sorted
le = LabelEncoder()
y_encoded = le.fit_transform(y)
target_names = le.classes_

# Canvas made
fig2, axes = plottt.subplots(1, 3, figsize=(18,5))
feature_names = ['Frequent Flyer Miles', 'Video Game Hours', 'Ice Cream (Liters)']
feature_pairs = [(0, 1), (0, 2), (1, 2)]

# Convert X to numpy array so column slicing works in the loop
X_array = X.to_numpy()

# Pairs get looped and graph drawn
for ax, (feat1, feat2) in zip(axes, feature_pairs):
    
    # Get just the two features for the 2D plot
    X_pair = X_array[:, [feat1, feat2]]

    # Train a mini SVM to draw the boundary lines
    clf = SVC(kernel='rbf', C=1.0)
    clf.fit(X_pair, y_encoded)

    # Draw colored backgrounds
    DecisionBoundaryDisplay.from_estimator(
        clf, X_pair, response_method="predict",
        cmap=plottt.cm.coolwarm, alpha=0.6, ax=ax, 
        xlabel=feature_names[feat1], ylabel=feature_names[feat2]
    )

    # Draw actual data points
    scatter = ax.scatter(X_pair[:,0], X_pair[:,1], c=y_encoded, cmap=plottt.cm.coolwarm, edgecolors='k', s=25)
    ax.set_title(f"{feature_names[feat1]} vs {feature_names[feat2]} (Scaled)")

# Add a master legend to the top
handles, _ = scatter.legend_elements()
fig2.legend(handles, target_names, loc='upper center', ncol=3, title="Compatibility Category")

# Save the 2D image and finally show everything
plottt.tight_layout(rect=[0, 0, 1, 0.90])
st.pyplot(fig2)

with st.expander("What am I looking at?"):
    st.write("""
    **Support Vector Machine (SVM):** 
    This machine learning model helps classify dating profiles by showing decision boundaries between user filters. 
    A user wants help in understanding what characteristics make for a perfect match on a dating app.
    The data used to make this folder is within the GitHub folder and values have been normalized to fit a 0 to 1 scale called Minimum-Maximum Scaler.

    **The filters observed**
    - Frequent Flyer Miles (Amount Travelled)
    
    -The amount of hours spent gaming
    
    -The amount of Ice Cream litres consumed
    
    **The Scatter Plots:** 
    1. The **3D Plot** shows how often a person travels, how long is a game played and ice creaam consumed interacts in a three dimensional space.

    
    2. The **2D Plots** shows the real lines or decision boundaries the SVM drew to seperate the filters in sets of two to accurately show the relationships between them.
    
    **FINAL RESULTS**
    Ice Cream has no effect on what makes a perfect match for the user.
    The amount travlled usually makes for a perfect match but there must be a balancing act where time spent gaming is mid range in terms of value.
    
    
    """)












