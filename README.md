**Compatibility of Dating Detection: Support Vector Machine Model**

**Overview**
A customer has many preferences on when to date certain men and how much contact they keep with them. Dating sites are the median she uses but wants to know what makes a perfect match for her. The Machine Learning approach of a Support Vector Machine was used as the algorithm to make a system based on her past preference history.

**Scenario**
The fully automation of dating preferences into three distinct categories namely, large doses, small doses and none at all. The features analyzed were amount of ice cream liters consumed in a week, frequent flyer miles and the percentage time spent on gaming.

**Tech Stack Used**
IDE: PyCharm (with Python 3.13)

Programming Language: Python

Libraries: Pandas and Scikit-learn

Data Normalization: Minimum-Maximum scaling was used to normalize data so frequent flyer miles don't unfairly skew model

Machine Learning Paradigm: Supervised Learning

Machine Learning Approach: Support Vector Machine with a Radial Basis Function kernel to handle non-linear data

**Visualizations**
There being three filters mean the Support Vector Machine (SVM) will be in three dimensions but to also display the hidden patterns easier three two-dimensional graphs will be shown comparing two filters at a time. Three dimensions can be difficult to identify patterns in that space.

Below is the PyCharm generated three-dimensional scatter plot

<img width="940" height="752" alt="Image" src="https://github.com/user-attachments/assets/00d52afd-9c73-4477-9248-db3108bf8d3d" />

Below is the three PyCharm generated two-dimensional plots

<img width="764" height="636" alt="Image" src="https://github.com/user-attachments/assets/80d9705e-5e0d-43e4-90da-f9cb05836daa" />


<img width="765" height="640" alt="Image" src="https://github.com/user-attachments/assets/106781ec-a8fd-4f91-a814-87111a032019" />


<img width="764" height="643" alt="Image" src="https://github.com/user-attachments/assets/819fa4bf-f807-428a-a5e1-6e17a5e361de" />



**Key Insights**
Automatic rejections: Extremely high video gaming hours and low frequent flyer miles 

Perfect Range: Very high frequent flyer miles with a combination of moderately ranged gaming habits 

Non-factor filter: The amount of ice cream consumed had no effect on the compatibility for a partner

