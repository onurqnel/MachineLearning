import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ------------------------------------------------------------------
# 1. Data Loading & Feature Engineering
# ------------------------------------------------------------------

# 1A. Load the College dataset from a CSV file.
College = pd.read_csv('College.csv')

# 1B. Rename the first column to 'College' and use it as the index.
CollegeData = College.rename({'Unnamed: 0': 'College'}, axis=1).set_index('College')

# 1C. Create new features to enrich the dataset:
#     - AcceptanceRate: (Accept / Apps) * 100
#       This represents the percentage of applicants who were accepted.
#     - StudentCount: F.Undergrad + P.Undergrad + Enroll
#       This represents the total number of students at each institution 
CollegeData['AcceptanceRate'] = (CollegeData['Accept'] / CollegeData['Apps']) * 100
CollegeData['StudentCount'] = (CollegeData['F.Undergrad'] 
                               + CollegeData['P.Undergrad'] 
                               + CollegeData['Enroll'])

# 1D. Create an 'InstutionSize' column to categorize each college as 'Small', 
#     'Medium', or 'Large' based on the StudentCount feature.
def classify_size(row):
    if row['StudentCount'] > 5925:
        return 'Large'
    elif row['StudentCount'] < 1502:
        return 'Small'
    else:
        return 'Medium'
CollegeData['InstutionSize'] = CollegeData.apply(classify_size, axis=1)

# ------------------------------------------------------------------
# 2. Select and Scale Features for Clustering
# ------------------------------------------------------------------

# 2A. Choose the subset of features (columns) used for clustering.
features = ['AcceptanceRate', 'Top10perc', 'Outstate', 'Expend', 'PhD', 'S.F.Ratio', 'StudentCount']
X = CollegeData[features].copy()

# 2B. Scale (normalize) the features using StandardScaler.
#     StandardScaler transforms each feature to have mean=0 and std=1.
#     This ensures that all features contribute equally to the distance metric in K-Means 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------
# 3. Determine Optimal Number of Clusters (k) using Silhouette Score
# ------------------------------------------------------------------
# We will vary the number of clusters (k) from 2 to 9.
# For each k, we fit a K-Means model and compute the silhouette score.
# The k with the highest silhouette score is chosen as the optimal k.

best_k = None
best_score = -1

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    if score > best_score:
        best_score = score
        best_k = k

print(f"Best k by silhouette score: {best_k} with score={best_score:.4f}")

# ------------------------------------------------------------------
# 4. Fit Final K-Means Model
# ------------------------------------------------------------------
# Once we find the optimal k, we use it to fit a final K-Means model on the data.

kmeans_final = KMeans(n_clusters=best_k, random_state=42).fit(X_scaled)
CollegeData['Cluster'] = kmeans_final.labels_

# ------------------------------------------------------------------
# 5. Basic Cluster Analysis
# ------------------------------------------------------------------
# We can look at cluster-wise statistics to see how each cluster differs.

# 5A. Print the mean values of each clustering feature by cluster.
cluster_summary = CollegeData.groupby('Cluster')[features].mean()
print("\n=== Cluster Summary (Mean of Each Feature) ===")
print(cluster_summary)

# 5B. Check the composition of Institution Size within each cluster.
#     This shows the proportion of 'Small', 'Medium', and 'Large' colleges
#     in each cluster.
cluster_size_composition = CollegeData.groupby('Cluster')['InstutionSize'].value_counts(normalize=True)
print("\n=== Institution Size Composition by Cluster ===")
print(cluster_size_composition)

# 5C. Check the composition of Institution Type within each cluster.
#     This shows the proportion of 'Private' vs. 'Public' colleges in each cluster.
cluster_type_composition = CollegeData.groupby('Cluster')['Private'].value_counts(normalize=True)
print("\n=== Institution Type Composition by Cluster ===")
print(cluster_type_composition)

# 5D. Count the number of colleges in each cluster.
count_cluster_0 = (CollegeData['Cluster'] == 0).sum()
count_cluster_1 = (CollegeData['Cluster'] == 1).sum()

print(f"\nCount for cluster 0: {count_cluster_0}")
print(f"Count for cluster 1: {count_cluster_1}")



# ------------------------------------------------------------------
# 6. Visualizations with Descriptive Information
# ------------------------------------------------------------------

# 6A. PCA Plot
#     We use PCA (Principal Component Analysis) to reduce the scaled data 
#     to 2 principal components. Then we create a scatter plot of the 
#     transformed data colored by cluster labels.

pca = PCA(n_components=2)               # Initialize PCA to reduce data to 2 dimensions
X_pca = pca.fit_transform(X_scaled)     # Fit PCA to scaled data and transform

plt.figure(figsize=(8,6))               # Create a new figure (width=8, height=6)
scatter = sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=CollegeData['Cluster'],         # Color each point by its cluster label
    palette='Set2',
    alpha=0.7                           # Slight transparency so points don't overlap too much
)
plt.title('Clusters (PCA Projection)', fontsize=14)   # Plot title
plt.xlabel('PCA Component 1', fontsize=12)            # X-axis label
plt.ylabel('PCA Component 2', fontsize=12)            # Y-axis label
plt.legend(title='Cluster')                           # Add legend with a title
plt.tight_layout()                                    # Adjust figure padding
plt.show()

# 6B. Boxplots for Selected Features by Cluster
#     Boxplots help visualize the distribution of numerical features
#     across different clusters.

# 1) Out-of-State Tuition by Cluster
plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y='Outstate', data=CollegeData, palette='Set2')
plt.title('International Tuition Distribution by Cluster', fontsize=14)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('International Tuition (USD)', fontsize=12)
plt.tight_layout()
plt.show()

# 2) Instructional Expenditure per Student by Cluster
plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y='Expend', data=CollegeData, palette='Set2')
plt.title('Instructional Expenditure per Student by Cluster', fontsize=14)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Instructional Expenditure Per Student', fontsize=12)
plt.tight_layout()
plt.show()

# 3) Graduation Rate by Cluster
plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y='Grad.Rate', data=CollegeData, palette='Set2')
plt.title('Graduation Rate by Cluster', fontsize=14)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Graduation Rate', fontsize=12)
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

# 4) PhD level Instructor Rate by Cluster
plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y='PhD', data=CollegeData, palette='Set2')
plt.title('PhD level Instructor Rate by Cluster', fontsize=14)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('PhD level Instructor Rate', fontsize=12)
plt.tight_layout()
plt.show()

# 5) Student/Faculty Ratio by Cluster
plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y='S.F.Ratio', data=CollegeData, palette='Set2')
plt.title('Student/Faculty Ratio by Cluster', fontsize=14)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Student/Faculty Ratio', fontsize=12)
plt.tight_layout()
plt.show()

# Acceptence rate
CollegeData['AcceptenceRate'] = (CollegeData['Accept'] / CollegeData['Apps']) * 100
def classify_size(row):
    if row['StudentCount'] > 5925:
        return 'Large'
    elif row['StudentCount'] < 1502:
        return 'Small'
    else:
        return 'Medium'
CollegeData['InstitutionSize'] = CollegeData.apply(classify_size, axis=1)
Cluster1Data = CollegeData[CollegeData['Cluster'] == 1]
print(Cluster1Data.describe())
print(Cluster1Data)
sns.scatterplot(
    data=Cluster1Data, 
    x='Grad.Rate',         
    y='AcceptanceRate', 
    hue='Private'           
)
plt.title('Acceptance Rate vs. Graduation Rate Top %25 Institutions')
plt.xlabel('Graduation Rate (%)')
plt.ylabel('Acceptance Rate of Institution (%)')
plt.legend(title='Private')
plt.show()

# Scatter Plot 2: Acceptance Rate vs. S.F. Ratio with Hue by Institution Size
sns.scatterplot(
    data=Cluster1Data, 
    x='Grad.Rate',         
    y='AcceptanceRate', 
    hue='InstitutionSize' 
)
plt.title('Acceptance Rate vs. Graduation Rate Top %25 Institutions')
plt.xlabel('Graduation Rate (%)')
plt.ylabel('Acceptance Rate of Institution (%)')
plt.legend(title='Institution Size')
plt.show()

# ------------------------------------------------------------------
# End of Script
# ------------------------------------------------------------------
