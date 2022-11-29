import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("country-data.csv")
print(data.info())  # Taking a first look to our dataset.
# We see there are only one column which is not numeric
# That column represents country, which does not represent any real data,
# so we can just drop it during process instead of vectorizing it.
X = data.drop(["country"], axis=1)

# Calculating skewness for each column
print(X.skew(axis=0))
# First thing to notice is inflation column.
# We can infer that a very big part of the countries has lower inflation rates.
# Other three columns to notice are exports, income, and gdpp

# Calculating kurtosis for each column
print(X.kurt(axis=0))


# We can easily observe that inflation and exports are far away from being normally distributed
# We can infer that there is a big difference between high and low inflation rates among countries
# Same can be said for exports


def handle_na(X: pd.DataFrame):  # Handling missing values

    nan_cols = []
    X.apply(lambda col: nan_cols.append(col.name) if col.isna().sum() > col.count() else col)
    X.drop(nan_cols, axis=1, inplace=True)  # Getting rid of columns if they are unusably sparse

    imputer = SimpleImputer(strategy='mean')
    # Imputing mean of column for null values
    imputed_X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return imputed_X


X = handle_na(X)  # Now that we got rid of null values, let's take another look at our dataset.

sns.histplot(data=X, x='health')
plt.show()  # Trying to understand most from graphs

sns.jointplot(data=X, x='income', y='life_expec', kind='kde', fill=True)
plt.show()  # Very useful graph. Clearly seeing the correlation.

sns.boxplot(data=X, orient='h')
plt.show()  # We can clearly see that we have some outliers to handle


# This graph also shows that we should apply normalization to our dataset


def handle_outliers(X: pd.DataFrame):  # Handling outliers using IQR method

    q1 = X.quantile(0.25, numeric_only=True)
    q3 = X.quantile(0.75, numeric_only=True)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Capping values to upper/lower bounds
    capped_X = X.apply(
        lambda col: [(lower_bound[col.name] if i < lower_bound[col.name] else i) for i in col])
    capped_X = capped_X.apply(
        lambda col: [(upper_bound[col.name] if i > upper_bound[col.name] else i) for i in col])

    return capped_X


X = handle_outliers(X)

# Plotting the correlation matrix with the help of heatmap.
corr_matrix = X.corr()
sns.heatmap(corr_matrix, cmap="YlGnBu")
plt.show()  # It seems that distribution is as it should be.


# From previous boxplot we have seen we should scale our data.
def scale(X: pd.DataFrame):
    scaler = StandardScaler()
    scaled_X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)  # Transforming to Dataframe before returning
    return scaled_X


X = scale(X)


def number_of_clusters(X: pd.DataFrame):  # Figuring out how many clusters we need using elbow method
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}

    for i in range(1, 10):
        model = KMeans(n_clusters=i).fit(X)

        distortions.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        inertias.append(model.inertia_)

        mapping1[i] = sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]
        mapping2[i] = model.inertia_

    sns.lineplot(x=range(1, 10), y=distortions)  # Using a graph to clearly see
    plt.xlabel("n_clusters")
    plt.ylabel("Distortion")
    plt.title("Elbow Method Using Distortion")
    plt.show()  # Seems like 5 clusters is optimal for this data.


number_of_clusters(X)  # Deciding on cluster number


# After deciding the cluster number, time to apply clustering algorithms and see which performs better on our dataset


def apply_kmeans(X: pd.DataFrame):
    kmeans = KMeans(n_clusters=5)
    y_kmeans = kmeans.fit_predict(X)
    cluster_id = pd.Series(y_kmeans, name="cluster_id")

    # Concatenating with country column to take a better look
    data_kmeans = pd.concat([data["country"], X, cluster_id], axis=1)
    return data_kmeans


def apply_hierarchical(X: pd.DataFrame):
    hierarchical_cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    y_hierarchical = hierarchical_cluster.fit_predict(X)
    cluster_id = pd.Series(y_hierarchical, name="cluster_id")

    # Concatenating with country column to take a better look
    data_hierarchical = pd.concat([data["country"], X, cluster_id], axis=1)
    return data_hierarchical


def eps_elbow_dbscan(X: pd.DataFrame):
    from sklearn.neighbors import NearestNeighbors

    neighbors = NearestNeighbors(n_neighbors=20)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    sns.lineplot(distances)
    plt.show()


def apply_dbscan(X: pd.DataFrame):
    dbscan = DBSCAN(eps=1.5, min_samples=2)
    y_dbscan = dbscan.fit_predict(X)
    cluster_id = pd.Series(y_dbscan, name="cluster_id")

    # Concatenating with country column to take a better look
    data_dbscan = pd.concat([data["country"], X, cluster_id], axis=1)
    return data_dbscan


# Applying various clustering algorithms
data_kmeans = apply_kmeans(X)
data_hierarchical = apply_hierarchical(X)

# DBSCAN works a little different from other clustering methods, so we will use an elbow function
# to determine the best parameters for dbscan.
eps_elbow_dbscan(X)  # After zooming into graph, we see that the optimum value for eps is around 3000.

# Applying dbscan with optimum parameters
data_dbscan = apply_dbscan(X)


# Now we need to decide which algorithm performs best for our dataset. A method to compare them is to
# plot the data in 2D space to see clusters clearly.

# Reducing dimensions to 2
def apply_pca_2d(X: pd.DataFrame):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X.drop(["cluster_id", "country"], axis=1))
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['PC1', 'PC2'])
    principalDf = pd.concat([X["country"], principalDf, X["cluster_id"]], axis=1)
    return principalDf


# Applying pca to every dataset
pca_kmeans = apply_pca_2d(data_kmeans)
pca_hierarchical = apply_pca_2d(data_hierarchical)
pca_dbscan = apply_pca_2d(data_dbscan)

# Plotting every algorithm side by side to easy comparison
fig, axes = plt.subplots(1, 3)
sns.scatterplot(data=pca_kmeans, x="PC1", y="PC2", hue="cluster_id", palette="deep", ax=axes[0])
sns.scatterplot(data=pca_hierarchical, x="PC1", y="PC2", hue="cluster_id", palette="deep", ax=axes[1])
sns.scatterplot(data=pca_dbscan, x="PC1", y="PC2", hue="cluster_id", palette="deep", ax=axes[2])
fig.suptitle("Clustering Algorithms Comparison")
axes[0].set_title("KMeans")
axes[1].set_title("Hierarchical")
axes[2].set_title("DBSCAN")
plt.show()

# After examining the graphs, I decided that the KMeans clustering did the best for our data.
X_kmeans = data_kmeans.drop(["country"], axis=1)


# We will use ANOVA Test to see if our clusters actually different and meaningful in terms of statistics.
def anova(X: pd.DataFrame):
    from scipy.stats import f_oneway

    clusters = X.groupby("cluster_id")
    cluster0 = clusters.get_group(0).drop(["cluster_id"], axis=1)
    cluster1 = clusters.get_group(1).drop(["cluster_id"], axis=1)
    cluster2 = clusters.get_group(2).drop(["cluster_id"], axis=1)
    cluster3 = clusters.get_group(3).drop(["cluster_id"], axis=1)
    cluster4 = clusters.get_group(4).drop(["cluster_id"], axis=1)
    return f_oneway(cluster0, cluster1, cluster2, cluster3, cluster4)


def tukey(X: pd.DataFrame, col: str):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    return pairwise_tukeyhsd(endog=X[col], groups=X["cluster_id"], alpha=0.05)


f, p = anova(X_kmeans)
print(f"{f}\n{p}")  # We infer that there is no significant difference between values in imports column.
# Same goes for the inflation column.

# We can observe how the clusters differs from each other column by column.
# For example, first cluster1 significantly differs from cluster4 in terms of 'exports'.
# But it does not significantly differ from cluster3 in terms of 'exports'.
tukey_res = tukey(X_kmeans, col="exports")
print(tukey_res)


# Finally, we plot our cluster.
sns.scatterplot(pca_hierarchical, x="PC1", y="PC2", hue="cluster_id", palette="deep")
plt.title("Final Cluster")
plt.show()

pass
