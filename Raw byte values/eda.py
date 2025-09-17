import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load dataset CSV
df = pd.read_csv("malware_dataset.csv")

# Class distribution
sns.countplot(x="label", data=df)
plt.title("Class Distribution")
plt.show()

# Trung bình byte value theo class
benign_mean = df[df["label"]=="Benign"].drop("label", axis=1).mean()
virus_mean = df[df["label"]=="Virus"].drop("label", axis=1).mean()

plt.plot(benign_mean[:200], label="Benign")
plt.plot(virus_mean[:200], label="Virus")
plt.legend()
plt.title("Byte Distribution (first 200 bytes)")
plt.show()

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.drop("label", axis=1))

plt.scatter(X_pca[:,0], X_pca[:,1], 
            c=(df["label"]=="Virus"), cmap="coolwarm", alpha=0.6)
plt.title("PCA Malware vs Benign")
plt.show()
