import pandas as pd
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Veri setini yükle (CSV dosyası)
df = pd.read_csv(r'C:\Users\emirh\Desktop\survey lung cancer emir edition.csv', delimiter=';')
# Sütun isimlerinden baştaki/sondaki boşlukları kaldır
df.columns = df.columns.str.strip()

# İlgili sütunları seç
X = df[['AGE', 'LUNG_CANCER']]

# Kategorik değişkenleri sayısala dönüştür (eğer gerekiyorsa)
X = pd.get_dummies(X)

# Verileri normalize et (isteğe bağlı)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Kümeleme modelini oluştur ve eğit (MiniBatchKMeans)
n_clusters = 3  # İstenen küme sayısı
minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
minibatch_kmeans.fit(X_scaled)

# Küme merkezlerini ve etiketleri al
cluster_centers = scaler.inverse_transform(minibatch_kmeans.cluster_centers_)
labels_kmeans = minibatch_kmeans.labels_

# 'Cluster' sütununu DataFrame'e ekle
df['Cluster_KMeans'] = labels_kmeans

# Küme merkezlerini görüntüle (isteğe bağlı)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=X.columns)
print("Cluster Centers (K-Means):")
print(cluster_centers_df)

# Küme sayısını belirlemek için Elbow Method kullanabilirsiniz (isteğe bağlı)
# WCSS (Within-Cluster Sum of Squares) değerlerini depolamak için bir liste oluşturun
wcss = []
for i in range(1, 5):
    minibatch_kmeans = MiniBatchKMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=12, random_state=42)
    minibatch_kmeans.fit(X_scaled)
    wcss.append(minibatch_kmeans.inertia_)

# Elbow Method'un grafiğini çizin
plt.plot(range(1, 5), wcss)
plt.title('Elbow Method (K-Means)')
plt.xlabel('küme sayısı')
plt.ylabel('WCSS')  # Within-Cluster Sum of Squares
plt.show()

# DBSCAN modelini oluştur ve eğit
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

# 'Cluster_DBSCAN' sütununu DataFrame'e ekle
df['Cluster_DBSCAN'] = labels_dbscan

# Kümeleme sonuçlarını görüntüle
print("Kümeleme Sonuçları (K-Means ve DBSCAN):")
print(df[['AGE', 'LUNG_CANCER', 'Cluster_KMeans', 'Cluster_DBSCAN']])

# Kümeleme sonuçlarını görselleştir (DBSCAN)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_dbscan, cmap='magma', marker='x', alpha=0.5, label='DBSCAN Clusters')
plt.title('DBSCAN ile Kümeleme Sonuçları')
plt.xlabel('Age')
plt.ylabel('Lung Cancer')
plt.legend()
plt.show()

# Kümeleme sonuçlarını görselleştir (K-Means)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_kmeans, cmap='viridis', alpha=0.5, marker='o', label='K-Means Clusters')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers (K-Means)')

plt.title('K-Means ile Kümeleme Sonuçları')
plt.xlabel('Age')
plt.ylabel('Lung Cancer')
plt.legend()
plt.show()

# Kümeleme sonuçlarını görselleştir (Original Data)
plt.scatter(df['LUNG_CANCER'], df['AGE'], c=labels_kmeans, cmap='coolwarm', marker='.', alpha=0.5, label='Original Data (K-Means)')
plt.title('Original Data with K-Means Clusters')
plt.xlabel('LUNG_CANCER')
plt.ylabel('AGE')
plt.legend()
plt.show()


# Kümeleme sonuçlarını bir metin dosyasına yazdır (K-Means)
output_file_path_kmeans = 'kumeleme_sonuclari_kmeans.txt'
output_file_path_dbscan = 'kumeleme_sonuclari_dbscan.txt'

# Cluster Centers'ı dosyaya yazdır (K-Means)
with open(output_file_path_kmeans, 'w') as file_kmeans:
    file_kmeans.write("Cluster Centers (K-Means):\n")
    file_kmeans.write(cluster_centers_df.to_string())

# Elde edilen kümeleme sonuçlarını dosyaya yazdır (K-Means)
with open(output_file_path_kmeans, 'a') as file_kmeans:
    file_kmeans.write("\n\nKümeleme Sonuçları (K-Means):\n")
    file_kmeans.write(df[['AGE', 'LUNG_CANCER', 'Cluster_KMeans']].to_string())

# Kümeleme sonuçlarını bir metin dosyasına yazdır (DBSCAN)
with open(output_file_path_dbscan, 'w') as file_dbscan:
    file_dbscan.write("Cluster Centers (DBSCAN):\n")
    file_dbscan.write("Not applicable for DBSCAN\n")

# Elde edilen kümeleme sonuçlarını dosyaya yazdır (DBSCAN)
with open(output_file_path_dbscan, 'a') as file_dbscan:
    file_dbscan.write("\n\nKümeleme Sonuçları (DBSCAN):\n")
    file_dbscan.write(df[['AGE', 'LUNG_CANCER', 'Cluster_DBSCAN']].to_string())

# Dosyaya yazma işlemi tamamlandı mesajlarını ekrana yazdır
print(f"Kümeleme sonuçları (K-Means) {output_file_path_kmeans} adlı dosyaya başarıyla yazıldı.")
print(f"Kümeleme sonuçları (DBSCAN) {output_file_path_dbscan} adlı dosyaya başarıyla yazıldı.")

# Kümeleme sonuçlarını görselleştir (Original Data without normalization)
plt.scatter(X['LUNG_CANCER'], X['AGE'], c=labels_kmeans, cmap='coolwarm', marker='.', alpha=0.5, label='Original Data (K-Means)')
plt.title('Original Data with K-Means Clusters (Without Normalization)')
plt.xlabel('LUNG_CANCER')
plt.ylabel('AGE')
plt.legend()
plt.show()

mixed_labels = df[(df['Cluster_KMeans'] == 0) & (df['LUNG_CANCER'] == 'YES') | (df['Cluster_KMeans'] == 1) & (df['LUNG_CANCER'] == 'NO')]
print("Aynı kümede hem hasta hem de sağlıklı bireylerin bulunduğu gözlemler:")
print(mixed_labels[['AGE', 'LUNG_CANCER', 'Cluster_KMeans']])
conflicting_clusters = df[df.duplicated(subset=['AGE'], keep=False) & (df['Cluster_KMeans'].duplicated(keep=False))]
print(conflicting_clusters)

