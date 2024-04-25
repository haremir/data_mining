import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# Veri setini yükle (CSV dosyası)
df = pd.read_csv(r'C:\Users\emirh\Desktop\survey lung cancer emir edition.csv', delimiter=';')
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

# Küme etiketlerini al
labels_kmeans = minibatch_kmeans.labels_

# 'Cluster' sütununu DataFrame'e ekle
df['Cluster_KMeans'] = labels_kmeans

# Sınıf etiketlerini belirle ve 'LUNG_CANCER' sütununu güncelle
for cluster_label in range(n_clusters):
    # Her küme için sınıf etiketini belirle
    cluster_class = df.loc[df['Cluster_KMeans'] == cluster_label, 'LUNG_CANCER'].value_counts().idxmax()
    # Kümeleme sonuçlarına göre 'LUNG_CANCER' sütununu güncelle
    df.loc[df['Cluster_KMeans'] == cluster_label, 'LUNG_CANCER'] = cluster_class

# Sonuçları Excel dosyasına yaz
output_excel_path = 'kumeleme_sonuclari.xlsx'
df.to_excel(output_excel_path, index=False)

# Dosyaya yazma işlemi tamamlandı mesajını ekrana yazdır
print(f"Kümeleme sonuçları {output_excel_path} adlı Excel dosyasına başarıyla yazıldı.")
