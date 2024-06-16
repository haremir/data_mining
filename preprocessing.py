import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

#-----------------Sentetik veri ekleme güncellemesi--------------

def add_synthetic_data_numeric(df, target_column):
    if target_column not in df.columns:
        raise ValueError(f"Hedef sütun '{target_column}' DataFrame'de bulunamadı.")
    if df[target_column].isnull().any():
        raise ValueError(f"Hedef sütun '{target_column}' eksik değerlere sahip.")
    
    X = df.drop(target_column, axis=1).select_dtypes(include=['float64', 'int64'])
    y = df[target_column]
    
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    synthetic_data = pd.DataFrame(X_resampled, columns=X.columns)
    synthetic_data[target_column] = y_resampled
    
    df_combined = pd.concat([df, synthetic_data], ignore_index=True)
    return df_combined

def add_synthetic_data_categorical(df, target_column):
    if target_column not in df.columns:
        raise ValueError(f"Hedef sütun '{target_column}' DataFrame'de bulunamadı.")
    if df[target_column].isnull().any():
        raise ValueError(f"Hedef sütun '{target_column}' eksik değerlere sahip.")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    minority_class = y.value_counts().idxmin()
    minority_class_count = y.value_counts().min()
    majority_class_count = y.value_counts().max()
    
    minority_data = df[df[target_column] == minority_class]
    synthetic_data = minority_data.sample(majority_class_count - minority_class_count, replace=True, random_state=42)
    
    df_combined = pd.concat([df, synthetic_data], ignore_index=True)
    return df_combined

def add_synthetic_data_mixed(df, target_column):
    if target_column not in df.columns:
        raise ValueError(f"Hedef sütun '{target_column}' DataFrame'de bulunamadı.")
    if df[target_column].isnull().any():
        raise ValueError(f"Hedef sütun '{target_column}' eksik değerlere sahip.")
    
    # Kategorik veriler için azınlık sınıfı çoğaltma
    df_balanced = add_synthetic_data_categorical(df, target_column)
    
    # Numerik veriler için SMOTE
    X = df_balanced.drop(target_column, axis=1)
    X_numeric = X.select_dtypes(include=['float64', 'int64'])
    y = df_balanced[target_column]
    
    # Sınıf dağılımını kontrol et
    class_counts = y.value_counts()
    print(f"Sınıf dağılımı: {class_counts}")

    # SMOTE ile her sınıfın sayısını 2 katına çıkar
    sampling_strategy = {cls: count * 3 for cls, count in class_counts.items()}
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_numeric, y)
    
    # Sentetik verileri orijinal veri setiyle birleştir
    X_resampled_df = pd.DataFrame(X_resampled, columns=X_numeric.columns)
    X_resampled_df[target_column] = y_resampled
    
    df_combined = pd.concat([df_balanced, X_resampled_df], ignore_index=True)
    return df_combined
#-----------------dengesizlik çözüm--------------
def balanced_data(data):
    # 1 ve 2 sınıflarını ayrı ayrı alt örneklemlerle güncelle
    class_1 = data[data['LUNG_CANCER'] == 1]
    class_2 = data[data['LUNG_CANCER'] == 2]

    # Her iki sınıf için de örnek sayısını kontrol et
    min_samples = min(len(class_1), len(class_2))
    print(len(class_1), len(class_2))

    if min_samples < 30:
        raise ValueError("Her iki sınıf için de minimum 30 örnek olmalı.")

    # Her iki sınıf için de örnek sayısını 30'a çıkar
    class_1_resampled = resample(class_1, replace=True, n_samples=30, random_state=42)
    class_2_resampled = resample(class_2, replace=True, n_samples=30, random_state=42)


    # Güncellenmiş veri setini oluştur
    balanced_data = pd.concat([class_1_resampled, class_2_resampled])

    return balanced_data

def add_synthetic_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # SMOTE kullanarak sentetik veri ekleyin
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Sentetik verileri DataFrame'e çevirin
    synthetic_data = pd.DataFrame(X_resampled, columns=X.columns)
    synthetic_data[target_column] = y_resampled

    # Orijinal veri ile birleştirin
    df = pd.concat([df, synthetic_data], ignore_index=True)

    return df

def add_synthetic_data_cluster(df, target_column):
    # Hedef sütununun DataFrame'de bulunduğundan emin olun
    if target_column not in df.columns:
        raise ValueError(f"Hedef sütun '{target_column}' DataFrame'de bulunamadı.")
    
    # Hedef sütunda eksik değerler olup olmadığını kontrol edin
    if df[target_column].isnull().any():
        raise ValueError(f"Hedef sütun '{target_column}' eksik değerlere sahip.")
    
    # Özellikleri (X) ve hedef sütunu (y) ayırın
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # SMOTE kullanarak sentetik veri ekleyin
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Sentetik verileri DataFrame'e çevirin
    synthetic_data = pd.DataFrame(X_resampled, columns=X.columns)
    synthetic_data[target_column] = y_resampled
    
    # Orijinal veri ile birleştirin
    df_combined = pd.concat([df, synthetic_data], ignore_index=True)
    
    return df_combined



def preprocess_categorical_and_balance_data(file_path):
    print("-" * 80, end="\n\n")
    print("Veri ön işleme ve dengeleme işlemleri yapılıyor...", end="\n\n")
    print("-" * 80, end="\n\n")

    # Veri setini yükleyin
    df = pd.read_excel(file_path)
    print(df['LUNG_CANCER'].value_counts())

    # 'ALLERGY' sütununu işleme ekleme
    if 'ALLERGY' in df.columns:
        df['ALLERGY'] = df['ALLERGY'].map({2: 'YES', 1: 'NO'}).astype(str)

    # Diğer kategorik sütunları dönüştürme
    categorical_columns = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                            'CHRONIC DISEASE', 'FATIGUE', 'WHEEZING', 'ALCOHOL CONSUMING',
                            'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'COUGHING', 'CHEST PAIN']

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].map({2: 'YES', 1: 'NO'}).astype(str)

    # Boşluk karakterlerini temizle
    df.columns = df.columns.str.strip()

    # 'ALLERGY' sütununu tekrar kontrol et
    if 'ALLERGY' in df.columns:
        df['ALLERGY'] = df['ALLERGY'].astype(str)

    # Sınıf dengesini kontrol et
    class_counts = df['LUNG_CANCER'].value_counts()
    
    # Minimum örnek sayısı
    min_samples = 30
    
    # Sınıf dengesi uygunsa veriyi dengeleme işlemine gönder
    while any(count < min_samples for count in class_counts):
        # Dengeleme yapılacak sınıfı bul
        imbalanced_class = class_counts.idxmin()
        
        # Sınıfı dengele
        class_df = df[df['LUNG_CANCER'] == imbalanced_class]
        class_resampled = resample(class_df, replace=True, n_samples=min_samples, random_state=42)
        
        # Dengeleme sonucunu diğer sınıf ile birleştir
        df = pd.concat([df[df['LUNG_CANCER'] != imbalanced_class], class_resampled])
        
        # Sınıf dengesini güncelle
        class_counts = df['LUNG_CANCER'].value_counts()

    print("Veri ön işleme ve dengeleme tamamlandı:")
    print(df.head())
    print(df.info())
    print(df['LUNG_CANCER'].value_counts())

    return df

def balance_classes(data, target_column='LUNG_CANCER', minority_class='NO', majority_class='YES', sample_size=30, random_state=42):
    # Sınıfları ayır
    minority_class_data = data[data[target_column] == minority_class]
    majority_class_data = data[data[target_column] == majority_class]

    # Minority class'ı belirtilen örnek sayısına getir
    minority_class_resampled = resample(minority_class_data, replace=True, n_samples=sample_size, random_state=random_state)

    # Majority class'ı belirtilen örnek sayısına düşür
    majority_class_resampled = resample(majority_class_data, replace=False, n_samples=sample_size, random_state=random_state)

    # Dengelenmiş eğitim verilerini birleştir
    balanced_data = pd.concat([minority_class_resampled, majority_class_resampled])

    return balanced_data

#----------------- düz veri seti-----------------

def preprocess_numeric_data(df):
    print("-"*80, end="\n\n")
    print("burada şuanda sadece numerik değerler var", end="\n\n")
    print("-"*80, end="\n\n")

    # Cinsiyet sütununu numerik değerlere dönüştürün (F: 2, M: 1)
    df['GENDER'] = df['GENDER'].map({'F': 2, 'M': 1})
    
    # Lung Cancer sütununu numerik değerlere dönüştürün (YES: 2, NO: 1)
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 2, 'NO': 1})

    # Yaş verisini normalize etme
    scaler = MinMaxScaler()
    df[['AGE']] = scaler.fit_transform(df[['AGE']])

    # Boşluk karakterlerini temizle
    df.columns = df.columns.str.strip()
    print("sadece numerik")
    print(df.head())
    print(df.info())

    return df




def preprocess_categorical_data(file_path):
    print("-" * 80, end="\n\n")
    print("Şu anda sadece kategorik değerler var", end="\n\n")
    print("-" * 80, end="\n\n")

    # Veri setini yükleyin
    df = pd.read_excel(file_path)
    print(df['LUNG_CANCER'].value_counts())

    # 'ALLERGY' sütununu işleme ekleme
    if 'ALLERGY' in df.columns:
        df['ALLERGY'] = df['ALLERGY'].map({2: 'YES', 1: 'NO'}).astype(str)

    # Diğer kategorik sütunları dönüştürme
    categorical_columns = [ 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                            'CHRONIC DISEASE', 'FATIGUE', 'WHEEZING', 'ALCOHOL CONSUMING',
                            'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'COUGHING', 'CHEST PAIN']

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].map({2: 'YES', 1: 'NO'}).astype(str)

    # Boşluk karakterlerini temizle
    df.columns = df.columns.str.strip()

    # 'ALLERGY' sütununu tekrar kontrol et
    if 'ALLERGY' in df.columns:
        df['ALLERGY'] = df['ALLERGY'].astype(str)

    print("Sadece kategorik:")
    print(df.head())
    print(df.info())
    print(df['LUNG_CANCER'].value_counts())


    return df

def clean_and_normalize_data(file_path):
    print("-"*80, end="\n\n")
    print("burada şuanda sadece karma değerler var", end="\n\n")
    print("-"*80, end="\n\n")

    # Veri setinizi yükleyin
    df = pd.read_excel(file_path)

    # Yaş verisini normalize etme
    scaler = MinMaxScaler()
    df[['AGE']] = scaler.fit_transform(df[['AGE']])

    # Cinsiyet sütununu numerik değerlere dönüştürün (F: 2, M: 1)
    #df['GENDER'] = df['GENDER'].map({'F': 2, 'M': 1})

    # Lung Cancer sütununu numerik değerlere dönüştürün (YES: 2, NO: 1)
    #df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 2, 'NO': 1})

    # Boşluk karakterlerini temizle
    df.columns = df.columns.str.strip()
    print("karma")
    print(df.head())
    print(df.info())
    return df

# ----------kümelemeler için olan alan------------

def preprocess_categorical_data_cluster(file_path):
    print("-" * 80, end="\n\n")
    print("Şu anda sadece kategorik değerler var", end="\n\n")
    print("-" * 80, end="\n\n")

    # Veri setini yükleyin
    df = pd.read_excel(file_path)
    print(df['LUNG_CANCER'].value_counts())

    # 'ALLERGY' sütununu işleme ekleme
    if 'ALLERGY' in df.columns:
        df['ALLERGY'] = df['ALLERGY'].map({2: 'YES', 1: 'NO'}).astype(str)

    # Diğer kategorik sütunları dönüştürme
    categorical_columns = [ 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                            'CHRONIC DISEASE', 'FATIGUE', 'WHEEZING', 'ALCOHOL CONSUMING',
                            'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'COUGHING', 'CHEST PAIN']

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].map({2: 'YES', 1: 'NO'}).astype(str)

    # 'Cluster_KMeans' sütununu dönüştürme
    if 'Cluster_KMeans' in df.columns:
        df['Cluster_KMeans'] = df['Cluster_KMeans'].map({0: 'ZERO', 1: 'ONE', 2: 'TWO'}).astype(str)

    # Boşluk karakterlerini temizle
    df.columns = df.columns.str.strip()

    # 'ALLERGY' sütununu tekrar kontrol et
    if 'ALLERGY' in df.columns:
        df['ALLERGY'] = df['ALLERGY'].astype(str)

    print("Sadece kategorik:")
    print(df.head())
    print(df.info())
    print(df['LUNG_CANCER'].value_counts())

    return df
# ----------kümelemeler için olan alan sentetik veriye uygun------------

def preprocess_categorical_data_cluster_synthetic(df):
    print("-" * 80, end="\n\n")
    print("Şu anda sadece kategorik değerler var", end="\n\n")
    print("-" * 80, end="\n\n")

    print(df['LUNG_CANCER'].value_counts())

    # 'ALLERGY' sütununu işleme ekleme
    if 'ALLERGY' in df.columns:
        df['ALLERGY'] = df['ALLERGY'].map({2: 'YES', 1: 'NO'}).astype(str)

    # Diğer kategorik sütunları dönüştürme
    categorical_columns = [ 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                            'CHRONIC DISEASE', 'FATIGUE', 'WHEEZING', 'ALCOHOL CONSUMING',
                            'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'COUGHING', 'CHEST PAIN']

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].map({2: 'YES', 1: 'NO'}).astype(str)

    # 'Cluster_KMeans' sütununu dönüştürme
    if 'Cluster_KMeans' in df.columns:
        df['Cluster_KMeans'] = df['Cluster_KMeans'].map({0: 'ZERO', 1: 'ONE', 2: 'TWO'}).astype(str)

    # Boşluk karakterlerini temizle
    df.columns = df.columns.str.strip()

    # 'ALLERGY' sütununu tekrar kontrol et
    if 'ALLERGY' in df.columns:
        df['ALLERGY'] = df['ALLERGY'].astype(str)

    print("Sadece kategorik:")
    print(df.head())
    print(df.info())
    print(df['LUNG_CANCER'].value_counts())

    return df


def clean_and_normalize_data_cluster_synthetic(df):
    print("-"*80, end="\n\n")
    print("burada şuanda sadece karma değerler var", end="\n\n")
    print("-"*80, end="\n\n")

    # Yaş verisini normalize etme
    scaler = MinMaxScaler()
    df[['AGE']] = scaler.fit_transform(df[['AGE']])

    # Cinsiyet sütununu string değerlere dönüştürün (F: 'F', M: 'M')
    df['GENDER'] = df['GENDER'].map({'F': 'F', 'M': 'M'}).astype(str)

    # Lung Cancer sütununu string değerlere dönüştürün (YES: 'YES', NO: 'NO')
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 'YES', 'NO': 'NO'}).astype(str)

    # Boşluk karakterlerini temizle
    df.columns = df.columns.str.strip()

    # Eksik değerleri doldurma
    df = df.fillna({
        'GENDER': 'Unknown',       # GENDER sütunundaki eksik değerleri 'Unknown' ile dolduruyoruz
        'LUNG_CANCER': 'Unknown'   # LUNG_CANCER sütunundaki eksik değerleri 'Unknown' ile dolduruyoruz
    })

    print("karma")
    print(df.head())
    print(df.info())
    return df