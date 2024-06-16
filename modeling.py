from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict,KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
from preprocessing import preprocess_numeric_data, preprocess_categorical_data, clean_and_normalize_data
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold


#-------------------------k-fold modelleri----------------------
def train_decision_tree_model_with_numeric_data_kFold(data, n_splits=5):
    print("-"*80, end="\n\n")
    print("Burada şu anda DECISION TREE k-fold algoritması çalışıyor", end="\n\n")
    print("-"*80, end="\n\n")

    # Bağımsız değişkenler (X) ve bağımlı değişken (y) ayrımı
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']

    # Decision Tree modelini oluştur
    model = DecisionTreeClassifier(random_state=42)

    # K-Fold çapraz doğrulama için StratifiedKFold kullanın
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Çapraz doğrulama skorlarını alın
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    # Çapraz doğrulama skorlarını ekrana yazdırın
    print(f'Cross-Validation Scores: {cv_scores}')
    print(f'Mean Accuracy: {cv_scores.mean()}')

    # Modeli eğitip tahminleri alın
    y_pred = cross_val_predict(model, X, y, cv=skf)

    # Classification Report ve Confusion Matrix'i ekrana yazdırın
    print('\nClassification Report:')
    print(classification_report(y, y_pred))

    print('\nConfusion Matrix:')
    conf_matrix = confusion_matrix(y, y_pred)
    print(conf_matrix)

    return model

def train_catboost_model_categorical_kFold(data, n_splits=5):
    print("-"*80, end="\n\n")
    print("Burada şu anda CATBOOST k-folds algoritması çalışıyor sadece kategorik verilerle", end="\n\n")
    print("-"*80, end="\n\n")

    # Bağımsız değişkenler (X) ve bağımlı değişken (y) ayrımı
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']

    # K-Fold çapraz doğrulama için StratifiedKFold kullanın
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # CatBoost modelini oluştururken kategorik özellikleri belirtin
    cat_features = ['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                    'CHRONIC DISEASE', 'FATIGUE', 'WHEEZING', 'ALCOHOL CONSUMING',
                    'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'COUGHING', 'CHEST PAIN']

    # Çapraz doğrulama işlemi
    cv_scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # CatBoost modelini oluşturun
        model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='Logloss', random_state=42, verbose=False)

        # Modeli eğit
        model.fit(X_train, y_train, cat_features=cat_features)

        # Test verisi üzerinde modelin performansı
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f'Model Accuracy with CatBoost: {accuracy}')

        # Diğer performans verileri
        print('Classification Report:')
        print(classification_report(y_test, y_pred))

        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))

        cv_scores.append(accuracy)

    # Çapraz doğrulama skorlarını ekrana yazdırın
    print(f'Cross-Validation Scores: {cv_scores}')
    print(f'Mean Accuracy: {np.mean(cv_scores)}')

    return model

def train_catboost_model_mixed_kFold(data, n_splits=5):
    print("-"*80, end="\n\n")
    print("Burada şu anda CATBOOST algoritması çalışıyor karışık verilerle ve k-fold çapraz doğrulama kullanılıyor.", end="\n\n")
    print("-"*80, end="\n\n")

    # Bağımsız değişkenler (X) ve bağımlı değişken (y) ayrımı
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']

    # Kategorik sütunları belirle (örneğin, 'GENDER' gibi)
    cat_features = ['GENDER']

    # CatBoost modelini oluştur
    model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', random_state=42, cat_features=cat_features, verbose=False)

    # K-fold çapraz doğrulama nesnesini oluştur
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Her k-fold için modeli eğit ve değerlendir
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        test_pool = Pool(X_test, y_test, cat_features=cat_features)

        model.fit(train_pool, eval_set=test_pool, verbose=False)

        # Test verisi üzerinde modelin performansı
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f'Model Accuracy with CatBoost (Fold {fold}): {accuracy}')

        # Diğer performans verileri
        print('Classification Report:')
        print(classification_report(y_test, y_pred))

        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print("-" * 80, end="\n\n")

    return model

#-------------------------yeni modeller----------------------
def train_decision_tree_model_with_balanced_data(data):
    print("-"*80, end="\n\n")
    print("Burada şu anda DECISION TREE EMİR algoritması çalışıyor", end="\n\n")
    print("-"*80, end="\n\n")

    # Bağımsız değişkenler (X) ve bağımlı değişken (y) ayrımı
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']

    # Decision Tree modelini oluştur
    model = DecisionTreeClassifier(random_state=42)

    # Modeli eğitim verisi ile eğit
    model.fit(X, y)

    # Test verisi üzerinde modelin performansı
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    print(f'Model Accuracy with Balanced Data: {accuracy}')

    # Diğer performans verileri
    print('Classification Report:')
    print(classification_report(y, y_pred))

    print('Confusion Matrix:')
    print(confusion_matrix(y, y_pred))

    return model

def train_catboost_model_categorical_resample(data):
    print("-" * 80, end="\n\n")
    print("Burada şu anda CATBOAST EMİR algoritması çalışıyor sadece kategorik verilerle", end="\n\n")
    print("-" * 80, end="\n\n")

    # Sınıfları ayır
    class_1 = data[data['LUNG_CANCER'] == 'YES']
    class_2 = data[data['LUNG_CANCER'] == 'NO']

    # Sınıf 1'in eğitim verilerini belirli bir sayıda örnek seçerek oluştur
    class_1_train = class_1.sample(n=30, random_state=42)

    # Sınıf 2'nin eğitim verilerini belirli bir sayıda örnek seçerek oluştur
    class_2_train = class_2.sample(n=30, random_state=42)

    # Dengelenmiş eğitim verilerini birleştir
    train_data = pd.concat([class_1_train, class_2_train])

    # Bağımsız değişkenler (X) ve bağımlı değişken (y) ayrımı
    X_train = train_data.drop('LUNG_CANCER', axis=1)
    y_train = train_data['LUNG_CANCER']

    # Veri setini eğitim ve test setlerine bölelim
    X, y = data.drop('LUNG_CANCER', axis=1), data['LUNG_CANCER']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Sınıf ağırlıklarını belirle (manuel olarak)
    class_weights = [len(y_train) / (2 * np.bincount(pd.Categorical(y_train).codes))]  # Sınıf sayısına göre dengeli ağırlıklar

    # CatBoost modelini oluştur
    model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='Logloss', random_state=42, verbose=False)

    # Modeli eğitim verisi ile eğit
    model.fit(X_train, y_train, cat_features=['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                                              'CHRONIC DISEASE', 'FATIGUE', 'WHEEZING', 'ALCOHOL CONSUMING',
                                              'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'COUGHING', 'CHEST PAIN'],
              sample_weight=list(map(lambda x: class_weights[0][pd.Categorical(x).codes[0]], y_train)))

    # Test verisi üzerinde modelin performansı
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Model Accuracy with CatBoost: {accuracy}')

    # Diğer performans verileri
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    return model

def train_catboost_model_mixed_resample(data):
    print("-" * 80, end="\n\n")
    print("Burada şu anda CATBOAST EMİR algoritması çalışıyor karma verilerle", end="\n\n")
    print("-" * 80, end="\n\n")

    # Sınıfları ayır
    class_1 = data[data['LUNG_CANCER'] == 'YES']
    class_2 = data[data['LUNG_CANCER'] == 'NO']

    # Sınıf 1 ve 2 için eğitim ve test verilerini ayarla (15 örnek)
    class_1_train, class_1_test = train_test_split(class_1, test_size=0.50, random_state=42)
    class_2_train, class_2_test = train_test_split(class_2, test_size=0.50, random_state=42)

    # Sınıfları dengeli hale getir
    while len(class_1_train) < 30:
        class_1_train = pd.concat([class_1_train, class_1_train.sample(n=min(30, len(class_1_train)), replace=True)])

    while len(class_2_train) < 30:
        class_2_train = pd.concat([class_2_train, class_2_train.sample(n=min(30, len(class_2_train)), replace=True)])

    # Dengelenmiş eğitim verilerini birleştir
    train_data = pd.concat([class_1_train, class_2_train])

    # Test verisi üzerinde modelin performansı
    test_data = pd.concat([class_1_test, class_2_test])
    X_test, y_test = test_data.drop('LUNG_CANCER', axis=1), test_data['LUNG_CANCER']

    # Bağımsız değişkenler (X) ve bağımlı değişken (y) ayrımı
    X_train, y_train = train_data.drop('LUNG_CANCER', axis=1), train_data['LUNG_CANCER']

    # Class weighting hesaplaması
    class_weights = [len(class_2_train) / (2 * len(class_1_train)), len(class_1_train) / (2 * len(class_2_train))]

    # CatBoost modelini oluştur
    model = CatBoostClassifier(iterations=400, depth=16, learning_rate=0.1, loss_function='Logloss', random_state=42, verbose=False,
                               class_weights=class_weights)

    # Modeli eğitim verisi ile eğit
    model.fit(X_train, y_train, cat_features=['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                                              'CHRONIC DISEASE', 'FATIGUE', 'WHEEZING', 'ALCOHOL CONSUMING',
                                              'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'COUGHING', 'CHEST PAIN'])

    # Test verisi üzerinde modelin performansı
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Model Accuracy with CatBoost: {accuracy}')

    # Diğer performans verileri
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    return model
#-------------------------eski modeller----------------------

# sadece numerik datayla çalışıyor
def train_decision_tree_model_with_numeric_data(data):
    print("-"*80, end="\n\n")
    print("burada şuanda DECISION TREE algoritması çalışıyor", end="\n\n")
    print("-"*80, end="\n\n")


    # Bağımsız değişkenler (X) ve bağımlı değişken (y) ayrımı
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']

    # Veri setini eğitim ve test setlerine bölelim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.229, random_state=42)

    # Decision Tree modelini oluştur
    model = DecisionTreeClassifier(random_state=42)

    # Modeli eğitim verisi ile eğit
    model.fit(X_train, y_train)

    # Test verisi üzerinde modelin performansı
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Model Accuracy with Numeric Data: {accuracy}')

    # Diğer performans verileri
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    return model

# sadece kategorik datayla çalışıyor
def train_catboost_model_categorical(data):
    print("-"*80, end="\n\n")
    print("burada şuanda CATBOAST algoritması çalışıyor sadece kategorik verilerle", end="\n\n")
    print("-"*80, end="\n\n")


    # Bağımsız değişkenler (X) ve bağımlı değişken (y) ayrımı
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']

    # Veri setini eğitim ve test setlerine bölelim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # CatBoost modelini oluştur
    model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='Logloss', random_state=42, verbose=False)

    # Modeli eğitim verisi ile eğit
    model.fit(X_train, y_train, cat_features=['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                                              'CHRONIC DISEASE', 'FATIGUE', 'WHEEZING', 'ALCOHOL CONSUMING',
                                              'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'COUGHING', 'CHEST PAIN'])

    # Test verisi üzerinde modelin performansı
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Model Accuracy with CatBoost: {accuracy}')

    # Diğer performans verileri
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    return model

# karma
def train_catboost_model_mixed(data):
    
    print("-"*80, end="\n\n")
    print("burada şuanda CATBOAST algoritması çalışıyor karışık verilerle", end="\n\n")
    print("-"*80, end="\n\n")

    # Bağımsız değişkenler (X) ve bağımlı değişken (y) ayrımı
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']

    # Veri setini eğitim ve test setlerine bölelim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.239, random_state=42)

    # Kategorik sütunları belirle (örneğin, 'GENDER' gibi)
    cat_features = ['GENDER']

    # CatBoost modelini oluştur
    model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', random_state=42, cat_features=cat_features, verbose=False)

    # Modeli eğitim verisi ile eğit
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)
    model.fit(train_pool, eval_set=test_pool)

    # Test verisi üzerinde modelin performansı
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Model Accuracy with CatBoost: {accuracy}')

    # Diğer performans verileri
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    return model


#------------------- kümeleme ------------------

# kümeleme CART 
def train_catboost_model_mixed_cluster(data):
    print("-"*80, end="\n\n")
    print("Burada şu anda CATBOOST algoritması çalışıyor karışık verilerle kümeleme için", end="\n\n")
    print("-"*80, end="\n\n")

    # Bağımsız değişkenler (X) ve bağımlı değişken (y) ayrımı
    X = data.drop(['LUNG_CANCER', 'Cluster_KMeans'], axis=1)
    y = data['Cluster_KMeans']

    # Veri setini eğitim ve test setlerine bölelim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.239, random_state=42)

    # Kategorik sütunları belirle (örneğin, 'GENDER' gibi)
    cat_features = ['GENDER']

    # CatBoost modelini oluştur
    model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', random_state=42, verbose=False)

    # Modeli eğitim verisi ile eğit
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)
    model.fit(train_pool, eval_set=test_pool)

    # Test verisi üzerinde modelin performansı
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Model Accuracy with CatBoost: {accuracy}')

    # Diğer performans verileri
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    return model


def train_catboost_model_categorical_cluster(data):
    print("-" * 80, end="\n\n")
    print("Burada şu anda CATBOAST algoritması çalışıyor kategorik verilerle kümeleme için", end="\n\n")
    print("-" * 80, end="\n\n")

    # Bağımlı değişken (y) ayrımı
    y = data['Cluster_KMeans']

    # Veri setini eğitim ve test setlerine bölelim
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state=42)

    # Kategorik özellikleri belirt
    cat_features = data.columns.tolist()

    # CatBoost modelini oluştur
    model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', random_state=42, verbose=False)

    # Modeli eğitim verisi ile eğit
    train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
    model.fit(train_pool)

    # Test verisi üzerinde modelin performansı
    test_pool = Pool(X_test, label=y_test, cat_features=cat_features)
    y_pred = model.predict(test_pool)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Model Accuracy with CatBoost: {accuracy}')

    # Diğer performans verileri
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    return model

def train_decision_tree_model_with_numeric_data_cluster(data):
    print("-"*80, end="\n\n")
    print("Burada şu anda DECISION TREE algoritması çalışıyor kümeleme için", end="\n\n")
    print("-"*80, end="\n\n")

    # Bağımsız değişkenler (X) ve bağımlı değişken (y) ayrımı
    X = data.drop('Cluster_KMeans', axis=1)
    y = data['Cluster_KMeans']

    # Veri setini eğitim ve test setlerine bölelim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.229, random_state=42)

    # Decision Tree modelini oluştur
    model = DecisionTreeClassifier(random_state=42)

    # Modeli eğitim verisi ile eğit
    model.fit(X_train, y_train)

    # Test verisi üzerinde modelin performansı
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Model Accuracy with Cluster Data: {accuracy}')

    # Diğer performans verileri
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))








#joblib kütüphanesi ilerde işe yarar  Python nesnelerini dosyalara seri hale getirip kaydetmeye ve daha sonra bu dosyalardan geri yüklemeye olanak tanır. Özellikle büyük veri setlerini veya eğitilmiş makine öğrenimi modellerini kaydetmek ve paylaşmak için yaygın olarak kullanılır.