# main.py

import pandas as pd

from preprocessing import (
    preprocess_numeric_data, 
    balanced_data ,
    preprocess_categorical_and_balance_data, 
    preprocess_categorical_data, 
    clean_and_normalize_data, 
    preprocess_categorical_data_cluster,
    balance_classes,
    add_synthetic_data
    )

from visualization import (
    plot_age_lung_cancer_relation,
    plot_pairplot_by_lung_cancer_status,
    plot_boxplot_by_lung_cancer_status,
    plot_categorical_distributions,
    plot_allergy_wheezing_relation,
    plot_gender_lung_cancer_relation,
    plot_smoking_lung_cancer_relation,
    plot_age_distribution,
    plot_cluster_distribution  # Yeni eklenen görselleştirme
)

from modeling import (
    train_decision_tree_model_with_numeric_data,
    train_catboost_model_categorical,
    train_catboost_model_mixed,
    
    #yeni modeller
    train_catboost_model_categorical_resample,
    train_decision_tree_model_with_balanced_data,
    train_catboost_model_mixed_resample,
    
    #kümelemelerle çalışan modeller
    train_catboost_model_categorical_cluster,
    train_catboost_model_mixed_cluster,
    train_decision_tree_model_with_numeric_data_cluster,
    
    #k-folds modeleri
    train_decision_tree_model_with_numeric_data_kFold,
    train_catboost_model_categorical_kFold,
    train_catboost_model_mixed_kFold
)

def main():
    # Veri setini yükleme ve ön işleme adımları (preprocessing.py'den)
    
    file_path = r'C:\Users\emirh\Desktop\bitirme_projesi\lung_cancer.xlsx'
    cluster_results_file_path = r'C:\Users\emirh\Desktop\bitirme_projesi\kumeleme_sonuclari.xlsx'
    
    #------------------k-fold modelleri sentetik veriyle -------------
    print("-"*100, end="\n\n")
    print("Burada şu anda k-fold modelleri sentetik veriyle çalışıyor", end="\n\n")
    print("-"*100, end="\n\n")
    #sadece numerik
    df_numeric = preprocess_numeric_data(pd.read_excel(file_path))
    df_numeric = add_synthetic_data(df_numeric, 'LUNG_CANCER')
    trained_model_numeric = train_decision_tree_model_with_numeric_data_kFold(df_numeric)
    
    # Sadece kategorik veri işleme
    df_categorical = preprocess_categorical_data(file_path)
    df_categorical = add_synthetic_data(df_numeric, 'LUNG_CANCER')
    trained_model_categorical = train_catboost_model_categorical_kFold(df_categorical)
    
    # Normalizasyon ve temizlik işlemleri
    df_clean_normalized = clean_and_normalize_data(file_path)
    df_clean_normalized = add_synthetic_data(df_numeric, 'LUNG_CANCER')
    trained_model_clean_normalized = train_catboost_model_mixed_kFold(df_clean_normalized)
    """
    #------------------k-fold modelleri-------------
    print("-"*100, end="\n\n")
    print("Burada şu anda k-fold modelleri çalışıyor", end="\n\n")
    print("-"*100, end="\n\n")
    
    #sadece numerik
    df_numeric = preprocess_numeric_data(pd.read_excel(file_path))
    trained_model_numeric = train_decision_tree_model_with_numeric_data_kFold(df_numeric)

    # Sadece kategorik veri işleme
    df_categorical = preprocess_categorical_data(file_path)
    trained_model_categorical = train_catboost_model_categorical_kFold(df_categorical)
    
    # Normalizasyon ve temizlik işlemleri
    df_clean_normalized = clean_and_normalize_data(file_path)
    trained_model_clean_normalized = train_catboost_model_mixed_kFold(df_clean_normalized)
    
    #------------------sentetik veri ile çalışan modeller-------------
    print("-"*100, end="\n\n")
    print("Burada şu anda sentetik veriyle çalışan modelleri çalışıyor", end="\n\n")
    print("-"*100, end="\n\n")
    #sadece numerik
    df_numeric = preprocess_numeric_data(pd.read_excel(file_path))
    df_numeric = add_synthetic_data(df_numeric, 'LUNG_CANCER')
    trained_model_numeric = train_decision_tree_model_with_numeric_data(df_numeric)

    # Sadece kategorik veri işleme
    df_categorical = preprocess_categorical_data(file_path)
    df_categorical = add_synthetic_data(df_numeric, 'LUNG_CANCER')
    trained_model_categorical = train_catboost_model_categorical(df_categorical)

    # Normalizasyon ve temizlik işlemleri
    df_clean_normalized = clean_and_normalize_data(file_path)
    df_clean_normalized = add_synthetic_data(df_numeric, 'LUNG_CANCER')
    trained_model_clean_normalized = train_catboost_model_mixed(df_clean_normalized)
  
    #------------------yeni modeller-------------
    print("-"*100, end="\n\n")
    print("Burada şu anda EMİR modelleri çalışıyor", end="\n\n")
    print("-"*100, end="\n\n")
    #decision tree
    df_numeric = preprocess_numeric_data(pd.read_excel(file_path))
    df_balanced = balanced_data(df_numeric)
    trained_model_balanced = train_decision_tree_model_with_balanced_data(df_balanced)

    #catboost kategorik
    df_categorical = preprocess_categorical_data(file_path)
    trained_model_categorical_resample = train_catboost_model_categorical_resample(df_categorical)

    #catboost karma
    df_clean_normalized = clean_and_normalize_data(file_path)
    df_balanced = balance_classes(df_clean_normalized)
    trained_catboost_model_mixed_resample = train_catboost_model_mixed_resample(df_balanced)

    #------------------eski modeller-------------
    print("-"*100, end="\n\n")
    print("Burada şu anda ESKİ modeller çalışıyor", end="\n\n")
    print("-"*100, end="\n\n")
    # Sadece numerik veri işleme
    df_numeric = preprocess_numeric_data(pd.read_excel(file_path))
    trained_model_numeric = train_decision_tree_model_with_numeric_data(df_numeric)

    # Sadece kategorik veri işleme
    df_categorical = preprocess_categorical_data(file_path)
    trained_model_categorical = train_catboost_model_categorical(df_categorical)

    # Normalizasyon ve temizlik işlemleri
    df_clean_normalized = clean_and_normalize_data(file_path)
    trained_model_clean_normalized = train_catboost_model_mixed(df_clean_normalized)
    """
    """
    # --------------------Kümeleme---------------
    
    print("-" * 80, end="\n\n\n\n\n\n\n\n")
    print("       BURADAN İTİBAREN KÜMELEMELERE GÖRE MODEL SONUÇLARI YER ALIYOR", end="\n\n\n\n\n\n\n\n")
    
    # Sadece numerik veri işleme
    df_numeric = preprocess_numeric_data(pd.read_excel(cluster_results_file_path ))
    trained_model_numeric = train_decision_tree_model_with_numeric_data_cluster(df_numeric)

    # Sadece kategorik veri işleme
    df_categorical = preprocess_categorical_data_cluster(cluster_results_file_path )
    trained_model_categorical = train_catboost_model_categorical_cluster(df_categorical)

    # Normalizasyon ve temizlik işlemleri
    df_clean_normalized = clean_and_normalize_data(cluster_results_file_path )
    trained_model_clean_normalized = train_catboost_model_mixed_cluster(df_clean_normalized)
    """
"""
    # Görselleştirmeler
    plot_age_distribution(df_clean_normalized)
    plot_smoking_lung_cancer_relation(df_clean_normalized)
    plot_gender_lung_cancer_relation(df_clean_normalized)
    plot_allergy_wheezing_relation(df_clean_normalized)
    plot_categorical_distributions(df_clean_normalized)
    plot_age_lung_cancer_relation(df_clean_normalized)
    plot_boxplot_by_lung_cancer_status(df_clean_normalized)
    #Óplot_pairplot_by_lung_cancer_status(df_clean_normalized)
"""

if __name__ == "__main__":
    main()
