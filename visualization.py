import matplotlib.pyplot as plt
import seaborn as sns

def plot_age_distribution(data):
    column_name = 'AGE'
    plt.hist(data[column_name], bins=20, edgecolor='black')
    plt.xlabel('YAŞ')
    plt.ylabel('SAYI')
    plt.title('Yaş Dağılımı')
    plt.show()

def plot_smoking_lung_cancer_relation(data):
    x_column_name = 'SMOKING'
    hue_column_name = 'LUNG_CANCER'
    sns.countplot(x=x_column_name, hue=hue_column_name, data=data)
    plt.xlabel('Sigara')
    plt.ylabel('SAYI')
    plt.title('Sigara ile Akciğer Kanseri İlişkisi')
    plt.show()

def plot_gender_lung_cancer_relation(data):
    x_column_name = 'GENDER'
    hue_column_name = 'LUNG_CANCER'
    sns.countplot(x=x_column_name, hue=hue_column_name, data=data)
    plt.xlabel('Cinsiyet')
    plt.ylabel('SAYI')
    plt.title('Cinsiyet ile Akciğer Kanseri İlişkisi')
    plt.show()

def plot_allergy_wheezing_relation(data):
    x_column_name = 'ALLERGY'
    y_column_name = 'WHEEZING'
    hue_column_name = 'LUNG_CANCER'
    plt.scatter(data[x_column_name], data[y_column_name], c=data[hue_column_name], cmap='viridis')
    plt.xlabel('Alerji')
    plt.ylabel('Hapşırma')
    plt.title('Alerji ve Hapşırma ile Akciğer Kanseri İlişkisi')
    plt.show()

def plot_categorical_distributions(data):
    categorical_columns = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER']

    for column in categorical_columns:
        sns.countplot(x=column, data=data)
        plt.xlabel(column)
        plt.ylabel('SAYI')
        plt.title(f'{column} Dağılımı')
        plt.show()

def plot_age_lung_cancer_relation(data):
    gender_colors = {1: 'blue', 2: 'pink'}  # Örnek renk atamaları, gerçek renklere göre değiştirilebilir
    plt.scatter(data['AGE'], data['LUNG_CANCER'], c=data['GENDER'].map(gender_colors), cmap='viridis')
    plt.xlabel('YAŞ')
    plt.ylabel('Akciğer Kanseri')
    plt.title('Yaş vs. Akciğer Kanseri (Cinsiyet Renkli)')
    plt.show()

def plot_boxplot_by_lung_cancer_status(data):
    numerical_columns = ['AGE']

    for numerical_column in numerical_columns:
        sns.boxplot(x='LUNG_CANCER', y=numerical_column, data=data)
        plt.xlabel('Akciğer Kanseri')
        plt.ylabel(numerical_column)
        plt.title(f'{numerical_column} Akciğer Kanseri Durumuna Göre')
        plt.show()

def plot_pairplot_by_lung_cancer_status(data):
    sns.pairplot(data, hue='LUNG_CANCER', diag_kind='kde')
    plt.suptitle('Sayısal Özelliklerin Akciğer Kanseri Durumuna Göre Çift Grafiği', y=1.02)
    plt.show()

def plot_cluster_distribution(data, labels):
    plt.scatter(data['AGE'], data['LUNG_CANCER'], c=labels, cmap='viridis', alpha=0.5, marker='o')
    plt.xlabel('YAŞ')
    plt.ylabel('Akciğer Kanseri')
    plt.title('Küme Dağılımı')
    plt.show()

