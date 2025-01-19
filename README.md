# Makine Öğrenmesi İle Ruh Sağlığı Analizi 
Beslenme bilgilerinin, vücut ağırlığının ve enerji seviyesinin, ruh hali üzerindeki etkisi analiz edilmek üzere bu proje gelişitirilmiştir. Kullanılan veri setine [buradan](https://www.kaggle.com/datasets/inpursuitofnothing/nutrition-vs-weight-mood-energy-dataset) erişebilirsiniz. Bu projede kullanılan makine öğrenmesi algoritmalarından ikisi Supervised Learning algoritmalarından  Linear Regression ve Random Forest, diğer ikisi de Unsupervised Learning algoritmalarından K-Means ve PCA'dır. Gerekli analizlerin yapılması için makine öğrenmesi algoritmalarından önce data processing(veri işleme)  gerçekleştirilmiştir.
## İçindekiler

1. [Veri Seti Hakkında Bilgi](#1-veri-seti-hakkında-bilgi)
2. [Gerekli Kütüphaneleri Dahil Etme](#2-gerekli-kütüphaneleri-dahil-etme)
3. [Veri Yükleme ve Görüntüleme](#3-veri-yükleme-ve-görüntüleme)
4. [Veri Setini Anlama ve İşleme](#4-veri-setini-anlama-ve-işleme)
5. [Veri Görselleştirme](#5-veri-görselleştirme)
6. [Korelasyon Matrisi](#6-korelasyon-matrisi)
7. [Kodun İşileyişini Açıklayan Video](#7-kodun-işleyişini-açıklayan-video)
8. [Makine Öğrenmesi Modellerinin Eğitimi ve Skorları](#8-makine-öğrenmesi-modellerinin-eğitimi-ve-skorları)
9. [Skor Değerlendirmesi ve Doğruluk Oranı Artırma Yöntemleri](#9-skor-değerlendirmesi-ve-doğruluk-oranı-arttırma-yöntemleri)
10. [Sertifikalar](#10-sertifikalar)

## 1. Veri Seti Hakkında Bilgi
Veri seti, 9 sütun ve 100000 satırdan oluşmaktadır. Bu sütunlar; **Product Name** , **Calories** , **Body Type** , **Mood** , **Energy**, **Total Fat** , **Total Sugars**, **Carbohydrates (Carbs)** ve **Protein**'dir. Bu sütunlar aşağıdaki tabloda detaylandırılmıştır:

| **Sütun Adı**              | **Veri Tipi** | **Açıklama**                                                                  |
|----------------------------|---------------|--------------------------------------------------------------------------------|
| `Product Name`             | Kategorik     | Ürün İsimleri.                                                                |
| `Calories`                 | Sayısal       | Ürünün kalori miktarı.                                                        |
| `Body Type`                | Kategorik     | İnsan vücut ağırlığı ('Fat', 'Balanced', 'Slim').                             |
| `Mood`                     | Kategorik     | İnsan ruh hali (modu) ('Neutral', 'Sad', 'Happy').                            |
| `Energy`                   | Kategorik     | Enerji miktarı ('Energy Burst', 'Low', 'Normal').                             |
| `Total Fat`                | Kategorik     | Toplam yağ (g cinsinden).                                                     |
| `Total Sugars`             | Kategorik     | Toplam şeker (g cinsinden).                                                   |
| `Carbohydrates (Carbs)`    | Kategorik     | Karbonhidrat miktarı (g cinsinden).                                           |
| `Protein`                  | Kategorik     | Protein miktarı (g cinsinden).                                                |

## 2. Gerekli Kütüphaneleri Dahil Etme
Projede veriyi işlemek, algoritmaları uygulamak, doğruluk değerlendirmelerini yapmak ve görselleştirmeler yapabilmek için bazı kütüphaneler dahil etmeliyiz. 

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

## 3. Veri Yükleme ve Görüntüleme
Veriyi projeye dahil etmek ve görüntülemek için aşağıdaki kodlar yazılır:
```python
# Veri seti yükleme
data = pd.read_csv('nutrition_labels.csv')

# Veri setinin ilk beş satırını tablo halinde görme 
data.head()

# Veriler hakkında detaylı bilgi alma
data.info()

# null değerler var mı? köntrolünü sağlar
data.isnull().sum()
```
### Ekran Görüntüleri 
![Verisetibilgi](https://github.com/user-attachments/assets/387eeee8-a361-472e-898b-eb4c95227c1b)

## 4. Veri Setini Anlama ve İşleme
Verileri anladıktan sonra bu verileri istenen şekilde yani daha kullanılabilir hale getirmek gerekmektedir. Eksik veriler silinmeli veyahut eksik verileri bulundukları sütundaki verilerin medyanıyla doldurmalıyız. Object(kategorik) veriler sayısal verilere dönüştürülmeli. Veri dengesizlikleri varsa veriler dengeli hale getirilmeli. Gereksiz sütunlar varsa çıkarılmalı.Aşağıda bu işlemleri gerçekleştirdiğim kodlar yazmaktadır:
```python
# Veri setindeki eksik değerleri, sütunların medyan değerleriyle doldurur.
data_filled = data.fillna(data.median())
#---------------------------------------------------------------
# Belirtilen sütunlardaki değerlerden 'g' harfini kaldırır ve veriyi float türüne dönüştürür.
for column in ["Total Fat", "Total Sugars", "Carbohydrates (Carbs)", "Protein"]:
    data[column] = data[column].str.replace("g", "").astype(float)
#---------------------------------------------------------------
# 'Product Name' sütununu veri setinden çıkarır, her satır için benzersiz bir 'Product ID' oluşturur
# ve 'Product ID' sütununu veri setinin ilk sütunu olacak şekilde yeniden sıralar.
data = data.drop("Product Name", axis=1)
data["Product ID"] = range(1, len(data) + 1)
data = data[["Product ID"] + [col for col in data.columns if col != "Product ID"]]
print("Yeni Veri Seti:")
print(data)
#---------------------------------------------------------------
data = pd.DataFrame(data)

# Kategorik sütunları seçme
categorical_columns = data.select_dtypes(include=['object']).columns

# One-Hot Encoding uygulama ve veri çerçevesini güncelleme
data = pd.get_dummies(data, columns=categorical_columns)

boolean_columns = data.select_dtypes(include=['bool']).columns
for column in boolean_columns:
    data[column] = data[column].astype(int)

# Güncellenmiş veri setini görüntüleme
data.head()
#--------------------------------------------------------------
data = pd.DataFrame(data)

# Ruh sağlığı sütununu oluşturma
data["Mental_Health"] = ((data["Mood_Happy"] == 1) | (data["Mood_Neutral"] == 1)).astype(int)
#---------------------------------------------------------------
# 'Mood_Happy', 'Mood_Neutral' ve 'Mood_Sad' sütunlarını veri setinden kaldırır 
# çünkü bu sütunlardaki bilgiler daha önce 'Mental_Health' sütununda özetlenmiştir.
data = data.drop(columns=["Mood_Happy", "Mood_Neutral", "Mood_Sad"])
data.head()
#--------------------------------------------------------------
# 'Product ID' sütununu veri setinden kaldırır 
# çünkü bu sütun analiz veya modelleme için gerekli değildir.
data.drop(columns=['Product ID'], inplace=True)
#--------------------------------------------------------------
# 'Mental_Health' sütunundaki sınıf dağılımını normalleştirilmiş olarak hesaplar
# ve bu dağılımı yazdırır.
class_distribution = data['Mental_Health'].value_counts(normalize=True)
print("Ruh sağlığı oranları: ")
print(class_distribution)

# 'Mental_Health' sütunundaki sınıfların görsel olarak dağılımını gösteren bir countplot çizer.
plt.figure(figsize=(8, 6))
sns.countplot(x='Mental_Health', data=data)
plt.title("Ruh sağlığı dağılımı")
plt.show()
#--------------------------------------------------------------
# Bağımsız ve bağımlı değişkenleri ayır
X = data.drop('Mental_Health', axis=1)
y = data['Mental_Health']

# SMOTE ile oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Yeni sınıf dağılımını kontrol et
print("Yeni Ruh Sağlığı Dağılımı:")
print(y_resampled.value_counts())
```
### İlgili Görseller ve Veri Seti Son HAli
![image](https://github.com/user-attachments/assets/4ed4bb48-6454-402a-99c6-cdb8f9338338)</br>
![image](https://github.com/user-attachments/assets/be6886a5-cb24-4385-b75b-26c466dc2deb)

























