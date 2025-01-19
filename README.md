# Makine Öğrenmesi İle Ruh Sağlığı Analizi 
Beslenme bilgilerinin, vücut ağırlığının ve enerji seviyesinin, ruh hali üzerindeki etkisi analiz edilmek üzere bu proje gelişitirilmiştir. Kullanılan veri setine [buradan](https://www.kaggle.com/datasets/inpursuitofnothing/nutrition-vs-weight-mood-energy-dataset) erişebilirsiniz. Bu projede kullanılan makine öğrenmesi algoritmalarından ikisi Supervised Learning algoritmalarından  Linear Regression ve Random Forest, diğer ikisi de Unsupervised Learning algoritmalarından K-Means ve PCA'dır. Gerekli analizlerin yapılması için makine öğrenmesi algoritmalarından önce data processing(veri işleme)  gerçekleştirilmiştir.
## İçindekiler

1. [Veri Seti Hakkında Bilgi](#1-veri-seti-hakkında-bilgi)
2. [Gerekli Kütüphaneleri Dahil Etme](#2-gerekli-kütüphaneleri-dahil-etme)
3. [Veri Yükleme ve Görüntüleme](#3-veri-yükleme-ve-görüntüleme)
4. [Veri Seti Hakkında Bilgi](#4-veri-seti-özeti)
5. [Veri Setini Anlama ve İşleme](#5-veri-setini-anlama-ve-işleme)
6. [Veri Görselleştirme](#6-veri-görselleştirme)
7. [Korelasyon Matrisi](#7-korelasyon-matrisi)
8. [Kodun İşileyişini Açıklayan Video](#8-kodun-işleyişini-açıklayan-video)
9. [Makine Öğrenmesi Modellerinin Eğitimi ve Skorları](#9-makine-öğrenmesi-modellerinin-eğitimi-ve-skorları)
10. [Skor Değerlendirmesi ve Doğruluk Oranı Artırma Yöntemleri](#10-skor-değerlendirmesi-ve-doğruluk-oranı-arttırma-yöntemleri)
11. [Sertifikalar](#11-sertifikalar)

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

