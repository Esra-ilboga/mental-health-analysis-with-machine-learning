# Makine Öğrenmesi İle Ruh Sağlığı Analizi 
Beslenme bilgilerinin, vücut ağırlığının ve enerji seviyesinin, ruh hali üzerindeki etkisi analiz edilmek üzere bu proje gelişitirilmiştir. Kullanılan veri setine [buradan](https://www.kaggle.com/datasets/inpursuitofnothing/nutrition-vs-weight-mood-energy-dataset) erişebilirsiniz. Bu projede kullanılan makine öğrenmesi algoritmalarından ikisi Supervised Learning algoritmalarından  Linear Regression ve Random Forest, diğer ikisi de Unsupervised Learning algoritmalarından K-Means ve PCA'dır. Gerekli analizlerin yapılması için makine öğrenmesi algoritmalarından önce data processing(veri işleme)  gerçekleştirilmiştir.
## İçindekiler

1. [Veri Seti Hakkında Bilgi](#1-veri-seti-hakkında-bilgi)
2. [Gerekli Kütüphaneleri Dahil Etme](#2-gerekli-kütüphaneleri-dahil-etme)
3. [Veri Yükleme ve Görüntüleme](#3-veri-yükleme-ve-görüntüleme)
4. [Veri Setini Anlama ve İşleme](#4-veri-setini-anlama-ve-işleme)
5. [Veri Görselleştirme](#5-veri-görselleştirme)
6. [Korelasyon Matrisi](#6-korelasyon-matrisi)
7. [Makine Öğrenmesi Modellerinin Eğitimi ve Skorları](#7-makine-öğrenmesi-modellerinin-eğitimi-ve-skorları)
8. [Skor Değerlendirmesi ve Doğruluk Oranı Artırma Yöntemleri](#8-skor-değerlendirmesi-ve-doğruluk-oranı-arttırma-yöntemleri)
9. [Kodun İşileyişini Açıklayan Video](#9-kodun-işleyişini-açıklayan-video)
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
![veridengesizliği](https://github.com/user-attachments/assets/4ed4bb48-6454-402a-99c6-cdb8f9338338)</br>
![datasonhal](https://github.com/user-attachments/assets/be6886a5-cb24-4385-b75b-26c466dc2deb)

## 5. Veri Görselleştirme
Verileri görselleştirmek için histogram ve boxplot kullanıldı. Görüntüleri aşağıda verilmiştir.
### Histogramlar
![histogram](https://github.com/user-attachments/assets/a3f0f4b5-f0b3-46d0-bfa7-f7cd755aebf8)
### Boxplot
![boxplot](https://github.com/user-attachments/assets/1f5f0a54-66a4-49a3-be7a-be7a3efd1e26)

## 6. Korelasyon Matrisi
![korelasyon](https://github.com/user-attachments/assets/d750f848-95d5-4dce-8ce9-d8a2c78b2653)

## 7. Makine Öğrenmesi Modellerinin Eğitimi ve Skorları
Aşağıda adım adım uygulanan algoritmalar kendi başlıkları altında verilmiştir.</br>
### Random Forest
Random Forest, karar ağaçlarından (decision trees) oluşan bir topluluk yöntemidir. Birden fazla karar ağacı eğitilir ve sonuçlar oylama (sınıflandırma) veya ortalama (regresyon) yöntemiyle birleştirilir. Bu algoritmayı kullanarak aşağıdaki kod bloğu yazıldı: 
```python
# Bağımlı ve bağımsız değişkenlerin ayrılması
X = data.drop(columns=["Mental_Health"])
y = data["Mental_Health"]

# Eğitim ve test setine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Modeli
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Performans Değerlendirmesi
print("Random Forest")
print(classification_report(y_test, y_pred_rf))
```
+ kodun çıktısı aşağıdaki görselde verilmiştir.
![random](https://github.com/user-attachments/assets/1adb4514-7688-4080-b184-7857fc311eab)

### Linear Regression
Lineer regresyon, bağımlı bir değişken (y) ile bir veya daha fazla bağımsız değişken (x) arasındaki doğrusal ilişkiyi modellemek için kullanılan bir yöntemdir.  Bu algoritmayı kullanarak aşağıdaki kod bloğu yazıldı: 
```python
# Bağımlı ve bağımsız değişkenlerin ayrılması
X = data.drop(columns=["Mental_Health"])
y = data["Mental_Health"]

# Eğitim ve test setine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear Regression Modeli
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
# Performans Değerlendirmesi
mse = mean_squared_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)
print("Linear Regression")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```
+ kodun çıktısı görselde verilmiştir.
![lineer](https://github.com/user-attachments/assets/6f2f5569-a775-49e7-98ad-d993dfd51a59)

### K-Means
 K-Means, bir veri setini önceden belirlenmiş 𝐾 sayıda kümeye ayıran bir kümeleme algoritmasıdır. Her bir nokta, en yakın küme merkezine (centroid) atanır. Bu algoritma için öncelikle uygun küme sayısı bulunmak için aşağıdaki kod bloğu yazıldı: 
 ```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

inertias = []
silhouettes = []
# Burdaki range değerleri sırasıyla (10, 20), (21, 31), (40, 80) şekilne denendi
for k in range(10, 20):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(scaled_data, kmeans.labels_))

# Grafikleri çizdirme
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(10, 20), inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("Küme Sayısı")
plt.ylabel("Inertia")

plt.subplot(1, 2, 2)
plt.plot(range(10, 20), silhouettes, marker='o')
plt.title("Silhouette Analysis")
plt.xlabel("Küme Sayısı")
plt.ylabel("Silhouette Score")
plt.show()
```
Ardından uygun K (küme sayısı) değeri 11 olarak belirlendi ve ona göre aşağıdaki kod bloğu uygulandı: 
```python
# Veri ön işleme (ölçekleme)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# K-Means modelini eğitme
kmeans = KMeans(n_clusters=11, random_state=42)  # Küme sayısı: 11
kmeans.fit(scaled_data)

# Kümelere atanan etiketler
data["Cluster"] = kmeans.labels_

#  Performans analizi
# a) Inertia (toplam hata kareleri toplamı)
inertia = kmeans.inertia_

# b) Silhouette Score (kümelerin ayırt edilebilirliği)
silhouette_avg = silhouette_score(scaled_data, kmeans.labels_)

# Sonuçları görüntüleme
print("Kümeleme Sonuçları:")
print(data)
print("\nPerformans Analizi:")
print(f"Inertia (Hata Kareleri Toplamı): {inertia:.2f}")
print(f"Silhouette Score: {silhouette_avg:.2f}")
```
+ Kodların çıktısı aşağıdaki verilmiştir.
![kume1](https://github.com/user-attachments/assets/5ec4f106-d652-4bd3-99fe-123146963cc2)</br>
![kume2](https://github.com/user-attachments/assets/837105b6-fc42-40ab-a4fd-f8b6ef2e1321)</br>
![kume3](https://github.com/user-attachments/assets/572c6fe9-4e19-4d44-b079-4984033b2945)</br>
![performanss](https://github.com/user-attachments/assets/79315384-9256-4867-addf-321626bb3465)

### PCA 
PCA, yüksek boyutlu veriyi daha düşük boyutlara indirerek ana bileşenleri (principal components) çıkaran bir boyut indirgeme yöntemidir. Verideki değişkenlik (varyans) korunmaya çalışılır.  Bu algoritmayı kullanarak aşağıdaki kod bloğu yazıldı: 
```python
#  Veriyi Ölçeklendirme (Standartlaştırma)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

#  PCA Algoritmasını Uygulama
pca = PCA(n_components=None)  # Tüm bileşenleri çıkar
pca_data = pca.fit_transform(scaled_data)

#  Performans Analizi - Açıklanan Varyans Oranı
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

#  Açıklanan Varyans Oranını Görselleştirme
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center', label='Individual Variance')
plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid', label='Cumulative Variance')
plt.xlabel('Temel Bileşen İndeksi')
plt.ylabel('Açıklanan Varyans Oranı')
plt.title('Temel Bileşenlere Göre Açıklanan Varyans')
plt.legend(loc='best')
plt.grid(True)
plt.show()

#  En Önemli Bileşenleri Belirleme
# Örneğin, %95 açıklanan varyansa ulaşmak için gereken bileşen sayısı
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"%95 varyansı açıklamak için gerekli bileşen sayısı: {n_components_95}")

#  PCA ile Düşük Boyutlu Veriyi Elde Etme
# %95 varyansı açıklayan bileşenlerle yeniden PCA uygula
pca_optimized = PCA(n_components=n_components_95)
reduced_data = pca_optimized.fit_transform(scaled_data)

#  Sonuçları Görselleştirme (2D Örneği)
if n_components_95 >= 2:
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7, cmap='viridis')
    plt.title('PCA Düşük Boyutlu Görselleştirme (2 Bileşen)')
    plt.xlabel('Birinci Temel Bileşen')
    plt.ylabel('İkinci Temel Bileşen')
    plt.grid(True)
    plt.show()
```
+ kodun çıktısı aşağıdaki görselde verilmiştir.
![PCA1](https://github.com/user-attachments/assets/349cb897-b965-458e-ac69-3ee5762f09c2)</br>
![PCA2](https://github.com/user-attachments/assets/4fc7f261-a6ff-4872-8f3a-967548436995)

Bu algoritmanın performans değerlendirmesi için Aşağıdaki iki kod yazıldı:
```python
# Orijinal boyuta yeniden yapılandırma
reconstructed_data = pca_optimized.inverse_transform(reduced_data)

# Yeniden yapılandırma hatası (Mean Squared Error)
reconstruction_error = np.mean((scaled_data - reconstructed_data) ** 2)
print(f"Yeniden Yapılandırma Hatası: {reconstruction_error:.4f}")
#--------------------------------------------------------------
# Kümeleme
kmeans = KMeans(n_clusters=11, random_state=42)
kmeans.fit(reduced_data)
labels = kmeans.labels_

# Silhouette Score hesaplama
silhouette_avg = silhouette_score(reduced_data, labels)
print(f"Silhouette Score: {silhouette_avg:.2f}")
```
+ kodun çıktıları görselde verilmiştir.
![PCA4](https://github.com/user-attachments/assets/d0101922-4ba9-4e53-9f63-dcb5912e9a4b)
![PCA5](https://github.com/user-attachments/assets/68cb3d79-dae7-40a9-9314-b8adb9c35f91)

## 8.Skor Değerlendirmesi ve Doğruluk Oranı Artırma Yöntemleri
### Linear Regression Performans Değerlendirmesi
#### Performans
Mean Squared Error (MSE) 0.0829, modelin tahminleri ile gerçek değerler arasındaki farkların karesinin ortalamasının düşük olduğunu, ancak hala hata olduğunu gösteriyor. R-squared (R²) değeri ise 0.4777, modelin verinin %47.77'sini doğru tahmin edebildiğini belirtir, bu da doğrusal regresyon için ortalama bir performansı ifade eder ve daha iyi sonuçlar elde edilebileceğini gösterir.
#### Yorum
Linear Regression, verinin doğrusal ilişkilerini öğrenme konusunda başarılı olsa da, R² değeri, modelin yeterince güçlü olmadığına işaret ediyor. Bu nedenle, doğrusal ilişkilerin güçlü olduğu veri setlerinde bile daha iyi sonuçlar elde edilebilir.
### Random Forest Performans Değerlendirmesi
#### Performans
Modelin Precision, Recall, F1-Score ve Accuracy değerleri tüm sınıflar için 1.00 olup, bu da modelin her iki sınıfı da tamamen doğru şekilde sınıflandırdığını ve yanlış pozitif ya da yanlış negatif oranlarını sıfıra yakın tuttuğunu gösterir. Ayrıca, Accuracy'nin %100 olması, modelin tüm tahminlerinde mükemmel bir performans sergilediğini belirtir.
#### Yorum
 Random Forest modeli burada mükemmel bir performans sergiliyor. Precision, recall ve F1-score değerlerinin yüksekliği, modelin her iki sınıfı da doğru bir şekilde tahmin ettiğini ve   1.00'lık sonuçları bazen gerçek dünya verilerinde overfitting’e işaret edebileceğini gösteriyor bu nedenle modelin test verisiyle doğrulanması önemlidir
 ### K-Means Performans Değerlendirmesi
#### Performans
K-Means algoritmasının Inertia değeri 435330.75, kümelerin içindeki örneklerin merkezine olan mesafelerin karesinin toplamının yüksek olduğunu ve bu durumun kümeleme kalitesinin düşük olduğunu gösteriyor. Silhouette Score ise 0.27, kümeler arasındaki ayrımın zayıf olduğunu ve kümeleme kalitesinin yetersiz olduğunu işaret eder, bu da modelin başarılı olmadığına dair bir gösterge sunar.
#### Yorum
K-Means'in inertia değeri ve düşük Silhouette Score değeri, modelin veri setindeki yapıyı çok iyi öğrenmediğini ve kümeler arasında net ayrımlar oluşturmakta zorlandığını gösteriyor. Bu durumda, K-Means algoritması daha iyi sonuçlar verebilmesi için parametre ayarlarına veya başka bir kümeleme algoritmasına ihtiyaç duyabilir.
### PCA Performans Değerlendirmesi
#### Performans
PCA modelinin Yeniden Yapılandırma Hatası 0.0335, boyut indirgeme işlemi sonrası orijinal verilere oldukça yakın bir temsil oluşturulduğunu ancak biraz bilgi kaybı yaşandığını gösteriyor. Silhouette Score'un 0.24 olması, PCA'nın elde edilen düşük boyutlu verilere uygun bir şekilde kümeleme yapmada etkili olmadığını ve kümeleme kalitesinin düşük olduğunu işaret eder.
#### Yorum 
PCA, veriyi daha düşük boyutlarda temsil etmede başarılı olsa da, düşük Silhouette Score değeri, boyut indirgeme işleminden sonra kümeler arasında çok belirgin bir ayrım olmadığını gösteriyor. Bu, PCA'nın daha çok veriyi görselleştirme veya özellik seçimi için kullanılması gerektiğini gösteriyor.
### Genel Sonuç ve Değerlendirme
Random Forest, yüksek precision, recall, F1-score ve accuracy değerleriyle mükemmel bir sınıflandırma başarısı sergileyerek bu dört algoritma arasında en iyi performansı gösteriyor. Linear Regression, doğrusal olmayan verilerle çalışırken sınırlı bir performans sergileyip R²'nin 0.4777 olmasıyla yalnızca veri setinin yarısından biraz fazlasını açıklayabiliyor. K-Means, düşük Silhouette Score değeriyle kümelenmiş veri setlerinde başarılı olamayarak parametre ayarlarının ya da farklı bir kümeleme yönteminin gerekebileceğini gösteriyor. PCA ise boyut indirgeme ve görselleştirme için faydalı olsa da kümeleme başarısı sınırlıdır. 
Bu verilere dayanarak Random Forest algoritması, diğerlerinden çok daha iyi sonuçlar veriyor ve en iyi performansı sergileyen modeldir.

## 9. Kodun İşileyişini Açıklayan Video
Projeyi kısaca açıkladığım youtube videosuna erişmek için [YotubeLinki](https://youtu.be/72kCtWODcto) yazan yere tıklayınız.

## 10. Sertifikalar











