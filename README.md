# Makine Öğrenmesi İle Ruh Sağlığı Analizi 
Beslenme bilgilerinin, vücut ağırlığının ve enerji seviyesinin, ruh hali üzerindeki etkisi analiz edilmek üzere bu proje gelişitirilmiştir. Kullanılan veri setine [buradan](https://www.kaggle.com/datasets/inpursuitofnothing/nutrition-vs-weight-mood-energy-dataset) erişebilirsiniz. Bu projede kullanılan makine öğrenmesi algoritmalarından ikisi Supervised Learning algoritmalarından  Linear Regression ve Random Forest, diğer ikisi de Unsupervised Learning algoritmalarından K-Means ve PCA'dır. Gerekli analizlerin yapılması için makine öğrenmesi algoritmalarından önce data processing(veri işleme)  gerçekleştirilmiştir.
## İçindekiler

1. [Veri Seti Hakkında Bilgi](#1-veri-seti-hakkında-bilgi)
2. [Veri Yükleme ve Görüntüleme](#2-veri-yükleme-ve-görüntüleme)
3. [Veri Seti Hakkında Bilgi](#3-veri-seti-özeti)
4. [Veri Setini Anlama ve İşleme](#4-veri-setini-anlama-ve-işleme)
5. [Veri Görselleştirme](#5-veri-görselleştirme)
6. [Korelasyon Matrisi](#6-korelasyon-matrisi)
7. [Kodun İşileyişini Açıklayan Video](#7-kodun-işleyişini-açıklayan-video)
8. [Makine Öğrenmesi Modellerinin Eğitimi ve Skorları](#8-makine-öğrenmesi-modellerinin-eğitimi-ve-skorları)
9. [Skor Değerlendirmesi ve Doğruluk Oranı Artırma Yöntemleri](#9-skor-değerlendirmesi-ve-doğruluk-oranı-arttırma-yöntemleri)
10. [Sertifikalar](#9-sertifikalar)

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

### Açıklama:

