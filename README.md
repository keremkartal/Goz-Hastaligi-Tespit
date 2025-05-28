# Göz Hastalığı Tespit ve Sınıflandırma Sistemi 

Bu proje, yüklenen göz resimlerinden çeşitli göz hastalıklarını (**katarakt, glokom, diyabetik retinopati**) tespit edebilen ve **normal** gözleri ayırt edebilen bir derin öğrenme modelini temel alan web tabanlı bir uygulamadır. Proje, model geliştirme sürecini içeren bir Jupyter Notebook (`odev.ipynb`) ve bu modeli kullanan bir web uygulamasından (FastAPI backend, Docker ile paketlenmiş) oluşmaktadır.

---

## Proje Bileşenleri ve Yapısı

* **/eye\_disease\_app\_project/**: Ana proje klasörü.
    * **app/**: Web uygulaması dosyaları.
        * `main.py`: FastAPI backend uygulaması.
        * `static/index.html`: Kullanıcı arayüzü.
        * `tunnig_model.h5`: Web uygulamasında kullanılan, eğitilmiş en iyi Keras modeli.
    * `Dockerfile`: Uygulamanın Docker yapılandırması.
    * `requirements.txt`: Web uygulaması için Python bağımlılıkları.
    * `odev.ipynb`: Model eğitimi, veri ön işleme, hiperparametre optimizasyonu ve model karşılaştırmalarını içeren Jupyter Notebook.
    * `README.md`: Bu dosya.

---

## 1. Derin Öğrenme Modeli Geliştirme (`odev.ipynb`) 

Bu Jupyter Notebook, göz hastalığı sınıflandırma modellerinin geliştirilme sürecini detaylandırmaktadır.

### Notebook İçeriği ve Akışı:

* **Veri Yükleme ve Ön İşleme**: Göz görüntüleri (katarakt, glokom, diyabetik retinopati, normal) yüklenir, RGB formatına dönüştürülür ve 200x200 piksel boyutuna yeniden ölçeklendirilir.
* **Veri Artırma (Data Augmentation)**: Eğitim verisi çeşitliliğini artırmak için `ImageDataGenerator` ile yatay çevirme gibi teknikler uygulanır ve piksel değerleri [0,1] aralığına normalize edilir.
* **Model Deneyleri**:
    1.  **Base Model (MLP)**: Temel bir Çok Katmanlı Algılayıcı modeli.
    2.  **CNN Model**: Standart bir Evrişimli Sinir Ağı (CNN) modeli. Dropout ve L2 düzenlileştirme içerir.
    3.  **Tuning Model (Optimize Edilmiş CNN)**: Keras Tuner (`Hyperband` algoritması) kullanılarak filtre sayısı, padding türü ve öğrenme oranı gibi hiperparametreleri optimize edilmiş bir CNN modelidir. Optimizer olarak `RMSprop` kullanılmıştır.
* **Eğitim ve Değerlendirme**: Modeller eğitilir, en iyi ağırlıklar kaydedilir ve test seti üzerinde performansları (doğruluk, kayıp) değerlendirilir. Eğitim süreci görselleştirilir.
* **Sonuç**: Üç modelin performansları karşılaştırıldığında, hiperparametre optimizasyonu ile elde edilen **Tuning Model** (`tunnig_model.h5`) en yüksek test doğruluğuna (%87.28) ulaşmıştır ve web uygulamasında kullanılan model budur.

### Temel Kütüphaneler (Notebook):

* **TensorFlow & Keras**: Derin öğrenme modelleri için.
* **Keras Tuner**: Hiperparametre optimizasyonu için.
* **OpenCV (cv2)**: Görüntü işleme için.
* **Matplotlib**: Görselleştirme için.
* **Scikit-learn**: Veri seti bölme işlemleri için.
* **NumPy**: Sayısal işlemler için.

---

## 2. Web Uygulaması (Backend & Frontend) 

Eğitilen en iyi model (`tunnig_model.h5`), kullanıcıların göz resimlerini yükleyip hastalık tahmini alabilecekleri bir web uygulamasına entegre edilmiştir.

### Backend (`app/main.py`):

* **Framework**: Python tabanlı yüksek performanslı **FastAPI** framework'ü kullanılmıştır.
* **API Endpoint'leri**:
    * `GET /`: Kullanıcı arayüzünü (`index.html`) sunar.
    * `POST /predict/`: Kullanıcı tarafından yüklenen göz resmini alır, `tunnig_model.h5` modelini kullanarak hastalık tahmini yapar ve sonucu JSON formatında (hastalık adı ve güven skoru) döndürür.
* **Görüntü İşleme**: API'ye yüklenen görüntüler, modele uygun hale getirilmek üzere yeniden boyutlandırma (200x200), renk formatı dönüşümü (RGB) ve normalizasyon (0-1 aralığı) gibi ön işleme adımlarından geçirilir.

### Docker ile Paketleme (`Dockerfile`):

* Uygulama ve tüm bağımlılıkları, taşınabilir ve tekrarlanabilir bir çalışma ortamı sağlamak amacıyla **Docker** ile paketlenmiştir.
* `python:3.9-slim` temel imajı kullanılmıştır.
* Uygulama, Uvicorn ASGI sunucusu üzerinde 8000 portunda çalışacak şekilde yapılandırılmıştır.

---

## Kurulum ve Çalıştırma 

### Jupyter Notebook (`odev.ipynb`) Kullanımı:

1.  `odev.ipynb` dosyasını Google Colab veya yerel bir Jupyter Notebook ortamında açın.
2.  Gerekli kütüphanelerin (TensorFlow, Keras Tuner, OpenCV vb.) kurulu olduğundan emin olun.
3.  Veri seti yollarını kendi ortamınıza göre düzenleyin (Notebook Google Drive bağlantıları kullanmaktadır).
4.  Hücreleri sırayla çalıştırarak model geliştirme sürecini inceleyebilirsiniz.

### Web Uygulamasını Çalıştırma (Docker ile):

1.  **Gereksinimler**: Docker.
2.  **Projeyi Klonlayın**:
    ```bash
    git clone https://github.com/keremkartal/Goz-Hastaligi-Tespit.git
    cd Goz-Hastaligi-Tespit
    ```
3.  **Docker İmajını Oluşturun**:
    Proje ana dizinindeyken:
    ```bash
    docker build -t eye-disease-detector .
    ```
4.  **Docker Konteynerını Çalıştırın**:
    ```bash
    docker run -d -p 8000:8000 eye-disease-detector
    ```
5.  **Uygulamaya Erişin**:
    Tarayıcınızda `http://localhost:8000` adresine gidin.

---

## Kullanılan Teknolojiler 

* **Derin Öğrenme & Modelleme**: Python, TensorFlow, Keras, Keras Tuner, NumPy, OpenCV, Scikit-learn, Jupyter Notebook
* **Backend Web Geliştirme**: Python, FastAPI, Uvicorn
* **Frontend Web Geliştirme**: HTML, CSS, JavaScript
* **Paketleme & Dağıtım**: Docker
