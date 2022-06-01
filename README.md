# Brain-Tumor-Detection--Machine-Learning-Desicion-Tree-
Beyin tümörü tespiti için doku özelliği çıkartımı ile birlikte tespiti gerçekleştirilmiştir veri setimiz Kaggle den alınan bir veri setidir linki aşağıda belirtilmiştir.

Beyin Tümörü veriler: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection?resource=download

Yöntem:

Veri Seti:
Çalışmada 3000 Beyin MR görüntüsü kullanılmıştır. Bunlardan 1500'ü sağlıklı iken 1500'ü ise tümörlü beyinin MR görüntüsüdür.

Ön İşleme:
İlk olarak, görüntülerin boyutu, işlenmelerini kolaylaştırmak için 500x500 piksele ayarlanmıştır. Bir görüntü, her pikselin bir öğesi olduğu, 0 ile 255 arasında değişen (gri tonlamalı) sayısal bir değerle temsil edilen bir m × n matrisinden oluşmaktadır. 
Görüntülerden segmentasyon için görüntü işleme algoritmalarından bazıları kullanılmıştır. Bunun sonucunda MR görüntülerimizde tümörlü bölgeler açığa çıkarılmıştır 

Özellik Çıkarımı:
Özellik çıkarımı için GLCM doku özelliklerini kullandık.
zıtlık, farklılık, homojenlik, ASM, enerji ve korelasyon değerlerini alarak makine öğrenmesi algoritmamıza verdik.

Gray-Level Co-Occurrence Matrix (GLCM):
 Gri Seviye Eş Oluşum Matrisleri, görüntülerin doku özniteliklerinin çıkarılması için kullanılan istatistiksel bir yöntemdir. Gri seviye eş oluşum matrisleri hesaplanırken iki piksel arasındaki ilişki, uzaklık ve yön bilgisine bağlı olarak belirlenir. Tipik olarak bu yönler, 0 ˘ ◦ , 45◦ , 90◦ , 135◦ şeklinde alınabilir 
Doku hesaplamasında GLCM için görüntü öncelikle aşağıdaki formüle göre normalize hale dönüştürülür.
 

 ![1](https://user-images.githubusercontent.com/61785942/171360670-b20ede6f-33d0-4317-9f60-89c943f659e4.png)

 ![2](https://user-images.githubusercontent.com/61785942/171360825-4ed5d749-1cc8-4b62-a41d-5d572c917dea.png)

Her bir komşu ilişki için farklı bir yeniden oluşum matrisi mevcuttur. Yöney sabitesi elde etmek için, doku hesaplamasından önce 4 farklı yönün (0°, 45°, 90°, 135°) hepsi toplanır. 0°’lik açı yatay yönü, 90°’lik açı ise dikey yönü temsil eder.
 
 
![3](https://user-images.githubusercontent.com/61785942/171360908-0bc14238-a0a3-43bf-8806-f9b69097f3ef.png)

![4](https://user-images.githubusercontent.com/61785942/171361033-5798b555-aa16-4aba-8d15-591b85049bc1.png)


GLCM Homojenlik 
Bir hücrenin belli bölgelerinde homojenlik görünüyor ve GLCM değerleri de köşelerde yoğunlaşıyorsa bu hücrenin renk değeri yüksektir. Homojen alanların renk değerleri ile heterojen alanların renk değerleri bir birine zıtlık göstermektedir. Değerler heterojen alanlardan homojen alanlara doğru azalarak gitmektedir. Hesaplama için parametreler şu şekildedir:

 ![5](https://user-images.githubusercontent.com/61785942/171361105-98364d4c-1ee6-46ac-98f9-1d4fb7b8af55.png)

İ=0, j=0   :  2/(1+(0-0)2)  = 2
İ=0, j = 1 :  2/(1+(0-1)2)  = 1
İ=0, j=3  : 1/(1+(0-3)2)  =1/10 

Homojenlik =  2 + 1 + 1/10 …


GLCM Zıtlık
Zıtlık homojenliğin tam tersi olan heterojenliktir. Zıtlık görüntüde lokal değişim miktarının belirlenmesi yolu ile tespit edilir. Zıtlığın artmasına paralel olarak satır (i) ve sütun (j) sayısı da artış gösterir. Hesaplama için parametreler şu şekildedir:

 ![6](https://user-images.githubusercontent.com/61785942/171361177-4dd5d218-e444-4654-a704-b876a9e2b25d.png)


GLCM Farklılık 
Farklılık zıtlık kavramıyla yakın anlam taşımasına rağmen elde edilen değerlerin doğrusal bir artış göstermesi bakımından farklılık gösterir. Lokal değişim ne kadar fazla ise GLCM farklılığı da o kadar yüksektir. Hesaplama için parametreler şu şekildedir:

 ![7](https://user-images.githubusercontent.com/61785942/171361227-954d53ec-c7f7-440a-a6b7-85f89e6e6623.png)


GLCM Ortalama 

GLCM ortalaması adından da anlaşılacağı üzere GLCM değerlerinin ortalamasıdır. İşlem sonucunda hücre değeri, pikselin görüntüde bulunma oranına göre değil bu piksellerin komşu piksellerle olan kombinasyonlarına göre derecelendirilir. Hesaplama için parametreler şu şekildedir:

 ![8](https://user-images.githubusercontent.com/61785942/171361270-8f154bca-7fda-4e4b-8336-1f8fbef41dca.png)

GLCM Standart Sapma 
GLCM değerleri kullanılarak GLCM Standart Sapması hesaplanır. Bu durum, özellikle komşu pikseller ve bu komşu piksellerin kombinasyonları ile alakalıdır. Bu özelliğinden dolayı orijinal görüntüdeki gri düzeyin basit standart sapması ile farklılık arz etmektedir. GLCM simetrik olduğu için i ya da j değerleri kullanılarak standart sapmanın hesaplanması durumunda da aynı sonuçlar bulunacaktır. Standart sapma ortalamaya yakın değerlerin dağılımını ifade ettiği için zıtlık ve benzemezlik ile benzerlikler göstermektedir. Hesaplama için parametreler şu şekildedir:
 

![9](https://user-images.githubusercontent.com/61785942/171361320-ccd5922d-e7fa-4e06-94cf-a1050b89162f.png)


GLCM Korelasyon 
Gri renk düzeyindeki komşu piksellerin doğrusal bağımlılıklarını ölçmek için kullanılır. Hesaplama için parametreler şu şekildedir:


![10](https://user-images.githubusercontent.com/61785942/171361375-3e90f056-1b50-4d3f-83f4-83dc28472d5e.png)

