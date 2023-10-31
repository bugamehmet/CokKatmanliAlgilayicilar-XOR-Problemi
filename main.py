import numpy as np

# Giriş verileri
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Çıkış verileri
y = np.array([[0], [1], [1], [0]])

# Ağırlıklar ve bias değerleri
w1_1, w2_1, bias_i = np.random.rand(), np.random.rand(), np.random.rand()
w1_2, w2_2, bias_ii = np.random.rand(), np.random.rand(), np.random.rand()
w_c_i, w_c_ii, bias_cikis = np.random.rand(), np.random.rand(), np.random.rand()

# Sigmoid aktivasyon fonksiyonları
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_turevi(x):
    return x * (1 - x)

# Öğrenme hızı, momentum ve hata eşik değeri
ogrenme_katsayisi = 0.2
momentum_katsayisi = 0.8
sse = 0.03

# Eğitim döngüsü
for iterasyon in range(100):  # İterasyon sayısını artırabilirsiniz
    toplam_hata = 0

    for i in range(len(X)):
        # İleri yayılım (Forward Propagation)
        xi = X[i]

        net1 = (xi[0] * w1_1) + (xi[1] * w2_1) + bias_i
        cikti_1 = sigmoid(net1)

        net2 = (xi[0] * w1_2) + (xi[1] * w2_2) + bias_ii
        cikti_2 = sigmoid(net2)

        net_cikti = (cikti_1 * w_c_i) + (cikti_2 * w_c_ii) + bias_cikis
        agin_ciktisi = sigmoid(net_cikti)

        hata = y[i] - agin_ciktisi
        toplam_hata += hata. np

        delta_cikti = hata * sigmoid_turevi(agin_ciktisi)
        hata_arakatman = delta_cikti * (w_c_i + w_c_ii)  # Düzeltme: Ağırlıkları toplayın

        # Çıktı katmanının ağırlıklarının güncellenmesi
        w_c_i = (ogrenme_katsayisi * delta_cikti * cikti_1) + momentum_katsayisi * w_c_i
        w_c_ii = (ogrenme_katsayisi * delta_cikti * cikti_2) + momentum_katsayisi * w_c_ii
        bias_cikis = (ogrenme_katsayisi * delta_cikti) + momentum_katsayisi * bias_cikis

        # Ara katmanın ağırlıklarının güncellenmesi
        w1_1 = (ogrenme_katsayisi * hata_arakatman * cikti_1 * (1 - cikti_1) * xi[0]) + momentum_katsayisi * w1_1
        w2_1 = (ogrenme_katsayisi * hata_arakatman * cikti_1 * (1 - cikti_1) * xi[1]) + momentum_katsayisi * w2_1
        bias_i = (ogrenme_katsayisi * hata_arakatman * cikti_1 * (1 - cikti_1)) + momentum_katsayisi * bias_i

        w1_2 = (ogrenme_katsayisi * hata_arakatman * cikti_2 * (1 - cikti_2) * xi[0]) + momentum_katsayisi * w1_2
        w2_2 = (ogrenme_katsayisi * hata_arakatman * cikti_2 * (1 - cikti_2) * xi[1]) + momentum_katsayisi * w2_2
        bias_ii = (ogrenme_katsayisi * hata_arakatman * cikti_2 * (1 - cikti_2)) + momentum_katsayisi * bias_ii

    if toplam_hata < sse:
        print(f"Iterasyon {iterasyon}, Toplam Hata: {toplam_hata}")
        break

print("Eğitim Tamamlandı.")
