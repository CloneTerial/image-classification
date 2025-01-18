import tensorflow as tf
import os
from train_and_evaluate import load_cifar10_data, train_and_evaluate_model
from preprocess import preprocess_image
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def menu():
    print("Menu:")
    print("1. Training")
    print("2. Prediksi")
    print("0. Keluar")
    choice = input("Pilih opsi (1/2/0): ")
    return choice

def training():
    # Memuat dataset CIFAR-10
    train_images, train_labels, test_images, test_labels = load_cifar10_data()

    # Melatih model dan evaluasi
    model = train_and_evaluate_model(train_images, train_labels, test_images, test_labels, epochs=10)
    print("Training selesai dan model disimpan sebagai 'cifar10_image_classifier.keras'")

def prediksi():
    # Memuat model yang telah dilatih
    if not os.path.exists('cifar10_image_classifier.keras'):
        print("Model tidak ditemukan. Lakukan training terlebih dahulu.")
        return
    
    model = tf.keras.models.load_model('cifar10_image_classifier.keras')

    # Meminta input gambar untuk prediksi
    custom_image_path = input("Masukkan path file gambar (misalnya: 'plane.jpeg'): ")

    # Memeriksa apakah file gambar ada
    if not os.path.exists(custom_image_path):
        print(f"File gambar {custom_image_path} tidak ditemukan.")
        return

    # Preprocessing gambar dan melakukan prediksi
    custom_image = preprocess_image(custom_image_path)

    # Prediksi dengan Model
    prediction = model.predict(custom_image)

    # Menampilkan hasil prediksi
    class_names = ['Pesawat', 'Mobil', 'Burung', 'Kucing', 'Rusa', 'Anjing', 'Kodok', 'Kuda', 'Kapal', 'Truk']
    predicted_class = class_names[np.argmax(prediction)]
    predicted_prob = np.max(prediction)
    
    print(f"Prediksi: {predicted_class} (Akurasi: {predicted_prob:.2f})")

    # Menampilkan Gambar
    img = image.load_img(custom_image_path, target_size=(32, 32))
    plt.imshow(img)
    plt.title(f"Prediksi sebagai: {predicted_class} (Akurasi: {predicted_prob:.2f})")
    plt.axis('off')
    plt.show()

def main():
    while True:
        choice = menu()
        
        if choice == '1':
            # Jalankan Training
            training()
        elif choice == '2':
            # Jalankan Prediksi
            prediksi()
        elif choice == '0':
            # Keluar dari program
            print("!")
            break
        else:
            print("Opsi tidak valid. Silakan pilih lagi.")

if __name__ == "__main__":
    main()