from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
os.chdir('/content/drive/MyDrive/Colab Notebooks')

# Загрузка датасета с цифрами
digits = load_digits()
X, y = digits.data, digits.target

# Разделение на обучающий и тестовый наборы данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
X_train = X_train / 16.0  # Максимальное значение пикселя в изображении цифры - 16
X_test = X_test / 16.0

# Создание и обучение модели
model = Sequential([
    Dense(64, activation='relu', input_shape=(64,)),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Вывод случайного изображения из тестового набора
random_index = np.random.randint(0, len(X_test))
test_image = X_test[random_index].reshape(8, 8) * 16.0  # Восстановление оригинального значения пикселей
plt.imshow(test_image, cmap='gray')
plt.show()

# Предобработка и распознавание собственного изображения
file_data = Image.open('2right.jpg').convert('L')
width, height = file_data.size
print(f'Ширина: {width}, Высота: {height}')
test_img = np.array(file_data) / 16.0  # Нормализация
test_img_flat = test_img.reshape(1, 64)

result = model.predict(test_img_flat)
predicted_digit = np.argmax(result)

print(f'I think it\'s {predicted_digit}')
