import numpy as np

# Загружаем файл индексов
indices = np.load('/Users/victorkhudyakov/sx/3DDFA_V2/configs/indices.npy')

# Предположим, что у нас есть описания для этих индексов (например, в списке)
# Пример: описание точек лица (это нужно вручную определить по вашей документации)
face_landmarks = [
    "Нос", "Глаза", "Рот", "Лоб", "Щека",  # Примерные метки
    # Добавьте здесь более точные метки точек
]

# Если метки точек меньше, чем индексов, просто добавляем 'неизвестно'
landmark_descriptions = face_landmarks * (len(indices) // len(face_landmarks)) + ['неизвестно'] * (len(indices) % len(face_landmarks))

# Печатаем индексы с описаниями
print("Индексы и их описание:")
for i in range(len(indices)):
    print(f"Индекс {indices[i]}: {landmark_descriptions[i]}")

# Если вы хотите, можете сохранить это в файл
with open('indices_with_descriptions.txt', 'w') as f:
    for i in range(len(indices)):
        f.write(f"Индекс {indices[i]}: {landmark_descriptions[i]}\n")

print("\nИндексы и их описание сохранены в файл 'indices_with_descriptions.txt'")