import cv2  # Библиотека для работы с видео и изображениями
import mediapipe as mp  # Крутая штука для распознавания лица
import numpy as np  # Это для всякой математики


# Это список сообщений, которые будут показываться
texts = [
    "Ready to start?",
    "Put your face cream on",
    "Now put your primer on",
    "Now, Foundation",
    "Now Contour",
    "Now Concealer",
    "Now Blushes",
    "Setting Powder",
    "Now, Mascara",
    "Lip liner",
    "Lipstick",
    "Lip gloss"
]

# Настройки для текста (чтобы красиво писать на экране)
font = cv2.FONT_HERSHEY_SIMPLEX # Это шрифт я не знаю других шрифтов
org = (150, 50)  # Координаты, где будет текст
font_scale = 1  # Размер текста
color = (0, 0, 0)  # Цвет текста
thickness = 2  # Толщина текста

# Настраиваем MediaPipe (чтобы находить лицо)
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Загружаем маски (у нас их 12 штук, типа для каждого шага)
mask_paths = [f'mask{i}.png' for i in range(1, 13)]  # Названия файлов mask1.png, mask2.png и так далее
masks = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in mask_paths]

# Проверяем, загрузились ли маски
for i, mask in enumerate(masks):
    if mask is None:
        print(f"Error: Не получилось загрузить маску номер {mask_paths[i]}")
        exit()

# Включаем камеру (Мы в кино)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: Камера не работает")
    exit()

# Тепреь если короче минимальная конфиденс 0.5, то мы будем отображать маску, если нет, то нет
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    step = 0  # Чтобы считать кадры
    while True:
        ret, frame = cap.read()  # Считываем один кадр из камеры
        if not ret: # Потом удалю
            print("Ошибка: Не получилось прочитать кадр")
            break

        if step < len(texts):  # Если шаг еще не последний
            # Пишем текст на экране
            cv2.putText(frame, texts[step], org, font, font_scale, color, thickness, cv2.LINE_AA)

            # Конвертируем кадр в ргб, чтобы MediaPipe мог найти лицо
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)  # Тут лицо находится в этой переменной мы собрали точки лица

            # Если лицо нашлось
            if results.face_landmarks:
                # Берем размеры картинки
                height, width, _ = frame.shape # Там нижнее подчеркивание, потому что там цветовые каналы

                # Ключевые точки лица (они помогают найти глаза, нос и т.д.)
                landmarks = np.array([(int(landmark.x * width), int(landmark.y * height)) for landmark in results.face_landmarks.landmark])

                # Берем координаты глаз и носа (на этом строится маска)
                left_eye = landmarks[33]
                right_eye = landmarks[263]
                nose = landmarks[1]

                # Берем текущую маску
                mask = masks[step]
                mask_width = int(np.linalg.norm(right_eye - left_eye) * 2)  # Маска по ширине лица
                mask_height = int(mask_width * mask.shape[0] / mask.shape[1])  # Высота с сохранением пропорций
                resized_mask = cv2.resize(mask, (mask_width, mask_height))  # Изменяем размер маски

                # Считаем верхний левый угол, в котором будет стоять наша будущая маска
                center_x, center_y = nose
                top_left = (center_x - mask_width // 2, center_y - mask_height // 2)

                # Накладываем маску (тут что-то с альфа-каналом, с альфа каналом помог gpt)
                for i in range(mask_height):
                    for j in range(mask_width):
                        y = top_left[1] + i
                        x = top_left[0] + j
                        if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:  # Проверяем, не вылезли ли за картинку
                            alpha = resized_mask[i, j, 3] / 255.0  # Альфа-канал (прозрачность)
                            if alpha > 0:  # Если маска не прозрачная
                                frame[y, x] = (alpha * resized_mask[i, j, :3] + (1 - alpha) * frame[y, x])

            else:  # Если лица нет, пишем сообщение
                cv2.putText(frame, 'No face detected', (50, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Показываем видео
        cv2.imshow('Makeup Steps', frame)

        # Управление шагами
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):  # Кнопка "1" переключает шаги
            step = (step + 1) % len(texts)  # После последнего шага начинаем с первого
        elif key == ord('q'):  # Кнопка "q" для выхода
            break

# Выключаем камеру и закрываем окно
cap.release()
cv2.destroyAllWindows()
