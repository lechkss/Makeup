import cv2
import mediapipe as mp
import numpy as np

# Текстовые сообщения
texts = [
    "Ready to start?"
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

# Настройки шрифта
font = cv2.FONT_HERSHEY_SIMPLEX
org = (150, 50)
font_scale = 1
color = (0, 0, 0)
thickness = 2

# Настройка MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Настройки масок
mask = cv2.imread('mask.png', cv2.IMREAD_UNCHANGED) # Считываем маску


# Камера
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

# MediaPipe Holistic
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    step = 0  # Начальный шаг
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            break

        if step < len(texts):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # Пишем текст
            cv2.putText(bgr_frame, texts[step], org, font, font_scale, color, thickness, cv2.LINE_AA)

            # Преобразуем изображение в RGB для обработки MediaPipe

            # Рисуем ключевые точки лица
            if results.face_landmarks:  # Проверка, что лицо найдено
                # Преобразуем в пиксели (Ключевые точки лица и их координаты переводятся в пиксели из нормализованных координат)
                height, width, _ = frame.shape  # Получаем размеры кадра

                # Преобразуем координаты всех точек лица в пиксели
                landmarks = np.array([(int(landmark.x * width), int(landmark.y * height))
                                      for landmark in results.face_landmarks.landmark])  # Теперь получаем все точки

                # Определяем область для наложения маски по глазам
                left_eye = landmarks[33]  # Левая глаз маски
                right_eye = landmarks[263]  # Правый глаз маски
                nose = landmarks[1]  # Центр носа (Используем для масштабирования)

                # Рассчитываем размеры и позицию маски
                mask_width = int(np.linalg.norm(right_eye - left_eye) * 2)  # ширина маски
                mask_height = int(mask_width * mask.shape[0] / mask.shape[1])  # Пропорциональная ширине высота

                # Масштабируем маску в моменте. Когда лицо там дальше//ближе к камере
                resized_mask = cv2.resize(mask, (mask_width, mask_height))

                # Вычисляем координаты наложения маски
                center_x, center_y = nose  # Центр
                top_left = (center_x - mask_width // 2, center_y - mask_height // 2)  # Верхний левый угол
                bottom_right = (center_x + mask_width // 2, center_y + mask_height // 2)  # Нижний правый угол

                # Наложение маски
                # Проходим по всем пикселям изображения маски
                for i in range(mask_height):
                    for j in range(mask_width):
                        # Проверка границ кадра
                        if top_left[1] + i >= frame.shape[0] or top_left[0] + j >= frame.shape[1]:
                            continue
                        # Получаем прозрачность текущего пикселя маски
                        alpha = resized_mask[i, j, 3] / 255.0  # Альфа-канал
                        # Если альфа-канал больше 0, пиксель будет частично или полностью виден, и его нужно наложить на кадр.
                        # Нам надо отображать маску, а если она полупрозрачная или совсем непрозрачная, то мы меняем цвет пикселя
                        if alpha > 0:
                            frame[top_left[1] + i, top_left[0] + j] = (
                                    alpha * resized_mask[i, j, :3] + (1 - alpha) * frame[
                                top_left[1] + i, top_left[0] + j]
                            )
                cv2.imshow('Makeup Steps', bgr_frame)

            elif not results.face_landmarks:
                cv2.putText(frame, 'No face detected', (50, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            # Пока мы тыкаем на эту кнопу, у нас считаются steps, отображается текст. Пока текст не закончился, выводим текст


        # Показ видео
        cv2.imshow('Makeup Steps', bgr_frame)

        # Управление шагами с клавишами
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):  # Перейти к следующему шагу
            step += 1
        elif key == ord('q'):  # Выход
            break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
