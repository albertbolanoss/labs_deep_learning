# -*- coding: utf-8 -*-
"""
@author: abhilash
Refactored for performance and compatibility
"""

import numpy as np
import cv2

# --- SECCIÓN 1: CONFIGURACIÓN E INICIALIZACIÓN (Hacer esto UNA sola vez) ---

# Cargar etiquetas de clase
class_labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                    "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                    "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                    "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                    "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                    "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                    "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

# Configurar colores aleatorios para cada clase
class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255"]
class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
class_colors = np.array(class_colors)
class_colors = np.tile(class_colors,(16,1))

print("Cargando modelo YOLO... esto puede tardar unos segundos.")
# Cargar el modelo preentrenado (FUERA del bucle)
yolo_model = cv2.dnn.readNetFromDarknet('models/yolov3.cfg','models/yolov3.weights')

# Obtener las capas de salida (Corrección del IndexError)
yolo_layers = yolo_model.getLayerNames()
# La corrección clave: iteramos directamente sobre el resultado, sin buscar índice [0]
yolo_output_layer = [yolo_layers[i - 1] for i in yolo_model.getUnconnectedOutLayers()]

# Cargar video
file_video_stream = cv2.VideoCapture('images/video_sample.mp4')
print("Iniciando detección...")

# --- SECCIÓN 2: PROCESAMIENTO DEL VIDEO ---
# Crear la ventana con nombre
# cv2.namedWindow("Detection Output", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Detection Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.namedWindow("Detection Output", cv2.WINDOW_NORMAL) # WINDOW_NORMAL permite redimensionar
cv2.resizeWindow("Detection Output", 1080, 1920)

while file_video_stream.isOpened(): # Corrección: se agregan ()
    ret, current_frame = file_video_stream.read()
    
    # Si no se puede leer el frame (fin del video o error), salir del bucle
    if not ret:
        break

    img_to_detect = current_frame
    img_height = img_to_detect.shape[0]
    img_width = img_to_detect.shape[1]
    
    # Convertir a blob
    img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (320, 320), swapRB=True, crop=False)
    
    
    # Pasar el blob al modelo
    yolo_model.setInput(img_blob)
    obj_detection_layers = yolo_model.forward(yolo_output_layer)
    
    # Bucle sobre cada capa de salida
    for object_detection_layer in obj_detection_layers:
        # Bucle sobre las detecciones
        for object_detection in object_detection_layer:
            
            all_scores = object_detection[5:]
            predicted_class_id = np.argmax(all_scores)
            prediction_confidence = all_scores[predicted_class_id]
        
            # Filtrar por confianza > 50%
            if prediction_confidence > 0.50:
                predicted_class_label = class_labels[predicted_class_id]
                
                # Obtener coordenadas
                bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                start_x_pt = int(box_center_x_pt - (box_width / 2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))
                end_x_pt = start_x_pt + box_width
                end_y_pt = start_y_pt + box_height
                
                # Obtener color y dibujar
                box_color = class_colors[predicted_class_id]
                box_color = [int(c) for c in box_color]
                
                label_text = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
                
                cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 2)
                cv2.putText(img_to_detect, label_text, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    cv2.imshow("Detection Output", img_to_detect)
    
    # Terminar con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
file_video_stream.release()
cv2.destroyAllWindows()
print("Proceso finalizado.")