import numpy as np
import cv2

# --- SECCIÓN 1: CONFIGURACIÓN (Fuera del bucle para mejor rendimiento) ---
class_labels = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
                "fire hydrant","street sign","stop sign","parking meter","bench","bird","cat","dog","horse",
                "sheep","cow","elephant","bear","zebra","giraffe","hat","backpack","umbrella","shoe","eye glasses",
                "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
                "skateboard","surfboard","tennis racket","bottle","plate","wine glass","cup","fork","knife",
                "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
                "cake","chair","sofa","pottedplant","bed","mirror","diningtable","window","desk","toilet","door","tv",
                "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
                "blender","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255","255,0,255"]
class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
class_colors = np.array(class_colors)
class_colors = np.tile(class_colors,(15,1))

# CARGAR MODELO UNA SOLA VEZ
print("Cargando Mask R-CNN...")
maskrcnn = cv2.dnn.readNetFromTensorflow('models/maskrcnn_buffermodel.pb','models/maskrcnn_bufferconfig.txt')

file_video_stream = cv2.VideoCapture('images/video_sample.mp4')

# Configuración de ventana para que no se pierda nada de la pantalla
cv2.namedWindow("Detection Output", cv2.WINDOW_NORMAL)
# Ajustamos la ventana a un tamaño manejable, pero manteniendo la proporción
cv2.resizeWindow("Detection Output", 720, 1280) 

# --- SECCIÓN 2: PROCESAMIENTO ---
while (file_video_stream.isOpened()):
    ret, current_frame = file_video_stream.read()
    if not ret:
        break
        
    img_to_detect = current_frame
    img_height, img_width = img_to_detect.shape[:2]
    
    # Crear blob
    img_blob = cv2.dnn.blobFromImage(img_to_detect, swapRB=True, crop=False)
    maskrcnn.setInput(img_blob)
    (obj_detections_boxes, obj_detections_masks) = maskrcnn.forward(["detection_out_final","detection_masks"])
    
    no_of_detections = obj_detections_boxes.shape[2]
    
    for index in range(no_of_detections):
        prediction_confidence = obj_detections_boxes[0, 0, index, 2]
        
        if prediction_confidence > 0.20:
            predicted_class_index = int(obj_detections_boxes[0, 0, index, 1])
            
            # Coordenadas
            bounding_box = obj_detections_boxes[0, 0, index, 3:7] * np.array([img_width, img_height, img_width, img_height])
            (start_x_pt, start_y_pt, end_x_pt, end_y_pt) = bounding_box.astype("int")
            
            # Validar que las coordenadas estén dentro del frame para evitar errores de segmentación
            start_x_pt, start_y_pt = max(0, start_x_pt), max(0, start_y_pt)
            end_x_pt, end_y_pt = min(img_width, end_x_pt), min(img_height, end_y_pt)
            
            # Máscara
            object_mask = obj_detections_masks[index, predicted_class_index]
            object_mask = cv2.resize(object_mask, (end_x_pt - start_x_pt, end_y_pt - start_y_pt))
            object_mask = (object_mask > 0.3)
            
            # Aplicar color a la máscara
            mask_color = [int(c) for c in class_colors[predicted_class_index]]
            roi = img_to_detect[start_y_pt:end_y_pt, start_x_pt:end_x_pt]
            roi[object_mask] = (0.3 * np.array(mask_color) + 0.7 * roi[object_mask]).astype("uint8")
            
            # --- AJUSTE DE LABELS PARA ALTA RESOLUCIÓN ---
            label = "{}: {:.1f}%".format(class_labels[predicted_class_index], prediction_confidence * 100)
            
            # Dinamismo: Ajustamos grosor y tamaño según el ancho de la imagen
            font_scale = img_width / 1000.0  # Para 1080p será ~1.0
            thickness = int(font_scale * 2)
            
            cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), mask_color, thickness)
            cv2.putText(img_to_detect, label, (start_x_pt, start_y_pt - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, mask_color, thickness)

    cv2.imshow("Detection Output", img_to_detect)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

file_video_stream.release()
cv2.destroyAllWindows()
