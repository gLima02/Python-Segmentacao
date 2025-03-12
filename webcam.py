import cv2
import numpy as np

def process_frame(frame):
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    image_lower_hsv = np.array([0, 150, 100])
    image_upper_hsv = np.array([20, 255, 200])

    
    mask_hsv = cv2.inRange(img_hsv, image_lower_hsv, image_upper_hsv)
    contornos, _ = cv2.findContours(mask_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contornos_img = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contornos_img, contornos, -1, (255, 0, 0), 2)
    
    if contornos:
        cnt = max(contornos, key=cv2.contourArea)  # Pegando o maior contorno
        M = cv2.moments(cnt)
        
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Desenha a cruz no centro de massa
            size = 20
            color = (128, 128, 0)
            cv2.line(contornos_img, (cx - size, cy), (cx + size, cy), color, 2)
            cv2.line(contornos_img, (cx, cy - size), (cx, cy + size), color, 2)
            
            # Adiciona texto com as coordenadas
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"({cx}, {cy})"
            cv2.putText(contornos_img, text, (cx + 10, cy - 10), font, 0.5, (200, 50, 0), 2)
    
    return contornos_img

def main():
    #CAPTURA MINHA WEBCAM EXTERNA
    cap = cv2.VideoCapture(1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("WEBCAM NORMAL", frame)
        processed_frame = process_frame(frame)
        cv2.imshow("Detecção de Contorno", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()