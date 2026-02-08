# src/detector/yolo_detector.py
"""
Detector de personagens e equipamentos usando YOLO
"""

from ultralytics import YOLO
import cv2
import numpy as np

class StarRailDetector:
    """Detector YOLO para Star Rail"""
    
    def __init__(self, model_path='runs/detect/star_rail_detector/weights/best.pt'):
        """
        Args:
            model_path: caminho pro modelo treinado
        """
        self.model = YOLO(model_path)
        
        # Mapeamento de classes
        self.class_names = {
            0: 'character',
            1: 'equipment_icon',
            2: 'relic_icon',
            3: 'stat_value',
            4: 'character_name',
            5: 'equipment_name'
        }
    
    def detect(self, image_path, confidence=0.5):
        """
        Detecta objetos na imagem
        
        Args:
            image_path: caminho da screenshot
            confidence: threshold de confiança (0-1)
        
        Returns:
            dict com detecções organizadas por tipo
        """
        # Roda inferência
        results = self.model.predict(
            source=image_path,
            conf=confidence,
            iou=0.45,
            verbose=False
        )
        
        # Processa resultados
        detections = {
            'character': [],
            'equipment_icon': [],
            'relic_icon': [],
            'stat_value': [],
            'character_name': [],
            'equipment_name': []
        }
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extrai informações
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                class_name = self.class_names.get(class_id, 'unknown')
                
                detection = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'center': self._get_center(bbox)
                }
                
                detections[class_name].append(detection)
        
        return detections
    
    def _get_center(self, bbox):
        """Calcula centro da bbox"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def visualize(self, image_path, detections, output_path='detection_result.jpg'):
        """
        Desenha detecções na imagem
        """
        img = cv2.imread(image_path)
        
        # Cores por classe
        colors = {
            'character': (0, 255, 0),      # Verde
            'equipment_icon': (255, 0, 0),  # Azul
            'relic_icon': (0, 255, 255),    # Amarelo
            'stat_value': (255, 255, 0),    # Ciano
            'character_name': (255, 0, 255), # Magenta
            'equipment_name': (128, 128, 128) # Cinza
        }
        
        for class_name, items in detections.items():
            color = colors.get(class_name, (255, 255, 255))
            
            for item in items:
                x1, y1, x2, y2 = map(int, item['bbox'])
                conf = item['confidence']
                
                # Desenha bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Desenha label
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(output_path, img)
        print(f"✓ Visualização salva: {output_path}")
        
        return output_path


# Exemplo de uso
if __name__ == '__main__':
    # Inicializa detector
    detector = StarRailDetector()
    
    # Detecta
    detections = detector.detect('screenshot.png', confidence=0.6)
    
    # Mostra resultados
    print("\n📊 DETECÇÕES:")
    for class_name, items in detections.items():
        if items:
            print(f"\n{class_name.upper()}:")
            for i, item in enumerate(items, 1):
                print(f"  {i}. Confiança: {item['confidence']:.2%} | Posição: {item['center']}")
    
    # Visualiza
    detector.visualize('screenshot.png', detections)