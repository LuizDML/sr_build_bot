"""
Combina YOLO (detecção) + OCR (leitura de valores)
O melhor dos dois mundos!
"""

from src.detector.yolo_detector import StarRailDetector
from src.ocr.text_extractor import TextExtractor
import cv2
import pytesseract

class HybridAnalyzer:
    """Análise híbrida: YOLO encontra, OCR lê"""
    
    def __init__(self, yolo_model_path):
        self.detector = StarRailDetector(yolo_model_path)
        self.ocr = TextExtractor()
    
    def analyze_equipment_screen(self, screenshot_path):
        """
        Pipeline completo:
        1. YOLO detecta personagem e ícones de equipamento
        2. OCR lê os valores numéricos dos stats
        3. Retorna tudo estruturado
        """
        
        # 1. Detecta elementos visuais
        detections = self.detector.detect(screenshot_path, confidence=0.7)
        
        # 2. Carrega imagem para OCR
        img = cv2.imread(screenshot_path)
        
        # 3. Extrai valores numéricos das regiões de stats
        stats = []
        for stat_detection in detections.get('stat_value', []):
            bbox = stat_detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Recorta região do stat
            stat_region = img[y1:y2, x1:x2]
            
            # Pré-processa para OCR
            gray = cv2.cvtColor(stat_region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Lê valor
            text = pytesseract.image_to_string(binary, config='--psm 7 digits')
            
            stats.append({
                'value': text.strip(),
                'bbox': bbox,
                'confidence': stat_detection['confidence']
            })
        
        # 4. Monta resultado estruturado
        result = {
            'character': detections.get('character', []),
            'equipment': detections.get('equipment_icon', []),
            'relics': detections.get('relic_icon', []),
            'stats': stats,
            'raw_detections': detections
        }
        
        return result
"""

---

# Checklist de Execução

□ 1. Instalar dependências
    pip install ultralytics opencv-python pytesseract pillow pyyaml

□ 2. Coletar screenshots (50-200 imagens)
    - Diferentes personagens
    - Diferentes equipamentos
    - Variações de UI

□ 3. Extrair ícones de referência
    python run_extraction.py

□ 4. Anotar imagens
    - LabelImg ou Roboflow
    - Marcar: personagens, equipamentos, stats

□ 5. Treinar modelo
    python train_yolo.py

□ 6. Testar detecção
    python test_detector.py

□ 7. Integrar no bot principal

"""