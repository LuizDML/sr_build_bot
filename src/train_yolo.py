# train_yolo.py
"""
Script de treinamento YOLOv8 para Star Rail
"""

from ultralytics import YOLO
from pathlib import Path
import yaml

class StarRailYOLOTrainer:
    """Gerenciador de treinamento YOLO"""
    
    def __init__(self, dataset_root='star_rail_yolo'):
        self.dataset_root = Path(dataset_root)
        self.model = None
        
    def create_config(self):
        """Cria arquivo de configuração YAML pro YOLO"""
        
        config = {
            'path': str(self.dataset_root / 'dataset'),
            'train': 'images/train',
            'val': 'images/val',
            
            'names': {
                0: 'character',
                1: 'equipment_icon',
                2: 'relic_icon',
                3: 'stat_value',
                4: 'character_name',
                5: 'equipment_name'
            }
        }
        
        config_path = self.dataset_root / 'configs' / 'dataset.yaml'
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ Configuração criada: {config_path}")
        return config_path
    
    def train(self, epochs=100, img_size=640, batch_size=16, pretrained='yolov8n.pt'):
        """
        Treina o modelo
        
        Args:
            epochs: número de épocas de treinamento
            img_size: tamanho da imagem (640 é padrão)
            batch_size: tamanho do batch (ajuste conforme sua GPU)
            pretrained: modelo base (n=nano, s=small, m=medium, l=large, x=xlarge)
        """
        
        print(f"""
        ╔════════════════════════════════════════════════════════════╗
        ║  INICIANDO TREINAMENTO YOLO                                ║
        ║  Modelo: {pretrained:48s} ║
        ║  Épocas: {epochs:48d} ║
        ║  Tamanho: {img_size:47d} ║
        ╚════════════════════════════════════════════════════════════╝
        """)
        
        # Cria config
        config_path = self.create_config()
        
        # Carrega modelo pré-treinado
        self.model = YOLO(pretrained)
        
        # Treina
        results = self.model.train(
            data=str(config_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='star_rail_detector',
            patience=50,  # early stopping
            save=True,
            device=0,  # 0 = GPU, 'cpu' = CPU
            
            # Augmentations (ajuda a generalizar)
            hsv_h=0.015,  # ajuste de matiz
            hsv_s=0.7,    # ajuste de saturação
            hsv_v=0.4,    # ajuste de valor
            degrees=0,    # sem rotação (UI é sempre reta)
            translate=0.1,  # translação leve
            scale=0.5,    # escala
            flipud=0.0,   # sem flip vertical
            fliplr=0.0,   # sem flip horizontal (texto ficaria invertido)
            mosaic=1.0,   # mosaic augmentation
        )
        
        print("\n✓ Treinamento concluído!")
        print(f"  Modelo salvo em: runs/detect/star_rail_detector/weights/best.pt")
        
        return results
    
    def validate(self):
        """Valida modelo no conjunto de validação"""
        if self.model is None:
            print("❌ Treine o modelo primeiro!")
            return
        
        results = self.model.val()
        print("\n📊 Métricas de Validação:")
        print(f"  mAP50: {results.box.map50:.3f}")
        print(f"  mAP50-95: {results.box.map:.3f}")
        
        return results
    
    def export_model(self, format='onnx'):
        """
        Exporta modelo para produção
        
        Formatos: 'onnx', 'torchscript', 'tflite', 'coreml'
        """
        if self.model is None:
            # Carrega melhor modelo
            self.model = YOLO('runs/detect/star_rail_detector/weights/best.pt')
        
        self.model.export(format=format)
        print(f"✓ Modelo exportado para {format}")


# Script de execução
if __name__ == '__main__':
    trainer = StarRailYOLOTrainer()
    
    # Treina (ajuste conforme seu hardware)
    trainer.train(
        epochs=100,        # Comece com 100, aumente se necessário
        img_size=640,      # Padrão YOLO
        batch_size=16,     # Reduza se der OOM (out of memory)
        pretrained='yolov8n.pt'  # Nano - mais rápido
    )
    
    # Valida
    trainer.validate()
    
    # Exporta para produção
    trainer.export_model(format='onnx')