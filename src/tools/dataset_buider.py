"""
Constroi dataset dos equips e chars de Star Rail
- Captura screenshots
- Extrai ícones
- Organiza a estrutura em pastas
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import os

class DatasetBuilder:
    """
    Gerenciamento do dataset
    """

    def __init__(self, project_root='star_rail_dataset'):
        self.root = Path(project_root)
        self.setup_directories()

    def setup_directories(self):
        """Estruturas de pastas para o dataset"""
        directories = [
            'raw_screenshots',      # prints originais
            'icons/characters', 
            'icons/equipment',
            'icons/relics', 
            'dataset/images',       # Imagens para treino
            'dataset/labels',       # Anotações YOLO
            'dataset/images/train', # split treino
            'dataset/images/val',   # split validação
            'dataset/labels/train',
            'dataset/images/val',
            'models',               # Modelos treinados
            'configs'               # Arquivos de Configuração
        ]

        for  dir_path in directories:
            (self.root / dir_path).mkdir(parents=True, exist_ok=True)

        print(f"✓ Estrutura criada em: {self.root.absolute()}")

class IconExtractor:
    """Extratror interativo dos ícones"""

    def __init__(self, dataset_builder):
        self.dataset = dataset_builder
        self.current_image = None
        self.current_image_path = None
        self.icons_extrated = []

    def load_image(self, image_path):
        """Carrega uma imagem para extração"""
        self.current_image_path = image_path
        self.current_image = cv2.imread(str(image_path))

        if self.current_image is None:
            raise ValueError(f"Não foi possível carregar: {image_path}")
        
        print(f"✓ Imagem carregada: {image_path}")
        print(f"  Dimensões: {self.current_image.shape[1]}x{self.current_image.shape[0]}")

    def extract_interactive(self, category='characters'):
        """Modo interativo - seleciona multiplas regiões
        
        Args:
            category: 'charecaters'. 'equips' ou 'relics'
        
        Controles:
            - Clique e arraste para selecionar a região
            - ESC para terminar
            - Enter para confirmar a seleção
        """

        if self.current_image is None:
            print("❌ Carregue uma imagem primeiro com load_image()")
            return
        
        print(f"\n{'='*60}")
        print(f"MODO EXTRAÇÃO: {category.upper()}")
        print(f"\n{'='*60}")
        print("Instruções")
        print("   1. Clique e arraste para selecionar o ícone")
        print("   2. Pressione ENTER para confirmar")
        print("   3. Digite o nome do ícone")
        print("   4. Repita para outros ícones")
        print("   5. Pressione ESC quando terminar")
        print(f"\n{'='*60}")

        extraction_count = 0

        while True:
            # Cria uma cópia para desenhar
            display_img = self.current_image.copy()

            # Mostre ícones já extraídos
            for icon_info in self.icone_extracted:
                x, y, w, h = icon_info['bbox']
                cv2.rectangle(display_img, (x,y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_img, icon_info['name'], (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            # Seleção de ROI
            roi = cv2.selectROI("Selecione o ícone (ESC para sair)", display_img, False, False)

            # ESC pressionado ou janela fechada
            if roi == (0, 0, 0, 0):
                cv2.destroyAllWindows()
                break

            x, y, w, h = roi

            # Extrair a região
            icon_region = self.current_image[y:y+h], [x:x+w]

            # Exibir preview
            cv2.imshow("Preview - Está OK? (ESC=não, ENTER=sim)", icon_region)
            key = cv2.waitKey(0)
            cv2.destroyWindow("Preview - Está OK? (ESC=não, ENTER=sim)")

            if key == 27: # ESC
                continue

            # Qual o nome?
            name = input(f"\n Nome do ícone (ex: seele, kafka, musketeer_set): ").strip()

            if not name:
                print("  ⚠️  Nome vazio, pulando...")
                continue

            # Salvar o ícone
            output_path = self.dataset.root / 'icons' / category / f"{name}.png"
            cv2.imwrite(str(output_path), icon_region)

            extraction_count += 1

            # registrar extração
            icon_info = {
                'name': name,
                'category': category,
                'bbox': (x, y, w, h),
                'source_image': str(self.current_image_path),
                'output_path': str(output_path),
                'timestamp': datetime.now().isoformat()
            }
            self.icons_extrated.append(icon_info)

            print(f" ✓ Salvo: {output_path}")
            print(f"  Total extraído nesta sessão: {extraction_count}\n")

        print(f"\n{'='*60}")
        print(f"Sessão finalizada! Total de ícones extraídos: {extraction_count}")
        print(f"{'='*60}\n")

        # Salvar metadados
        self.save_metadata()

    def extract_by_coordinates(self, coords_list, category='characters'):
        """
        Extração em batch usando coordenadas predefinidas
        
        Args:
            coords_list: lista de dicts com 'name', 'x', 'y', 'w', 'h'
            
        Exemplo:
            coords = [
                {'name': 'seele', 'x': 100, 'y': 50, 'w': 64, 'h': 64},
                {'name': 'kafka', 'x': 200, 'y': 50, 'w': 64, 'h': 64},
            ]
        """

        if self.current_image is None:
            print("❌ Carregue uma imagem primeiro")
            return
        
        for coord in coords_list:
            name = coord['name']
            x, y, w, h = coord['x'], coord['y'], coord['w'], coord['h']
            
            # Extrai região
            icon_region = self.current_image[y:y+h, x:x+w]
            
            # Salva
            output_path = self.dataset.root / 'icons' / category / f"{name}.png"
            cv2.imwrite(str(output_path), icon_region)
            
            print(f"✓ Extraído: {name} → {output_path}")

    def save_metadata(self):
        """Salva informações sobre ícones extraídos"""
        metadata_path = self.dataset.root / 'icons' / 'extraction_metadata.json'
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.icons_extracted, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Metadados salvos em: {metadata_path}")  

class ScreenshotOrganizer:
    """Organiza e prepara screenshots para anotação"""
    
    def __init__(self, dataset_builder):
        self.dataset = dataset_builder
    
    def import_screenshots(self, source_folder):
        """
        Importa screenshots de uma pasta
        Renomeia e organiza automaticamente
        """
        source = Path(source_folder)
        
        if not source.exists():
            print(f"❌ Pasta não encontrada: {source}")
            return
        
        # Extensões suportadas
        extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        
        imported = 0
        for img_file in source.iterdir():
            if img_file.suffix.lower() in extensions:
                # Gera nome único
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                new_name = f"screenshot_{timestamp}_{imported:04d}{img_file.suffix}"
                
                dest = self.dataset.root / 'raw_screenshots' / new_name
                
                # Copia arquivo
                import shutil
                shutil.copy2(img_file, dest)
                
                imported += 1
                print(f"✓ Importado: {img_file.name} → {new_name}")
        
        print(f"\n📸 Total importado: {imported} screenshots")
    
    def prepare_for_annotation(self, sample_size=None):
        """
        Prepara screenshots para anotação no LabelImg/Roboflow
        Copia para pasta dataset/images
        """
        raw_path = self.dataset.root / 'raw_screenshots'
        dest_path = self.dataset.root / 'dataset' / 'images'
        
        screenshots = list(raw_path.glob('*.png')) + list(raw_path.glob('*.jpg'))
        
        if sample_size:
            screenshots = screenshots[:sample_size]
        
        for img_file in screenshots:
            dest = dest_path / img_file.name
            import shutil
            shutil.copy2(img_file, dest)
            print(f"✓ Preparado para anotação: {img_file.name}")
        
        print(f"\n📋 {len(screenshots)} imagens prontas para anotação")
        print(f"   Local: {dest_path}")


class DatasetSplitter:
    """Divide dataset em treino/validação"""
    
    def __init__(self, dataset_builder):
        self.dataset = dataset_builder
    
    def split_dataset(self, train_ratio=0.8):
        """
        Divide imagens anotadas em treino/validação
        
        Args:
            train_ratio: proporção para treino (ex: 0.8 = 80% treino, 20% validação)
        """
        images_path = self.dataset.root / 'dataset' / 'images'
        labels_path = self.dataset.root / 'dataset' / 'labels'
        
        # Pega todas as imagens que têm label correspondente
        images = []
        for img_file in images_path.glob('*.png'):
            label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                images.append(img_file)
        
        # Embaralha
        import random
        random.shuffle(images)
        
        # Divide
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Move arquivos
        import shutil
        
        for img in train_images:
            label = labels_path / f"{img.stem}.txt"
            
            shutil.move(str(img), str(self.dataset.root / 'dataset/images/train' / img.name))
            shutil.move(str(label), str(self.dataset.root / 'dataset/labels/train' / label.name))
        
        for img in val_images:
            label = labels_path / f"{img.stem}.txt"
            
            shutil.move(str(img), str(self.dataset.root / 'dataset/images/val' / img.name))
            shutil.move(str(label), str(self.dataset.root / 'dataset/labels/val' / label.name))
        
        print(f"✓ Dataset dividido:")
        print(f"  Treino: {len(train_images)} imagens")
        print(f"  Validação: {len(val_images)} imagens")