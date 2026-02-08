# run_extraction.py
"""
Script principal para extrair ícones e preparar dataset
"""

from tools.dataset_builder import DatasetBuilder, IconExtractor, ScreenshotOrganizer

def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     STAR RAIL YOLO DATASET BUILDER v1.0                   ║
    ║     Preparação de Dataset para Detecção de Objetos        ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # 1. Inicializa builder
    print("\n[1/5] Inicializando estrutura...")
    builder = DatasetBuilder(project_root='star_rail_yolo')
    
    # 2. Importa screenshots (coloque seus screenshots numa pasta)
    print("\n[2/5] Importando screenshots...")
    organizer = ScreenshotOrganizer(builder)
    organizer.import_screenshots('meus_screenshots')  # ← mude aqui
    
    # 3. Extração de ícones
    print("\n[3/5] Extração de ícones...")
    extractor = IconExtractor(builder)
    
    # Extrai ícones de personagens
    print("\n--- Extraindo PERSONAGENS ---")
    extractor.load_image('star_rail_yolo/raw_screenshots/screenshot_20250130_000001_0000.png')
    extractor.extract_interactive(category='characters')
    
    # Extrai ícones de equipamentos
    print("\n--- Extraindo EQUIPAMENTOS ---")
    extractor.load_image('star_rail_yolo/raw_screenshots/screenshot_20250130_000001_0001.png')
    extractor.extract_interactive(category='equipment')
    
    print("\n[4/5] Preparando imagens para anotação...")
    organizer.prepare_for_annotation()
    
    print("\n[5/5] Concluído!")
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  PRÓXIMOS PASSOS:                                          ║
    ║                                                            ║
    ║  1. Anote as imagens usando LabelImg ou Roboflow          ║
    ║     Localização: star_rail_yolo/dataset/images/           ║
    ║                                                            ║
    ║  2. Depois rode: python run_training.py                   ║
    ╚════════════════════════════════════════════════════════════╝
    """)

if __name__ == '__main__':
    main()