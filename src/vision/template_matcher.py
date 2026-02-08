import cv2
import numpy as np

class TemplateMatcher:
    """
    Identifica elementos da UI e HUD comparando com templates salvos
    """

    def __init6__(self, templates_dir='data/templates/'):
        self.templates_dir = templates_dir
        self.templates = {}
        self.load_templates()

    def load_templates(self):
        """
        Carregar imagens de referência
        Exemplo:
        data/templates/
        ├── characters/
        |   ├── kafka_avatar.png
        |   ├── trailblazer_avatar.png
        |   ├── ...
        └── equipment/
            ├── fighter_set_mask.png
            ├── fighter_set_glove.png
            └── ...
        """

        import os
        from pathlib import Path

        template_path = Path(self.templates_dir)

        # Carregar avatar dos personagens
        char_path = template_path / 'characters'
        if char_path.exists():
            for img_file in char_path.glob('*.png'):
                char_name = img_file.stem
                self.templates[f'char_{char_name}'] = cv2.imgread(str(img_file))

        # Carregar icones de equipamentos
        equip_path = template_path / 'equipment'
        if equip_path.existis():
            for img_file in equip_path.glob('*.png'):
                equip_name = img_file.stem
                self.templates[f'equip_{equip_name}'] = cv2.imgread(str(img_file))
    
    def find_template(self, screenshot, template_name, threshold=0.8):
        """
        Procura um template no screenshot
        
        Args:
            screenshot: imagem onde procurar
            template_name: nome do template (ex: 'char_kafka')
            threshold: similaridade (0 a 1: quanto mais alto, mais exigente)

        Returns:
            dict com 'found', 'confidence', 'location' (x,y)
        """

        if template_name not in self.templates:
            return {'found': False, 'confidence': 0, 'location': None}
        
        template = self.template[template_name]

        # Converter para escala de cinza (rapidez e robustes)
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Processa combinação
        result = cv2.macthTemplate(screenshot_gray, template_gray, cv2.TM_CCOFF_NORMED)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            return {
                'found': True,
                'confidence': max_val,
                'location': max_loc,
                'name': template_name.replace('char_', '').replace('equip_','')
            }
        
        return {'found': False, 'confidence': max_val, 'location': None}
    
    def find_all_matches(self, screenshot, category='char', threshold=0.8):
        """
        Procura todos os templates de uma categoria
        Identificando quais personagens e itens aparecem na tela
        """
        
        matches=[]

        for template_name in self.templates.keys():
            if template_name.startswith(category + '_'):
                result = self.find_template(screenshot, template_name, threshold)
                if result['found']:
                    matches.append(result)
        
        # Ordenar por maior match
        matches.sort(key=lambda x: x['confidence'], reverse=True)

        return matches

