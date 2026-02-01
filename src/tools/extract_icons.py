import cv2
import numpu as np

class IconExtractor:
    """
    Cria os templates
    Recorta ícones das screenshots
    """

    def extract_region(self, image_path, x, y, width, height, output_path):
        """
        Recorta uma região da imagem 
        """

        img = cv2.imread(image_path)
        region = img[y:y+height, x:x+width]
        cv2.imwrite(output_path, region)
        print(f"Ícone salvo em : {output_path}")
    
    def interactive_extract(self, image_path):
        """
        Modo interativo, permite clicar e arrastar para definir a região
        """

        img = cv2.imread(image_path)
        clone = img.copy()
        
        roi = cv2.selectROI("Selecione o ícone", img, False)
        cv2.destroyAllWindows()

        if roi != (0, 0, 0, 0):
            x, y, w, h = roi
            region = clone[y:y+h, x:x+w]
            
            # Pede nome pra salvar
            name = input("Nome do ícone (ex: avatar_kafka): ")
            cv2.imwrite(f"data/templates/{name}.png", region)
            print(f"✓ Salvo como {name}.png")