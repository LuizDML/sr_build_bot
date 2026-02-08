import cv2
import numpy as np

class FeatureMatcher:
    """
    Identifica imagens mesmo com pequenas variações, 
    Mais robusto que template matching, porém mais lento
    """

    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.macther = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract_features(self, image):
        """
        Extrai pontos caracteristicos da imagem
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def macth_images(self, img1, img2, min_matches=10):
        """
        Compara ambas imagens, retorna True se similares
        """
        kp1, desc1 = self.extract_features(img1)
        kp2, desc2 = self.extract_features(img2)

        if desc1 is None or desc2 is None:
            return False, 0 
        
        matches = self.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Distância baixa = melhores matches
        good_matches = [m for m in matches if m.distance < 50]

        similarity = len(good_matches) / max(len(kp1), len(kp2))

        return len(good_matches) >= min_matches, similarity