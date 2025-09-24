"""
Advanced Circle and Ellipse Analyzer - Análise Geométrica Híbrida
Análise completa de formas circulares e elípticas com rastreabilidade total
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse as MPLEllipse
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import pandas as pd
import os
import json
import datetime
import hashlib
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Union
import sys
from scipy import stats
from scipy.spatial.distance import mahalanobis
import seaborn as sns
from sklearn.decomposition import PCA

@dataclass
class HybridAnalysisParameters:
    """Parâmetros de configuração da análise híbrida"""
    # Parâmetros de pré-processamento
    binary_threshold: int = 127
    morphology_kernel_size: int = 3
    
    # Parâmetros Transformada de Hough (Círculos)
    hough_dp: int = 1
    hough_min_dist: int = 10
    hough_param1: int = 50
    hough_param2: int = 15
    hough_min_radius: int = 3
    hough_max_radius: int = 25
    
    # Parâmetros de filtros básicos
    min_area: int = 50
    min_circularity: float = 0.6
    min_aspect_ratio: float = 0.7
    max_aspect_ratio: float = 1.4
    
    # Parâmetros específicos para elipses
    min_ellipse_axis_ratio: float = 0.3
    max_ellipse_axis_ratio: float = 5.0
    min_eccentricity: float = 0.0
    max_eccentricity: float = 0.95
    ellipse_fit_threshold: float = 0.8
    
    # Parâmetros de clustering circular
    circular_clustering_eps: float = 35.0
    circular_clustering_min_samples: int = 2
    duplicate_distance_threshold: float = 15.0
    
    # Parâmetros de clustering elíptico
    elliptical_covariance_scale: float = 2.0
    elliptical_confidence_level: float = 0.95
    elliptical_overlap_threshold: float = 0.3
    gmm_max_components: int = 10
    
    # Parâmetros de análise direcional
    angle_bins: int = 18
    angular_tolerance: float = 15.0
    linearity_threshold: float = 0.7
    
    # Estratégia híbrida
    auto_select_best_fit: bool = True
    prefer_ellipse_for_elongated: bool = True
    elongation_threshold: float = 2.0
    
    def validate(self) -> List[str]:
        """Valida os parâmetros e retorna lista de erros"""
        errors = []
        
        if not (0 <= self.binary_threshold <= 255):
            errors.append("binary_threshold deve estar entre 0 e 255")
        
        if self.morphology_kernel_size < 1 or self.morphology_kernel_size % 2 == 0:
            errors.append("morphology_kernel_size deve ser ímpar e >= 1")
            
        if self.min_ellipse_axis_ratio >= self.max_ellipse_axis_ratio:
            errors.append("min_ellipse_axis_ratio deve ser < max_ellipse_axis_ratio")
            
        if not (0.0 <= self.min_eccentricity <= self.max_eccentricity <= 1.0):
            errors.append("Eccentricidade deve estar entre 0.0 e 1.0")
            
        if not (0.0 <= self.elliptical_confidence_level <= 1.0):
            errors.append("elliptical_confidence_level deve estar entre 0.0 e 1.0")
            
        if self.gmm_max_components < 1:
            errors.append("gmm_max_components deve ser >= 1")
            
        return errors

class HybridCircleEllipseAnalyzer:
    def __init__(self, image_path: str, output_dir: str = "hybrid_analysis_output", 
                 params: Optional[HybridAnalysisParameters] = None):
        """
        Inicializa o analisador híbrido de círculos e elipses
        """
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        self.params = params or HybridAnalysisParameters()
        
        # Dados da análise
        self.image_info = {}
        self.original_image = None
        self.gray_image = None
        self.binary_image = None
        
        # Resultados circulares
        self.detected_circles = []
        self.isolated_circles = []
        self.circular_formations = []
        
        # Resultados elípticos
        self.detected_ellipses = []
        self.isolated_ellipses = []
        self.elliptical_formations = []
        
        # Análise híbrida
        self.hybrid_objects = []
        self.directional_analysis = {}
        self.linear_formations = []
        self.shape_decisions = []
        
        # Metadados
        self.analysis_metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'version': '2.0.0_hybrid',
            'analysis_type': 'hybrid_circle_ellipse',
            'input_image_path': str(self.image_path),
            'output_directory': str(self.output_dir),
            'parameters': asdict(self.params)
        }
        
        self._setup_output_structure()
        self._setup_logging()
        
    def _setup_output_structure(self):
        """Cria estrutura de pastas de saída"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.images_dir = self.output_dir / "images"
        self.data_dir = self.output_dir / "data"
        self.logs_dir = self.output_dir / "logs"
        self.analysis_dir = self.output_dir / "analysis"
        
        for directory in [self.images_dir, self.data_dir, self.logs_dir, self.analysis_dir]:
            directory.mkdir(exist_ok=True)
            
    def _setup_logging(self):
        """Configura sistema de logging"""
        log_file = self.logs_dir / f"hybrid_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def validate_inputs(self) -> bool:
        """Valida parâmetros de entrada e arquivos"""
        self.logger.info("=== VALIDAÇÃO DE ENTRADA HÍBRIDA ===")
        
        if not self.image_path.exists():
            self.logger.error(f"Arquivo de imagem não encontrado: {self.image_path}")
            return False
            
        if not self.image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            self.logger.error(f"Formato de imagem não suportado: {self.image_path.suffix}")
            return False
            
        param_errors = self.params.validate()
        if param_errors:
            self.logger.error("Erros nos parâmetros:")
            for error in param_errors:
                self.logger.error(f"  - {error}")
            return False
            
        self.logger.info("✓ Validação híbrida concluída com sucesso")
        return True
        
    def calculate_image_hash(self, image: np.ndarray) -> str:
        """Calcula hash MD5 da imagem para rastreabilidade"""
        return hashlib.md5(image.tobytes()).hexdigest()
        
    def load_and_preprocess(self):
        """Carrega e pré-processa a imagem"""
        self.logger.info("=== PRÉ-PROCESSAMENTO HÍBRIDO ===")
        
        self.original_image = cv2.imread(str(self.image_path))
        if self.original_image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {self.image_path}")
            
        height, width = self.original_image.shape[:2]
        file_size = self.image_path.stat().st_size
        
        self.image_info = {
            'filename': self.image_path.name,
            'width': width,
            'height': height,
            'channels': self.original_image.shape[2] if len(self.original_image.shape) > 2 else 1,
            'file_size_bytes': file_size,
            'hash_original': self.calculate_image_hash(self.original_image)
        }
        
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        _, self.binary_image = cv2.threshold(
            self.gray_image, 
            self.params.binary_threshold, 
            255, 
            cv2.THRESH_BINARY_INV
        )
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.params.morphology_kernel_size, self.params.morphology_kernel_size)
        )
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_CLOSE, kernel)
        
        self.image_info['hash_gray'] = self.calculate_image_hash(self.gray_image)
        self.image_info['hash_binary'] = self.calculate_image_hash(self.binary_image)
        
        self.logger.info(f"✓ Imagem carregada: {width}x{height}, {file_size} bytes")

    def detect_circles_hough(self) -> List[Dict]:
        """Detecção usando Transformada de Hough"""
        self.logger.info("Detectando círculos com Transformada de Hough...")
        
        circles = cv2.HoughCircles(
            self.gray_image,
            cv2.HOUGH_GRADIENT,
            dp=self.params.hough_dp,
            minDist=self.params.hough_min_dist,
            param1=self.params.hough_param1,
            param2=self.params.hough_param2,
            minRadius=self.params.hough_min_radius,
            maxRadius=self.params.hough_max_radius
        )
        
        hough_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                hough_circles.append({
                    'method': 'hough',
                    'shape_type': 'circle',
                    'center': (x, y),
                    'radius': r,
                    'area': np.pi * r * r,
                    'confidence': 1.0,
                    'eccentricity': 0.0,
                    'major_axis': r * 2,
                    'minor_axis': r * 2,
                    'angle': 0.0
                })
        
        self.logger.info(f"✓ Hough detectou: {len(hough_circles)} círculos")
        return hough_circles
    
    def detect_circles_contours(self) -> List[Dict]:
        """Detecção usando análise de contornos"""
        self.logger.info("Detectando círculos por contornos...")
        
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_circles = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area > self.params.min_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > self.params.min_circularity:
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        
                        contour_circles.append({
                            'method': 'contour',
                            'shape_type': 'circle',
                            'center': (int(x), int(y)),
                            'radius': int(radius),
                            'area': area,
                            'circularity': circularity,
                            'perimeter': perimeter,
                            'contour_id': i,
                            'eccentricity': 0.0,
                            'major_axis': radius * 2,
                            'minor_axis': radius * 2,
                            'angle': 0.0
                        })
        
        self.logger.info(f"✓ Contornos detectaram: {len(contour_circles)} círculos")
        return contour_circles

    def detect_ellipses_contours(self) -> List[Dict]:
        """Detecção de elipses usando fitting em contornos"""
        self.logger.info("Detectando elipses por contornos...")
        
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ellipse_objects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area > self.params.min_area and len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (center_x, center_y), (width, height), angle = ellipse
                    
                    major_axis = max(width, height)
                    minor_axis = min(width, height)
                    axis_ratio = major_axis / minor_axis if minor_axis > 0 else float('inf')
                    
                    if major_axis > 0:
                        eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2)
                    else:
                        eccentricity = 0.0
                    
                    if (self.params.min_ellipse_axis_ratio <= axis_ratio <= self.params.max_ellipse_axis_ratio and
                        self.params.min_eccentricity <= eccentricity <= self.params.max_eccentricity):
                        
                        ellipse_area = np.pi * (major_axis/2) * (minor_axis/2)
                        fit_quality = min(area / ellipse_area, ellipse_area / area) if ellipse_area > 0 else 0
                        
                        if fit_quality > self.params.ellipse_fit_threshold:
                            ellipse_objects.append({
                                'method': 'contour_ellipse',
                                'shape_type': 'ellipse',
                                'center': (int(center_x), int(center_y)),
                                'major_axis': major_axis,
                                'minor_axis': minor_axis,
                                'angle': angle,
                                'area': area,
                                'eccentricity': eccentricity,
                                'axis_ratio': axis_ratio,
                                'fit_quality': fit_quality,
                                'contour_id': i,
                                'radius': np.sqrt(area / np.pi)
                            })
                
                except Exception as e:
                    self.logger.debug(f"Erro no fitting de elipse para contorno {i}: {e}")
                    continue
        
        self.logger.info(f"✓ Contornos detectaram: {len(ellipse_objects)} elipses")
        return ellipse_objects
    
    def detect_ellipses_moments(self) -> List[Dict]:
        """Detecção de elipses usando análise de momentos"""
        self.logger.info("Detectando elipses por momentos...")
        
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        moment_ellipses = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area > self.params.min_area:
                moments = cv2.moments(contour)
                
                if moments['m00'] > 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                    
                    mu20 = moments['mu20'] / moments['m00']
                    mu02 = moments['mu02'] / moments['m00']
                    mu11 = moments['mu11'] / moments['m00']
                    
                    denominator = mu20 - mu02
                    if abs(denominator) > 1e-6:
                        angle = 0.5 * np.arctan2(2 * mu11, denominator)
                    else:
                        angle = np.pi/4 if mu11 > 0 else -np.pi/4
                    
                    common = np.sqrt((mu20 - mu02)**2 + 4 * mu11**2)
                    major_axis = 2 * np.sqrt(2) * np.sqrt(mu20 + mu02 + common)
                    minor_axis = 2 * np.sqrt(2) * np.sqrt(mu20 + mu02 - common)
                    
                    if minor_axis > 0:
                        axis_ratio = major_axis / minor_axis
                        eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2) if major_axis > 0 else 0
                        
                        if (self.params.min_ellipse_axis_ratio <= axis_ratio <= self.params.max_ellipse_axis_ratio and
                            self.params.min_eccentricity <= eccentricity <= self.params.max_eccentricity):
                            
                            moment_ellipses.append({
                                'method': 'moments_ellipse',
                                'shape_type': 'ellipse',
                                'center': (cx, cy),
                                'major_axis': major_axis,
                                'minor_axis': minor_axis,
                                'angle': np.degrees(angle),
                                'area': area,
                                'eccentricity': eccentricity,
                                'axis_ratio': axis_ratio,
                                'contour_id': i,
                                'radius': np.sqrt(area / np.pi)
                            })
        
        self.logger.info(f"✓ Momentos detectaram: {len(moment_ellipses)} elipses")
        return moment_ellipses

    def merge_all_detections(self, circles: List[Dict], ellipses: List[Dict]) -> None:
        """Merge todas as detecções (círculos e elipses)"""
        self.logger.info("=== MERGE DE DETECÇÕES HÍBRIDAS ===")
        
        all_objects = circles + ellipses
        merged_objects = []
        
        for obj in all_objects:
            is_duplicate = False
            cx, cy = obj['center']
            
            for existing in merged_objects:
                ex, ey = existing['center']
                distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                
                if distance < self.params.duplicate_distance_threshold:
                    is_duplicate = True
                    if self._object_quality(obj) > self._object_quality(existing):
                        merged_objects.remove(existing)
                        merged_objects.append(obj)
                    break
            
            if not is_duplicate:
                merged_objects.append(obj)
        
        self.detected_circles = [obj for obj in merged_objects if obj['shape_type'] == 'circle']
        self.detected_ellipses = [obj for obj in merged_objects if obj['shape_type'] == 'ellipse']
        
        self.logger.info(f"✓ Merged: {len(self.detected_circles)} círculos, {len(self.detected_ellipses)} elipses")
    
    def _object_quality(self, obj: Dict) -> float:
        """Calcula qualidade de um objeto detectado"""
        base_quality = obj.get('area', 0) * obj.get('fit_quality', obj.get('confidence', 1.0))
        
        if obj['shape_type'] == 'circle' and obj.get('eccentricity', 0) > 0.3:
            base_quality *= 0.5
            
        return base_quality

    def cluster_circles_traditional(self) -> None:
        """Clustering tradicional para círculos usando DBSCAN"""
        self.logger.info("Clustering tradicional de círculos...")
        
        if len(self.detected_circles) < 2:
            self.isolated_circles = self.detected_circles.copy()
            return
        
        centers = np.array([circle['center'] for circle in self.detected_circles])
        
        clustering = DBSCAN(
            eps=self.params.circular_clustering_eps, 
            min_samples=self.params.circular_clustering_min_samples
        ).fit(centers)
        
        cluster_dict = {}
        
        for i, label in enumerate(clustering.labels_):
            if label == -1:
                self.isolated_circles.append(self.detected_circles[i])
            else:
                if label not in cluster_dict:
                    cluster_dict[label] = []
                cluster_dict[label].append(self.detected_circles[i])
        
        self.circular_formations = list(cluster_dict.values())
        
        self.logger.info(f"✓ Círculos - Isolados: {len(self.isolated_circles)}, Formações: {len(self.circular_formations)}")
    
    def cluster_ellipses_gmm(self) -> None:
        """Clustering de elipses usando Gaussian Mixture Models"""
        self.logger.info("Clustering elíptico usando GMM...")
        
        if len(self.detected_ellipses) < 2:
            self.isolated_ellipses = self.detected_ellipses.copy()
            return
        
        centers = np.array([ellipse['center'] for ellipse in self.detected_ellipses])
        
        n_components = min(len(self.detected_ellipses) // 2, self.params.gmm_max_components)
        
        if n_components < 1:
            self.isolated_ellipses = self.detected_ellipses.copy()
            return
        
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=42
            )
            labels = gmm.fit_predict(centers)
            
            cluster_dict = {}
            
            for i, label in enumerate(labels):
                if label not in cluster_dict:
                    cluster_dict[label] = []
                cluster_dict[label].append(self.detected_ellipses[i])
            
            for label, objects in cluster_dict.items():
                if len(objects) == 1:
                    self.isolated_ellipses.extend(objects)
                else:
                    self.elliptical_formations.append(objects)
                    
        except Exception as e:
            self.logger.warning(f"Erro no GMM clustering: {e}. Usando DBSCAN como fallback.")
            self._cluster_ellipses_dbscan_fallback()
        
        self.logger.info(f"✓ Elipses - Isoladas: {len(self.isolated_ellipses)}, Formações: {len(self.elliptical_formations)}")
    
    def _cluster_ellipses_dbscan_fallback(self) -> None:
        """Clustering de fallback para elipses usando DBSCAN"""
        centers = np.array([ellipse['center'] for ellipse in self.detected_ellipses])
        
        clustering = DBSCAN(
            eps=self.params.circular_clustering_eps * 1.5,
            min_samples=self.params.circular_clustering_min_samples
        ).fit(centers)
        
        cluster_dict = {}
        
        for i, label in enumerate(clustering.labels_):
            if label == -1:
                self.isolated_ellipses.append(self.detected_ellipses[i])
            else:
                if label not in cluster_dict:
                    cluster_dict[label] = []
                cluster_dict[label].append(self.detected_ellipses[i])
        
        self.elliptical_formations = list(cluster_dict.values())

    def perform_hybrid_analysis(self) -> None:
        """Realiza análise híbrida combinando círculos e elipses"""
        self.logger.info("=== ANÁLISE HÍBRIDA ===")
        
        if self.params.auto_select_best_fit:
            self._auto_select_best_shapes()
        
        self._analyze_directional_patterns()
        self._detect_linear_formations()
        
        self.logger.info("✓ Análise híbrida concluída")
    
    def _auto_select_best_shapes(self) -> None:
        """Seleciona automaticamente a melhor representação (círculo vs elipse) para cada objeto"""
        self.logger.info("Selecionando melhores representações de forma...")
        
        shape_decisions = []
        
        for circle in self.detected_circles:
            decision = {
                'object_id': f"circle_{circle.get('contour_id', 'unknown')}",
                'original_shape': 'circle',
                'recommended_shape': 'circle',
                'confidence': 1.0,
                'reason': 'circular_detection'
            }
            
            corresponding_ellipse = self._find_corresponding_ellipse(circle)
            if corresponding_ellipse:
                if corresponding_ellipse['axis_ratio'] > self.params.elongation_threshold:
                    decision['recommended_shape'] = 'ellipse'
                    decision['confidence'] = corresponding_ellipse['fit_quality']
                    decision['reason'] = f'elongated_ratio_{corresponding_ellipse["axis_ratio"]:.2f}'
            
            shape_decisions.append(decision)
        
        self.shape_decisions = shape_decisions
        self.logger.info(f"✓ Analisadas {len(shape_decisions)} decisões de forma")
    
    def _find_corresponding_ellipse(self, circle: Dict) -> Optional[Dict]:
        """Encontra elipse correspondente a um círculo"""
        cx, cy = circle['center']
        threshold = self.params.duplicate_distance_threshold
        
        for ellipse in self.detected_ellipses:
            ex, ey = ellipse['center']
            distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)
            
            if distance < threshold:
                return ellipse
        
        return None
    
    def _analyze_directional_patterns(self) -> None:
        """Analisa padrões direcionais nas formações"""
        self.logger.info("Analisando padrões direcionais...")
        
        all_angles = []
        
        for ellipse in self.detected_ellipses:
            all_angles.append(ellipse['angle'])
        
        for formation in self.circular_formations + self.elliptical_formations:
            if len(formation) > 1:
                centers = np.array([obj['center'] for obj in formation])
                pca = PCA(n_components=2)
                pca.fit(centers)
                main_angle = np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0]))
                all_angles.append(main_angle % 180)
        
        if all_angles:
            self.directional_analysis = {
                'angles': all_angles,
                'mean_angle': np.mean(all_angles),
                'std_angle': np.std(all_angles),
                'angle_bins': np.histogram(all_angles, bins=self.params.angle_bins)[0].tolist(),
                'dominant_direction': self._find_dominant_direction(all_angles),
                'anisotropy_index': self._calculate_anisotropy_index(all_angles)
            }
        else:
            self.directional_analysis = {'angles': [], 'message': 'Nenhum padrão direcional detectado'}
        
        self.logger.info(f"✓ Análise direcional: {len(all_angles)} ângulos analisados")
    
    def _find_dominant_direction(self, angles: List[float]) -> Dict:
        """Encontra direção dominante nos padrões"""
        if not angles:
            return {'direction': None, 'strength': 0}
        
        hist, bin_edges = np.histogram(angles, bins=self.params.angle_bins, range=(0, 180))
        dominant_bin = np.argmax(hist)
        dominant_direction = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2
        strength = hist[dominant_bin] / len(angles)
        
        return {
            'direction': dominant_direction,
            'strength': strength,
            'frequency': int(hist[dominant_bin])
        }
    
    def _calculate_anisotropy_index(self, angles: List[float]) -> float:
        """Calcula índice de anisotropia (0 = isotrópico, 1 = altamente anisotrópico)"""
        if len(angles) < 2:
            return 0.0
        
        vectors = np.array([[np.cos(np.radians(angle)), np.sin(np.radians(angle))] for angle in angles])
        
        orientation_tensor = np.cov(vectors.T)
        eigenvalues = np.linalg.eigvals(orientation_tensor)
        
        if np.max(eigenvalues) > 0:
            anisotropy = 1 - np.min(eigenvalues) / np.max(eigenvalues)
        else:
            anisotropy = 0.0
        
        return float(anisotropy)
    
    def _detect_linear_formations(self) -> None:
        """Detecta formações especificamente lineares"""
        self.logger.info("Detectando formações lineares...")
        
        linear_formations = []
        
        for i, formation in enumerate(self.circular_formations):
            if len(formation) >= 3:
                linearity = self._calculate_formation_linearity(formation)
                if linearity > self.params.linearity_threshold:
                    linear_formations.append({
                        'formation_id': f'circular_formation_{i}',
                        'type': 'linear_circular',
                        'objects': formation,
                        'linearity': linearity,
                        'n_objects': len(formation)
                    })
        
        for i, formation in enumerate(self.elliptical_formations):
            if len(formation) >= 3:
                linearity = self._calculate_formation_linearity(formation)
                if linearity > self.params.linearity_threshold:
                    linear_formations.append({
                        'formation_id': f'elliptical_formation_{i}',
                        'type': 'linear_elliptical',
                        'objects': formation,
                        'linearity': linearity,
                        'n_objects': len(formation)
                    })
        
        self.linear_formations = linear_formations
        self.logger.info(f"✓ Detectadas {len(linear_formations)} formações lineares")
    
    def _calculate_formation_linearity(self, formation: List[Dict]) -> float:
        """Calcula linearidade de uma formação"""
        if len(formation) < 3:
            return 0.0
        
        centers = np.array([obj['center'] for obj in formation])
        
        pca = PCA(n_components=2)
        pca.fit(centers)
        
        explained_variance_ratio = pca.explained_variance_ratio_
        linearity = explained_variance_ratio[0] / (explained_variance_ratio[0] + explained_variance_ratio[1])
        
        return float(linearity)

    def analyze_hybrid_formations(self) -> pd.DataFrame:
        """Analisa parâmetros geométricos das formações híbridas"""
        self.logger.info("=== ANÁLISE GEOMÉTRICA HÍBRIDA ===")
        
        formation_data = []
        
        # Analisar círculos isolados
        for i, circle in enumerate(self.isolated_circles):
            formation_data.append(self._analyze_single_object(circle, f'isolated_circle_{i}', 'isolated_circle'))
        
        # Analisar elipses isoladas
        for i, ellipse in enumerate(self.isolated_ellipses):
            formation_data.append(self._analyze_single_object(ellipse, f'isolated_ellipse_{i}', 'isolated_ellipse'))
        
        # Analisar formações circulares
        for i, formation in enumerate(self.circular_formations):
            formation_data.append(self._analyze_formation(formation, f'circular_formation_{i}', 'circular_formation'))
        
        # Analisar formações elípticas
        for i, formation in enumerate(self.elliptical_formations):
            formation_data.append(self._analyze_formation(formation, f'elliptical_formation_{i}', 'elliptical_formation'))
        
        # Analisar formações lineares
        for i, linear_formation in enumerate(self.linear_formations):
            formation_data.append(self._analyze_linear_formation(linear_formation, f'linear_formation_{i}'))
        
        df_results = pd.DataFrame(formation_data)
        self.logger.info(f"✓ Análise híbrida concluída para {len(df_results)} objetos")
        
        return df_results
    
    def _analyze_single_object(self, obj: Dict, obj_id: str, obj_type: str) -> Dict:
        """Analisa um objeto individual"""
        base_analysis = {
            'id': obj_id,
            'type': obj_type,
            'shape_type': obj['shape_type'],
            'n_objects': 1,
            'center_x': obj['center'][0],
            'center_y': obj['center'][1],
            'area_total': obj['area'],
            'method': obj['method']
        }
        
        if obj['shape_type'] == 'circle':
            base_analysis.update({
                'radius_avg': obj['radius'],
                'radius_std': 0.0,
                'major_axis_avg': obj['radius'] * 2,
                'minor_axis_avg': obj['radius'] * 2,
                'eccentricity_avg': 0.0,
                'angle_avg': 0.0,
                'width': obj['radius'] * 2,
                'height': obj['radius'] * 2,
                'angle_deg': 0.0,
                'linearity': 1.0,
                'axis_ratio_avg': 1.0
            })
        else:  # ellipse
            base_analysis.update({
                'radius_avg': obj['radius'],
                'radius_std': 0.0,
                'major_axis_avg': obj['major_axis'],
                'minor_axis_avg': obj['minor_axis'],
                'eccentricity_avg': obj['eccentricity'],
                'angle_avg': obj['angle'],
                'width': obj['major_axis'],
                'height': obj['minor_axis'],
                'angle_deg': obj['angle'],
                'linearity': obj['eccentricity'],
                'axis_ratio_avg': obj.get('axis_ratio', 1.0)
            })
        
        return base_analysis
    
    def _analyze_formation(self, formation: List[Dict], formation_id: str, formation_type: str) -> Dict:
        """Analisa uma formação de objetos"""
        centers = np.array([obj['center'] for obj in formation])
        areas = [obj['area'] for obj in formation]
        
        min_x, min_y = np.min(centers, axis=0)
        max_x, max_y = np.max(centers, axis=0)
        
        avg_size = np.mean([obj.get('radius', obj.get('major_axis', 0)/2) for obj in formation])
        width = max_x - min_x + 2 * avg_size
        height = max_y - min_y + 2 * avg_size
        
        centroid_x = np.mean(centers[:, 0])
        centroid_y = np.mean(centers[:, 1])
        
        if len(centers) > 1:
            pca = PCA(n_components=2)
            pca.fit(centers)
            main_angle = np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0]))
            linearity = pca.explained_variance_ratio_[0] / np.sum(pca.explained_variance_ratio_)
        else:
            main_angle = 0.0
            linearity = 0.0
        
        if formation[0]['shape_type'] == 'circle':
            radii = [obj['radius'] for obj in formation]
            base_stats = {
                'radius_avg': np.mean(radii),
                'radius_std': np.std(radii),
                'major_axis_avg': np.mean(radii) * 2,
                'minor_axis_avg': np.mean(radii) * 2,
                'eccentricity_avg': 0.0,
                'angle_avg': 0.0,
                'axis_ratio_avg': 1.0
            }
        else:  # ellipses
            major_axes = [obj['major_axis'] for obj in formation]
            minor_axes = [obj['minor_axis'] for obj in formation]
            eccentricities = [obj['eccentricity'] for obj in formation]
            angles = [obj['angle'] for obj in formation]
            axis_ratios = [obj.get('axis_ratio', 1.0) for obj in formation]
            
            base_stats = {
                'radius_avg': np.mean([obj['radius'] for obj in formation]),
                'radius_std': np.std([obj['radius'] for obj in formation]),
                'major_axis_avg': np.mean(major_axes),
                'minor_axis_avg': np.mean(minor_axes),
                'eccentricity_avg': np.mean(eccentricities),
                'angle_avg': np.mean(angles),
                'axis_ratio_avg': np.mean(axis_ratios)
            }
        
        bbox_area = width * height if width > 0 and height > 0 else 1
        density = len(formation) / bbox_area
        
        return {
            'id': formation_id,
            'type': formation_type,
            'shape_type': formation[0]['shape_type'],
            'n_objects': len(formation),
            'center_x': centroid_x,
            'center_y': centroid_y,
            'area_total': sum(areas),
            'width': width,
            'height': height,
            'angle_deg': main_angle % 180,
            'linearity': linearity,
            'density': density,
            'method': ','.join(list(set([obj['method'] for obj in formation]))),
            **base_stats
        }
    
    def _analyze_linear_formation(self, linear_formation: Dict, formation_id: str) -> Dict:
        """Analisa especificamente uma formação linear"""
        formation = linear_formation['objects']
        
        base_analysis = self._analyze_formation(formation, formation_id, 'linear_formation')
        base_analysis.update({
            'linearity': linear_formation['linearity'],
            'formation_subtype': linear_formation['type']
        })
        
        return base_analysis

    def save_hybrid_images(self) -> None:
        """Salva imagens específicas da análise híbrida"""
        self.logger.info("Salvando imagens híbridas...")
        
        # Imagens básicas
        cv2.imwrite(str(self.images_dir / "01_original.png"), self.original_image)
        cv2.imwrite(str(self.images_dir / "02_grayscale.png"), self.gray_image)
        cv2.imwrite(str(self.images_dir / "03_binary.png"), self.binary_image)
        
        # Resultado híbrido
        result_img = self.original_image.copy()
        
        # Círculos isolados em verde
        for circle in self.isolated_circles:
            center = circle['center']
            radius = circle['radius']
            cv2.circle(result_img, center, radius, (0, 255, 0), 2)
            cv2.circle(result_img, center, 2, (0, 255, 0), -1)
        
        # Elipses isoladas em azul
        for ellipse in self.isolated_ellipses:
            center = ellipse['center']
            axes = (int(ellipse['major_axis']/2), int(ellipse['minor_axis']/2))
            angle = int(ellipse['angle'])
            cv2.ellipse(result_img, center, axes, angle, 0, 360, (255, 0, 0), 2)
            cv2.circle(result_img, center, 2, (255, 0, 0), -1)
        
        # Formações em cores diferentes
        colors = [(0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128)]
        
        # Formações circulares
        for i, formation in enumerate(self.circular_formations):
            color = colors[i % len(colors)]
            for circle in formation:
                center = circle['center']
                radius = circle['radius']
                cv2.circle(result_img, center, radius, color, 2)
                cv2.circle(result_img, center, 2, color, -1)
        
        # Formações elípticas
        for i, formation in enumerate(self.elliptical_formations):
            color = colors[(i + len(self.circular_formations)) % len(colors)]
            for ellipse in formation:
                center = ellipse['center']
                axes = (int(ellipse['major_axis']/2), int(ellipse['minor_axis']/2))
                angle = int(ellipse['angle'])
                cv2.ellipse(result_img, center, axes, angle, 0, 360, color, 2)
                cv2.circle(result_img, center, 2, color, -1)
        
        cv2.imwrite(str(self.images_dir / "04_hybrid_result.png"), result_img)
        
        self.logger.info("✓ Imagens híbridas salvas")
    
    def save_detailed_hybrid_report(self, df_results: pd.DataFrame) -> None:
        """Salva relatório detalhado híbrido com metadados completos"""
        
        execution_stats = {
            'total_circles_detected': len(self.detected_circles),
            'total_ellipses_detected': len(self.detected_ellipses),
            'isolated_circles': len(self.isolated_circles),
            'isolated_ellipses': len(self.isolated_ellipses),
            'circular_formations': len(self.circular_formations),
            'elliptical_formations': len(self.elliptical_formations),
            'linear_formations': len(self.linear_formations),
            'detection_methods_used': list(set([
                obj['method'] for obj in (self.detected_circles + self.detected_ellipses)
            ]))
        }
        
        report = {
            'analysis_metadata': self.analysis_metadata,
            'image_info': self.image_info,
            'parameters_used': asdict(self.params),
            'execution_statistics': execution_stats,
            'directional_analysis': self.directional_analysis,
            'shape_decisions': self.shape_decisions,
            'detailed_results': df_results.to_dict('records')
        }
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Salvar relatório JSON
        report_file = self.data_dir / f"hybrid_analysis_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # Salvar CSV dos resultados
        csv_file = self.data_dir / f"hybrid_results_{timestamp}.csv"
        df_results.to_csv(csv_file, index=False)
        
        # Salvar parâmetros
        params_file = self.data_dir / f"hybrid_parameters_{timestamp}.json"
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.params), f, indent=2)
        
        self.logger.info(f"✓ Relatórios híbridos salvos:")
        self.logger.info(f"  - Relatório completo: {report_file}")
        self.logger.info(f"  - Resultados CSV: {csv_file}")
        self.logger.info(f"  - Parâmetros: {params_file}")

    def visualize_hybrid_results(self, df_results: pd.DataFrame) -> None:
        """Visualiza e salva gráficos dos resultados híbridos"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Imagem original
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Imagem Original')
        axes[0, 0].axis('off')
        
        # Imagem binária
        axes[0, 1].imshow(self.binary_image, cmap='gray')
        axes[0, 1].set_title('Imagem Binária')
        axes[0, 1].axis('off')
        
        # Resultado híbrido
        result_img = self.original_image.copy()
        
        # Plotar objetos detectados
        for circle in self.isolated_circles:
            center = circle['center']
            radius = circle['radius']
            cv2.circle(result_img, center, radius, (0, 255, 0), 2)
        
        for ellipse in self.isolated_ellipses:
            center = ellipse['center']
            axes_size = (int(ellipse['major_axis']/2), int(ellipse['minor_axis']/2))
            angle = int(ellipse['angle'])
            cv2.ellipse(result_img, center, axes_size, angle, 0, 360, (255, 0, 0), 2)
        
        axes[0, 2].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Detecções: Verde=Círculos, Azul=Elipses')
        axes[0, 2].axis('off')
        
        # Histogramas
        if not df_results.empty:
            # Distribuição de excentricidades
            eccentricities = df_results['eccentricity_avg'].dropna()
            if len(eccentricities) > 0:
                axes[1, 0].hist(eccentricities, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Distribuição de Excentricidades')
                axes[1, 0].set_xlabel('Excentricidade')
                axes[1, 0].set_ylabel('Frequência')
            
            # Distribuição de ângulos
            angles = df_results[df_results['type'].str.contains('ellipse', na=False)]['angle_avg'].dropna()
            if len(angles) > 0:
                axes[1, 1].hist(angles, bins=18, alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Distribuição de Ângulos das Elipses')
                axes[1, 1].set_xlabel('Ângulo (graus)')
                axes[1, 1].set_ylabel('Frequência')
            
            # Rose diagram dos ângulos
            if self.directional_analysis.get('angles'):
                angles_rad = np.radians(self.directional_analysis['angles'])
                ax_polar = plt.subplot(3, 3, 6, projection='polar')
                ax_polar.hist(angles_rad, bins=self.params.angle_bins, alpha=0.7)
                ax_polar.set_title('Rose Diagram - Orientações')
                ax_polar.set_theta_zero_location('E')
                ax_polar.set_theta_direction(1)
        
        # Scatter plot: Excentricidade vs Razão de Aspecto
        if not df_results.empty:
            scatter_data = df_results.dropna(subset=['eccentricity_avg', 'axis_ratio_avg'])
            if len(scatter_data) > 0:
                axes[1, 2].scatter(scatter_data['axis_ratio_avg'], scatter_data['eccentricity_avg'], 
                                  alpha=0.7, c=scatter_data.index, cmap='viridis')
                axes[1, 2].set_xlabel('Razão de Aspecto')
                axes[1, 2].set_ylabel('Excentricidade')
                axes[1, 2].set_title('Relação Aspecto vs Excentricidade')
        
        # Estatísticas resumidas
        total_objects = len(df_results)
        circles = len(df_results[df_results['shape_type'] == 'circle'])
        ellipses = len(df_results[df_results['shape_type'] == 'ellipse'])
        
        stats_text = f"""
ESTATÍSTICAS HÍBRIDAS

Detecção:
• Total de objetos: {total_objects}
• Círculos: {circles}
• Elipses: {ellipses}
• Formações lineares: {len(self.linear_formations)}

Análise Direcional:
• Índice de anisotropia: {self.directional_analysis.get('anisotropy_index', 0):.3f}
• Direção dominante: {self.directional_analysis.get('dominant_direction', {}).get('direction', 'N/A'):.1f}°
• Força direção: {self.directional_analysis.get('dominant_direction', {}).get('strength', 0):.3f}

Parâmetros:
• Threshold binário: {self.params.binary_threshold}
• Max razão de aspecto: {self.params.max_ellipse_axis_ratio}
• Threshold linearidade: {self.params.linearity_threshold}
        """
        
        axes[2, 0].text(0.05, 0.95, stats_text.strip(), transform=axes[2, 0].transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace')
        axes[2, 0].set_title('Estatísticas Detalhadas')
        axes[2, 0].axis('off')
        
        # Gráfico de barras por tipo
        if not df_results.empty:
            type_counts = df_results['type'].value_counts()
            axes[2, 1].bar(range(len(type_counts)), type_counts.values)
            axes[2, 1].set_xticks(range(len(type_counts)))
            axes[2, 1].set_xticklabels(type_counts.index, rotation=45, ha='right')
            axes[2, 1].set_title('Distribuição por Tipo')
            axes[2, 1].set_ylabel('Quantidade')
        
        # Análise de linearidade
        if not df_results.empty:
            linearity_data = df_results['linearity'].dropna()
            if len(linearity_data) > 0:
                axes[2, 2].hist(linearity_data, bins=20, alpha=0.7, edgecolor='black')
                axes[2, 2].axvline(self.params.linearity_threshold, color='red', 
                                  linestyle='--', label=f'Threshold ({self.params.linearity_threshold})')
                axes[2, 2].set_title('Distribuição de Linearidade')
                axes[2, 2].set_xlabel('Índice de Linearidade')
                axes[2, 2].set_ylabel('Frequência')
                axes[2, 2].legend()
        
        plt.tight_layout()
        
        # Salvar gráfico
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = self.images_dir / f"hybrid_analysis_summary_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"✓ Gráfico híbrido salvo: {plot_file}")

    def run_complete_hybrid_analysis(self) -> pd.DataFrame:
        """Executa análise híbrida completa"""
        self.logger.info("=" * 70)
        self.logger.info("INICIANDO ANÁLISE HÍBRIDA COMPLETA - CÍRCULOS E ELIPSES")
        self.logger.info("=" * 70)
        
        try:
            # 1. Validação
            if not self.validate_inputs():
                raise ValueError("Validação de entrada falhou")
            
            # 2. Pré-processamento
            self.load_and_preprocess()
            
            # 3. Detecções circulares
            hough_circles = self.detect_circles_hough()
            contour_circles = self.detect_circles_contours()
            
            # 4. Detecções elípticas
            contour_ellipses = self.detect_ellipses_contours()
            moment_ellipses = self.detect_ellipses_moments()
            
            # 5. Merge de todas as detecções
            all_circles = hough_circles + contour_circles
            all_ellipses = contour_ellipses + moment_ellipses
            self.merge_all_detections(all_circles, all_ellipses)
            
            # 6. Clustering separado
            self.cluster_circles_traditional()
            self.cluster_ellipses_gmm()
            
            # 7. Análise híbrida
            self.perform_hybrid_analysis()
            
            # 8. Análise geométrica
            df_results = self.analyze_hybrid_formations()
            
            # 9. Salvar resultados
            self.save_hybrid_images()
            self.save_detailed_hybrid_report(df_results)
            
            # 10. Visualização
            self.visualize_hybrid_results(df_results)
            
            self.logger.info("=" * 70)
            self.logger.info("ANÁLISE HÍBRIDA CONCLUÍDA COM SUCESSO")
            self.logger.info("=" * 70)
            
            return df_results
            
        except Exception as e:
            self.logger.error(f"Erro durante a análise híbrida: {str(e)}")
            raise

# ==== CONFIGURAÇÃO E EXECUÇÃO HÍBRIDA ====

def create_hybrid_default_config() -> HybridAnalysisParameters:
    """Cria configuração padrão híbrida"""
    return HybridAnalysisParameters(
        # Parâmetros básicos
        binary_threshold=127,
        morphology_kernel_size=3,
        
        # Círculos
        hough_min_radius=3,
        hough_max_radius=25,
        min_circularity=0.6,
        
        # Elipses
        min_ellipse_axis_ratio=0.3,
        max_ellipse_axis_ratio=5.0,
        min_eccentricity=0.0,
        max_eccentricity=0.95,
        ellipse_fit_threshold=0.8,
        
        # Clustering
        circular_clustering_eps=35.0,
        gmm_max_components=8,
        
        # Análise direcional
        angle_bins=18,
        linearity_threshold=0.7,
        
        # Híbrido
        auto_select_best_fit=True,
        elongation_threshold=2.0
    )

if __name__ == "__main__":
    # ==== CONFIGURAÇÃO HÍBRIDA ====
    
    # Caminho da imagem (ALTERAR AQUI)
    IMAGE_PATH = "L:\\res\\santos\\ne_tupi\\er\\er04\\GEO_V5\\ESTUDOS\\temp\\Konopka\\Estudos\\2025.09_Karst\\imagens\\vug_config_R2.91mm_N225_D2.91mm_4E98AD.tif"
    
    # Diretório de saída
    OUTPUT_DIR = "L:\\res\\santos\\ne_tupi\\er\\er04\\GEO_V5\\ESTUDOS\\temp\\Konopka\\Estudos\\2025.09_Karst\\imagens\\hybrid_circle_ellipse_analysis"

    
    # Parâmetros personalizados
    custom_params = HybridAnalysisParameters(
        binary_threshold=127,
        hough_min_radius=3,
        hough_max_radius=25,
        min_ellipse_axis_ratio=0.5,      # Elipses não muito alongadas
        max_ellipse_axis_ratio=4.0,      # Permitir boa elongação
        min_eccentricity=0.3,            # Mínimo para considerar elipse
        max_eccentricity=0.9,            # Máximo antes de ser linha
        ellipse_fit_threshold=0.7,       # Qualidade mínima do fitting
        circular_clustering_eps=35.0,    # Distância para agrupar círculos
        gmm_max_components=8,            # Máximo de clusters elípticos
        linearity_threshold=0.8,         # Threshold para formação linear
        auto_select_best_fit=True,       # Seleção automática círculo vs elipse
        elongation_threshold=2.5         # Razão para preferir elipse
    )
    
    # ==== EXECUÇÃO HÍBRIDA ====
    try:
        print("Iniciando análise híbrida de círculos e elipses...")
        print(f"Imagem: {IMAGE_PATH}")
        print(f"Saída: {OUTPUT_DIR}")
        
        # Criar analisador híbrido
        analyzer = HybridCircleEllipseAnalyzer(
            image_path=IMAGE_PATH,
            output_dir=OUTPUT_DIR,
            params=custom_params
        )
        
        # Executar análise completa
        results_df = analyzer.run_complete_hybrid_analysis()
        
        # Exibir resumo detalhado
        print("\n" + "="*60)
        print("RESUMO DOS RESULTADOS HÍBRIDOS:")
        print("="*60)
        
        total_objects = len(results_df)
        circles = len(results_df[results_df['shape_type'] == 'circle'])
        ellipses = len(results_df[results_df['shape_type'] == 'ellipse'])
        
        print(f"Total de objetos detectados: {total_objects}")
        print(f"  • Círculos: {circles}")
        print(f"  • Elipses: {ellipses}")
        print(f"  • Formações lineares: {len(analyzer.linear_formations)}")
        
        if analyzer.directional_analysis.get('angles'):
            print(f"\nAnálise Direcional:")
            print(f"  • Índice de anisotropia: {analyzer.directional_analysis.get('anisotropy_index', 0):.3f}")
            dom_dir = analyzer.directional_analysis.get('dominant_direction', {})
            if dom_dir.get('direction'):
                print(f"  • Direção dominante: {dom_dir['direction']:.1f}° (força: {dom_dir['strength']:.3f})")
        
        print(f"\nArquivos salvos em: {OUTPUT_DIR}/")
        print("✓ Imagens de processo e resultado híbrido salvos")
        print("✓ Relatório detalhado com análise direcional salvo")
        print("✓ Resultados CSV com parâmetros elípticos salvos")
        print("✓ Parâmetros híbridos registrados")
        print("✓ Gráficos de análise avançada gerados")
        
    except FileNotFoundError:
        print(f"ERRO: Arquivo de imagem não encontrado: {IMAGE_PATH}")
        print("Por favor, verifique o caminho da imagem e tente novamente.")
        
    except Exception as e:
        print(f"ERRO durante a análise híbrida: {str(e)}")
        print("Verifique os logs na pasta de saída para mais detalhes.")
