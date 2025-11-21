import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import time

class ImageSimilaritySystem:
    """
    Système de recherche d'images similaires utilisant l'extraction de features
    et différentes métriques de similarité.
    """
    
    def __init__(self, dataset_path="archive/animals/", image_size=(128, 128)):
        """
        Initialise le système de recherche d'images.

        Args:
            dataset_path: Chemin vers le dossier contenant les images (par défaut: archive/animals/)
            image_size: Taille de redimensionnement des images (largeur, hauteur)
        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.images = []
        self.image_paths = []
        self.features = None
        self.scaler = StandardScaler()

        # Télécharger le dataset si nécessaire
        self._download_dataset_if_needed()

    def _download_dataset_if_needed(self):
        """
        Télécharge le dataset depuis Kaggle si le dossier n'existe pas ou est vide.
        """
        # Vérifier si le dossier existe et contient des images
        has_images = False
        if os.path.exists(self.dataset_path):
            supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    if file.lower().endswith(supported_formats):
                        has_images = True
                        break
                if has_images:
                    break

        if not has_images:
            print("📥 Téléchargement du dataset depuis Kaggle...")
            start_time = time.time()
            try:
                # Créer le dossier parent si nécessaire
                os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)

                # Télécharger le dataset
                import kaggle
                kaggle.api.dataset_download_files(
                    'ashishsaxena2209/animal-image-datasetdog-cat-and-panda',
                    path=os.path.dirname(self.dataset_path),
                    unzip=True
                )

                elapsed = time.time() - start_time
                print(f"✅ Dataset téléchargé dans {self.dataset_path} en {elapsed:.2f}s")

            except Exception as e:
                print(f"❌ Erreur lors du téléchargement: {e}")
                print("Vérifiez que vous avez configuré votre API Kaggle (kaggle.json)")
                print("Ou placez manuellement le dataset dans le dossier archive/")
                raise
        else:
            print("✅ Dataset déjà présent")
        
    def load_dataset(self):
        """
        Charge toutes les images du dataset depuis le dossier spécifié.
        Supporte les sous-dossiers (train, test, validation).
        """
        print("📁 Chargement du dataset...")
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        
        # Parcourir tous les sous-dossiers
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(supported_formats):
                    img_path = os.path.join(root, file)
                    self.image_paths.append(img_path)
        
        print(f"✅ {len(self.image_paths)} images trouvées dans le dataset")
        
    def preprocess_images(self):
        """
        Prétraite toutes les images : redimensionnement et normalisation.
        """
        print("🔄 Prétraitement des images...")
        start_time = time.time()
        
        for idx, img_path in enumerate(self.image_paths):
            try:
                # Charger l'image
                img = Image.open(img_path).convert('RGB')
                
                # Redimensionner
                img = img.resize(self.image_size, Image.Resampling.LANCZOS)
                
                # Convertir en array numpy et normaliser [0, 1]
                img_array = np.array(img) / 255.0
                
                self.images.append(img_array)
                
                if (idx + 1) % 100 == 0:
                    print(f"   Traité: {idx + 1}/{len(self.image_paths)} images")
                    
            except Exception as e:
                print(f"⚠️  Erreur lors du traitement de {img_path}: {e}")
        
        self.images = np.array(self.images)
        elapsed = time.time() - start_time
        print(f"✅ Prétraitement terminé en {elapsed:.2f}s")
        
    def extract_features(self, method='histogram'):
        """
        Extrait les caractéristiques des images.

        Args:
            method: 'histogram' pour histogramme de couleurs
                   'raw' pour pixels bruts aplatis
                   'color_moments' pour moments statistiques de couleur
        """
        if len(self.images) == 0:
            print("⚠️  Aucune image trouvée dans le dataset. Impossible d'extraire les features.")
            return

        print(f"🎯 Extraction des features (méthode: {method})...")
        start_time = time.time()

        if method == 'histogram':
            # Histogramme de couleurs RGB (32 bins par canal)
            features_list = []
            for img in self.images:
                hist_r = np.histogram(img[:,:,0], bins=32, range=(0, 1))[0]
                hist_g = np.histogram(img[:,:,1], bins=32, range=(0, 1))[0]
                hist_b = np.histogram(img[:,:,2], bins=32, range=(0, 1))[0]
                features = np.concatenate([hist_r, hist_g, hist_b])
                features_list.append(features)
            self.features = np.array(features_list)

        elif method == 'raw':
            # Pixels bruts aplatis
            self.features = self.images.reshape(len(self.images), -1)

        elif method == 'color_moments':
            # Moments statistiques (moyenne, variance, skewness) par canal
            features_list = []
            for img in self.images:
                moments = []
                for c in range(3):  # Pour chaque canal RGB
                    channel = img[:,:,c].flatten()
                    mean = np.mean(channel)
                    std = np.std(channel)
                    skew = np.mean((channel - mean) ** 3) / (std ** 3 + 1e-10)
                    moments.extend([mean, std, skew])
                features_list.append(moments)
            self.features = np.array(features_list)

        # Normalisation des features
        self.features = self.scaler.fit_transform(self.features)

        elapsed = time.time() - start_time
        print(f"✅ Features extraites: {self.features.shape} en {elapsed:.2f}s")
        
    def compute_similarity(self, reference_idx, method='cosine'):
        """
        Calcule la similarité entre une image de référence et toutes les autres.
        
        Args:
            reference_idx: Index de l'image de référence
            method: 'cosine' pour similarité cosinus
                   'euclidean' pour distance euclidienne
        
        Returns:
            Array des scores de similarité
        """
        reference_features = self.features[reference_idx].reshape(1, -1)
        
        if method == 'cosine':
            # Similarité cosinus (1 = identique, 0 = orthogonal)
            similarities = cosine_similarity(reference_features, self.features)[0]
            
        elif method == 'euclidean':
            # Distance euclidienne (0 = identique, plus grand = plus différent)
            # On inverse pour avoir un score de similarité
            distances = euclidean_distances(reference_features, self.features)[0]
            # Normaliser et inverser: similarité = 1 / (1 + distance)
            similarities = 1 / (1 + distances)
        
        return similarities
    
    def find_top_k_similar(self, reference_idx, k=5, similarity_method='cosine'):
        """
        Trouve les K images les plus similaires à une image de référence.
        
        Args:
            reference_idx: Index de l'image de référence
            k: Nombre d'images similaires à retourner
            similarity_method: Méthode de calcul de similarité
        
        Returns:
            indices: Indices des K images les plus similaires
            scores: Scores de similarité correspondants
        """
        similarities = self.compute_similarity(reference_idx, similarity_method)
        
        # Trier par similarité décroissante et prendre les top-k
        # (on exclut l'image elle-même si elle est dans les résultats)
        top_indices = np.argsort(similarities)[::-1]
        
        # Filtrer l'image de référence elle-même
        top_indices = [idx for idx in top_indices if idx != reference_idx][:k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def display_results(self, reference_idx, top_indices, top_scores, 
                       similarity_method, feature_method):
        """
        Affiche l'image de référence et les images similaires trouvées.
        """
        k = len(top_indices)
        fig, axes = plt.subplots(2, k + 1, figsize=(3 * (k + 1), 6))
        
        # Titre principal
        fig.suptitle(f'Recherche d\'images similaires\n'
                    f'Features: {feature_method} | Similarité: {similarity_method}',
                    fontsize=14, fontweight='bold')
        
        # Image de référence
        axes[0, 0].imshow(self.images[reference_idx])
        axes[0, 0].set_title('Image de\nRÉFÉRENCE', fontweight='bold', color='red')
        axes[0, 0].axis('off')
        
        ref_name = os.path.basename(self.image_paths[reference_idx])
        axes[1, 0].text(0.5, 0.5, f'{ref_name}', 
                       ha='center', va='center', wrap=True, fontsize=8)
        axes[1, 0].axis('off')
        
        # Images similaires
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            axes[0, i + 1].imshow(self.images[idx])
            axes[0, i + 1].set_title(f'Top-{i+1}\nScore: {score:.4f}', 
                                    fontsize=10)
            axes[0, i + 1].axis('off')
            
            img_name = os.path.basename(self.image_paths[idx])
            axes[1, i + 1].text(0.5, 0.5, f'{img_name}', 
                               ha='center', va='center', wrap=True, fontsize=8)
            axes[1, i + 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def compare_methods(self, reference_idx, k=5):
        """
        Compare différentes méthodes de similarité et d'extraction de features.
        """
        print("\n" + "="*70)
        print("🔬 COMPARAISON DES MÉTHODES")
        print("="*70)
        
        feature_methods = ['histogram', 'color_moments']
        similarity_methods = ['cosine', 'euclidean']
        
        results = {}
        
        for feat_method in feature_methods:
            print(f"\n📊 Extraction de features: {feat_method}")
            self.extract_features(method=feat_method)
            
            for sim_method in similarity_methods:
                print(f"   ↳ Calcul de similarité: {sim_method}")
                start_time = time.time()
                
                top_indices, top_scores = self.find_top_k_similar(
                    reference_idx, k, sim_method
                )
                
                elapsed = time.time() - start_time
                
                key = f"{feat_method}_{sim_method}"
                results[key] = {
                    'indices': top_indices,
                    'scores': top_scores,
                    'time': elapsed
                }
                
                print(f"      ⏱️  Temps: {elapsed:.4f}s")
                print(f"      📈 Scores moyens: {np.mean(top_scores):.4f}")
                
                # Afficher les résultats
                self.display_results(reference_idx, top_indices, top_scores,
                                   sim_method, feat_method)
        
        return results
    
    def analyze_performance(self, results):
        """
        Analyse et affiche un résumé des performances des différentes méthodes.
        """
        print("\n" + "="*70)
        print("📊 ANALYSE DE PERFORMANCE")
        print("="*70)

        for method_name, data in results.items():
            parts = method_name.split('_')
            feat = '_'.join(parts[:-1])
            sim = parts[-1]
            print(f"\n🔸 {feat.upper()} + {sim.upper()}")
            print(f"   Temps d'exécution: {data['time']:.4f}s")
            print(f"   Score moyen: {np.mean(data['scores']):.4f}")
            print(f"   Score min: {np.min(data['scores']):.4f}")
            print(f"   Score max: {np.max(data['scores']):.4f}")


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    """
    Script principal pour tester le système de recherche d'images similaires.
    
    INSTRUCTIONS:
    1. Modifiez 'dataset_path' avec le chemin vers votre dataset
    2. Le dataset doit contenir des images (jpg, png, etc.)
    3. Ajustez 'reference_idx' pour changer l'image de référence
    4. Modifiez 'k' pour changer le nombre de résultats
    """
    
    # ⚙️ CONFIGURATION
    dataset_path = "archive/animals/"  # À MODIFIER
    reference_idx =  2000 # Index de l'image de référence
    k = 5  # Nombre d'images similaires à trouver
    
    # 🚀 EXÉCUTION
    print("="*70)
    
    # Initialiser le système
    system = ImageSimilaritySystem(dataset_path, image_size=(128, 128))
    
    # Charger et prétraiter
    system.load_dataset()
    system.preprocess_images()
    
    # Comparer les différentes méthodes
    results = system.compare_methods(reference_idx, k=k)
    
    # Analyser les performances
    system.analyze_performance(results)
    
    print("\n✅ Analyse terminée!")
    print("="*70)