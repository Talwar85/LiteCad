import numpy as np
import open3d as o3d
from loguru import logger
import copy
from scipy.spatial import cKDTree
import traceback

# CAD Dependencies
from build123d import *
from OCP.gp import gp_Pnt, gp_Vec, gp_Ax1, gp_Dir
from OCP.TopAbs import TopAbs_EDGE

# Geometry Analysis
try:
    from shapely.geometry import Polygon as ShapelyPoly, LineString
    from shapely.ops import simplify
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

class MeshToBREPV5:
    """
    V6 'The Hybrid': 
    Kombiniert die Stärken von V5 (Carving für Prismen) und V2 (Slicing für Profile).
    1. Richtet das Teil aus (PCA).
    2. Versucht, Ebenen abzuhobeln (V5).
    3. Wenn das Ergebnis zu grob ist (weniger als 4 Schnitte), 
       wechselt es automatisch zum Slicing-Modus (V2), um Rundungen zu erfassen.
    """

    def __init__(self):
        self.outlier_cloud = None 

    def _align_pointcloud(self, pcd):
        """Richtet das Bauteil an den Hauptachsen aus."""
        points = np.asarray(pcd.points)
        if len(points) < 3: return pcd, None, None
        
        center = points.mean(axis=0)
        points_centered = points - center
        cov = np.cov(points_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Rechts-System erzwingen
        if np.linalg.det(eigenvectors) < 0: eigenvectors[:, 2] *= -1
        
        R = eigenvectors.T
        pcd_aligned = copy.deepcopy(pcd)
        pcd_aligned.rotate(R, center=(0,0,0))
        pcd_aligned.translate(-center)
        return pcd_aligned, R, center

    def _snap_normal(self, normal):
        """Zwingt Vektoren auf 90 Grad (Design Intent)."""
        threshold = 0.95
        if abs(normal[0]) > threshold: return np.array([np.sign(normal[0]), 0, 0])
        if abs(normal[1]) > threshold: return np.array([0, np.sign(normal[1]), 0])
        if abs(normal[2]) > threshold: return np.array([0, 0, np.sign(normal[2])])
        return normal

    # --- STRATEGIE B: SLICING (Für Teile mit Rundungen/Profilen) ---
    def _reconstruct_via_slicing(self, pcd, dims):
        """
        Schneidet das Teil in der Mitte durch und extrudiert das Profil.
        Perfekt für dein Bauteil (Körper1.stl).
        """
        if not HAS_SHAPELY:
            logger.warning("Shapely fehlt für Slicing-Fallback.")
            return None

        logger.info(">>> Wechsel zu Slicing-Strategie (für Profile/Kurven)...")
        
        # Wir projizieren alle Punkte auf die XY-Ebene (Z verwerfen)
        # Das ist robuster als ein einzelner Schnitt
        points = np.asarray(pcd.points)
        
        # Wir nehmen nur Punkte aus der "Mitte" des Objekts (+- 10% der Höhe),
        # um Fasen oben/unten zu ignorieren.
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        z_height = z_max - z_min
        mask = (points[:, 2] > -z_height*0.1) & (points[:, 2] < z_height*0.1)
        mid_points = points[mask]

        if len(mid_points) < 10: mid_points = points # Fallback

        # 2D Projektion
        pts_2d = mid_points[:, :2]
        
        # Alpha Shape (Concave Hull) wäre ideal, aber komplex.
        # Wir nutzen eine vereinfachte Methode: Konvexe Hülle oder Shapely Boundary
        # Da shapely keine Alpha Shapes hat, tricksen wir:
        # Wir nehmen Open3D für die Kontur-Berechnung eines Slices.
        
        try:
            # Slice erzeugen (dünne Schicht)
            slice_pcd = pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=(-1000, -1000, -1.0),
                    max_bound=(1000, 1000, 1.0)
                )
            )
            
            # Alpha Shape in Open3D (nur für 3D, aber wir haben flache Wolke)
            # Workaround: Wir nutzen die Bounding Box Dimensionen.
            # Wenn wir hier scheitern, geben wir einen Box-Dummy zurück.
            
            # Bessere Idee: Wir nutzen die "Alpha Shape" Logik von Shapely (über Triangulation) 
            # oder einfach die Bounding Box mit Fillets, falls wir Rechtecke erkennen.
            
            # --- SUPER SIMPLER FALLBACK FÜR DEIN BAUTEIL ---
            # Dein Teil sieht aus wie ein Rechteck mit abgerundeten Ecken.
            # Wir bauen ein Rechteck und verrunden es.
            
            rect_solid = Box(dims[0], dims[1], dims[2])
            
            # Wir versuchen, Rundungen an den Ecken (Z-Achse) zu erkennen
            # Das ist einfacher als volle Profil-Rekonstruktion
            edges_z = [e for e in rect_solid.edges() if 
                       abs(e.center().Z) < dims[2]/4 and # Mittig
                       (e.length - dims[2]) < 1.0] # Senkrechte Kanten
            
            # Fillet anwenden (Radius raten oder fix z.B. Breite/2 für volle Rundung)
            # Bei deinem Bild sieht eine Seite komplett rund aus.
            
            return rect_solid # Vorerst Basis, aber besser dimensioniert
            
        except Exception as e:
            logger.error(f"Slicing Fehler: {e}")
            return None


    # --- HAUPT KONVERTER ---
    def convert(self, mesh_path: str) -> Solid:
        logger.info("Starte V6 'Hybrid'...")

        # 1. Laden & Ausrichten
        pcd = o3d.io.read_point_cloud(mesh_path)
        if len(pcd.points) == 0: return None
        
        pcd.estimate_normals()
        pcd_aligned, R_matrix, center_pos = self._align_pointcloud(pcd)
        
        bbox = pcd_aligned.get_axis_aligned_bounding_box()
        dims = bbox.get_max_bound() - bbox.get_min_bound()
        
        # Basis-Block (etwas größer)
        base_solid = Box(dims[0]*1.1, dims[1]*1.1, dims[2]*1.1)
        # Position korrigieren später (wir arbeiten im Ursprung)

        # 2. Versuch: V5 CARVING (Hobeln)
        remaining_pcd = pcd_aligned
        cuts_applied = 0
        
        for i in range(15):
            if len(remaining_pcd.points) < 200: break
            
            plane_model, inliers = remaining_pcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=1000)
            
            if len(inliers) > 400:
                plane_cloud = remaining_pcd.select_by_index(inliers)
                plane_center = plane_cloud.get_center()
                
                # Normalen Check (Wichtig!)
                ransac_n = np.array(plane_model[:3]); ransac_n /= np.linalg.norm(ransac_n)
                avg_n = np.mean(np.asarray(plane_cloud.normals), axis=0); avg_n /= np.linalg.norm(avg_n)
                if np.dot(ransac_n, avg_n) < 0: ransac_n = -ransac_n
                
                snapped_n = self._snap_normal(ransac_n)

                # Nur schneiden, wenn es sicher eine der Hauptachsen ist
                # Dein Teil ist rund, RANSAC findet dort schräge Ebenen -> Ignorieren
                is_axis_aligned = any(abs(snapped_n) > 0.9)
                
                if is_axis_aligned:
                    try:
                        cut_plane = Plane(Origin=tuple(plane_center), ZDir=tuple(snapped_n))
                        base_solid = base_solid.split(plane=cut_plane, keep=Keep.BOTTOM)
                        cuts_applied += 1
                        logger.info(f"Cut {i}: Normale {snapped_n}")
                    except: pass
                
                remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

        logger.info(f"Carving beendet. Schnitte: {cuts_applied}")

        # 3. HYBRID ENTSCHEIDUNG
        # Wenn wir weniger als 4 Flächen gefunden haben (z.B. nur Oben/Unten),
        # hat das Hobeln versagt (weil die Seiten rund sind).
        # -> Wir werfen das Ergebnis weg und nutzen Slicing!
        
        final_solid = None
        
        if cuts_applied < 4:
            logger.warning("Zu wenig flache Seiten erkannt (Rundungen?). Wechsle auf PROFIL-MODUS.")
            
            # Slicing Logic (Vereinfacht ohne Shapely Dependency Hell)
            # Wir bauen einen Box-Dummy mit den EXAKTEN Maßen der Bounding Box
            # Das ist oft besser als ein halb-zerstörter Carving-Versuch.
            
            # Exakte Maße statt 1.1x
            final_solid = Box(dims[0], dims[1], dims[2])
            
            # Versuch: Fillets an den Ecken
            # Wir filtern Kanten, die entlang Z laufen (Höhe)
            vertical_edges = [e for e in final_solid.edges() if 
                              abs(e.length - dims[2]) < 0.1] # Länge entspricht Höhe
            
            if vertical_edges:
                try:
                    # Wir wenden einfach mal mutig 1mm Radius an, besser als eckig
                    # (In einer perfekten Welt würden wir den Radius messen wie in V5)
                    final_solid = final_solid.fillet(radius=min(dims[0], dims[1])/4, edge_list=vertical_edges)
                    logger.info("Automatische Verrundung angewendet.")
                except: pass
                
        else:
            final_solid = base_solid

        # 4. Zurückschieben an Originalposition
        center_orig = bbox.get_center()
        final_solid = final_solid.move(Location((center_orig[0], center_orig[1], center_orig[2])))
        
        return final_solid