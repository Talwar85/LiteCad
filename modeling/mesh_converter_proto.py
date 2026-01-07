import numpy as np
import open3d as o3d
from loguru import logger
import copy
from scipy.spatial import cKDTree

# CAD Kern
from build123d import *
from OCP.gp import gp_Pnt, gp_Vec, gp_Ax1, gp_Dir
from OCP.TopAbs import TopAbs_EDGE

class MeshToBREPV5:
    """
    V5 'The Detailer'
    Kombiniert V4 (CSG Construction) mit einer intelligenten Kanten-Analyse,
    um Rundungen (Fillets) und Fasen (Chamfers) wiederherzustellen.
    """

    def __init__(self):
        self.outlier_cloud = None # Hier speichern wir die Punkte der Rundungen

    def _align_pointcloud(self, pcd):
        """(Wie in V4: PCA Ausrichtung)"""
        points = np.asarray(pcd.points)
        center = points.mean(axis=0)
        points_centered = points - center
        cov = np.cov(points_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        if np.linalg.det(eigenvectors) < 0: eigenvectors[:, 2] *= -1
        R = eigenvectors.T
        pcd_aligned = copy.deepcopy(pcd)
        pcd_aligned.rotate(R, center=(0,0,0))
        pcd_aligned.translate(-center)
        return pcd_aligned, R, center

    def _snap_normal(self, normal):
        """(Wie in V4: Macht krumme Wände gerade)"""
        threshold = 0.95
        if abs(normal[0]) > threshold: return np.array([np.sign(normal[0]), 0, 0])
        if abs(normal[1]) > threshold: return np.array([0, np.sign(normal[1]), 0])
        if abs(normal[2]) > threshold: return np.array([0, 0, np.sign(normal[2])])
        return normal

    def _estimate_fillet_radius(self, edge, kdtree, sample_points=10):
        """
        Der Kern von V5:
        Prüft eine Kante gegen die Punktwolke. 
        Wenn Punkte 'in der Kurve' liegen, schätzen wir den Radius.
        """
        # 1. Kanten-Geometrie holen
        # Wir sampeln Punkte entlang der Kante
        curve = edge.wrapped.Curve()
        f, l = edge.wrapped.FirstParameter(), edge.wrapped.LastParameter()
        
        distances = []
        
        # Wir prüfen an 10 Stellen der Kante
        for i in np.linspace(f, l, sample_points):
            p_edge = curve.Value(i) # OCP Point
            p_np = np.array([p_edge.X(), p_edge.Y(), p_edge.Z()])
            
            # Suche Punkte im Umkreis von z.B. 10mm (Fillet Search Radius)
            # Wir suchen im KDTree der "Outlier Points" (die wir beim Plane Cutting weggeworfen haben)
            idx = kdtree.query_ball_point(p_np, r=10.0)
            
            if not idx: continue
            
            # Hole die Punkte
            nearby_points = self.outlier_cloud[idx]
            
            # Berechne den minimalen Abstand dieser Punktwolke zur Kante
            # Bei einem Fillet sind die Punkte NICHT auf der Kante, sondern "innen".
            # Der Abstand einer 90° Ecke zur Oberfläche eines Fillets mit Radius R 
            # beträgt R * (sqrt(2) - 1) bei 45° Schnitt.
            # Vereinfacht: Wir messen den Abstand der innersten Punkte zur theoretischen Ecke.
            
            # Distanz jedes Punktes zur Kanten-Position
            dists = np.linalg.norm(nearby_points - p_np, axis=1)
            # Wir interessieren uns für die Punkte, die am nächsten an der Kante sind, 
            # aber nicht AUF der Kante (Noise)
            
            # Heuristik: Der Durchschnittsabstand der Punkte, die 'nahe' sind.
            avg_dist = np.mean(dists)
            distances.append(avg_dist)

        if not distances: return 0.0
        
        # Median der Abstände über die Kantenlänge
        metric = np.median(distances)
        
        # Rückrechnung auf Radius (für 90° Ecken)
        # Formel: Distance_Corner_to_Surface = Radius * (sqrt(2) - 1)  (~0.414)
        # Radius = Distance / 0.414
        estimated_radius = metric / 0.414
        
        # Plausibilitätscheck: Runde auf 0.5mm Schritte (Design Intent)
        if 0.5 < estimated_radius < 20.0: # Filtert Rauschen (<0.5) und Riesiges (>20)
            return round(estimated_radius * 2) / 2
        
        return 0.0

    def convert(self, mesh_path: str) -> Solid:
        logger.info("Starte V5 'The Detailer'...")

        # 1. Setup & Alignment (Wie V4)
        pcd = o3d.io.read_point_cloud(mesh_path)
        pcd.estimate_normals()
        pcd_aligned, _, _ = self._align_pointcloud(pcd)
        
        bbox = pcd_aligned.get_axis_aligned_bounding_box()
        dims = bbox.get_max_bound() - bbox.get_min_bound()
        
        # Base Box
        base_solid = Box(dims[0]+10, dims[1]+10, dims[2]+10)
        center_pos = bbox.get_center()
        base_solid = base_solid.move(Location((center_pos[0], center_pos[1], center_pos[2])))

        # 2. Plane Cutting (Der "Hobel")
        remaining_pcd = pcd_aligned
        all_points = np.asarray(pcd_aligned.points)
        inlier_indices_total = [] # Indices die wir verbraucht haben

        for i in range(15):
            if len(remaining_pcd.points) < 100: break
            
            plane_model, inliers = remaining_pcd.segment_plane(distance_threshold=0.25, ransac_n=3, num_iterations=1000)
            
            if len(inliers) > 500:
                # Merken welche Punkte wir genutzt haben (keine Fillets)
                # (Hier vereinfacht, da Indices sich bei select_by_index verschieben, 
                #  müsste man eigentlich global tracken. Wir nutzen später einfach "Restmenge")
                
                normal = self._snap_normal(np.array(plane_model[:3]))
                plane_cloud = remaining_pcd.select_by_index(inliers)
                center = plane_cloud.get_center()
                
                try:
                    cut_plane = Plane(Origin=tuple(center), ZDir=tuple(normal))
                    base_solid = base_solid.split(plane=cut_plane, keep=Keep.BOTTOM)
                except: pass
                
                remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

        # 3. FILLET PREPARATION
        # Die Punkte, die jetzt noch in 'remaining_pcd' sind, sind meistens die Rundungen!
        self.outlier_cloud = np.asarray(remaining_pcd.points)
        logger.info(f"{len(self.outlier_cloud)} Punkte für Detail-Analyse (Rundungen) übrig.")
        
        if len(self.outlier_cloud) > 50:
            # KDTree bauen für schnelle Suche
            kdtree = cKDTree(self.outlier_cloud)
            
            # Wir iterieren über ALLE Kanten des aktuellen Solids
            edges_to_fillet = []
            
            for edge in base_solid.edges():
                # Fillet nur an geraden Linien, keine Kreise (Bohrungskanten)
                if edge.geom_type in ["LINE"]: 
                    r = self._estimate_fillet_radius(edge, kdtree)
                    if r > 0:
                        logger.info(f"Kante erkannt -> Radius {r}mm")
                        edges_to_fillet.append((edge, r))

            # 4. APPLY FILLETS
            # Wir gruppieren nach Radius, um Build123d Aufrufe zu bündeln
            from collections import defaultdict
            radius_groups = defaultdict(list)
            for e, r in edges_to_fillet:
                radius_groups[r].append(e)
            
            for r, edges in radius_groups.items():
                try:
                    # Vorsicht: Fillets können scheitern, wenn Ecken sich treffen
                    base_solid = base_solid.fillet(radius=r, edge_list=edges)
                    logger.success(f"Fillet R={r}mm auf {len(edges)} Kanten angewendet.")
                except Exception as e:
                    logger.warning(f"Fillet R={r}mm fehlgeschlagen (Geometrie zu komplex): {e}")

        # 5. DRILLING (Bohrungen wie in V4) - Code hier gekürzt der Übersicht halber
        # ... (Hier den Cylinder Code aus V4 einfügen) ...
        # --- 4. CYLINDER DETECTION (Bohren) ---
        # Zylinder suchen wir im Original-PCD (aligned), da wir oben Punkte gelöscht haben
        cyl_search_pcd = pcd_aligned
        
        # Random Downsampling für Speed, Zylinder Fitting ist teuer
        cyl_search_pcd = cyl_search_pcd.voxel_down_sample(voxel_size=0.5)
        
        if len(cyl_search_pcd.points) > 100:
             # Hier nutzen wir RANSAC Cylinder Fitting
             # Leider hat Open3D Python API noch kein natives RANSAC Cylinder (nur C++).
             # Wir improvisieren oder nutzen pyransac3d wenn vorhanden.
             try:
                 import pyransac3d as pyr
                 
                 points_np = np.asarray(cyl_search_pcd.points)
                 # Wir suchen iterativ nach Löchern
                 for k in range(5): # Max 5 Löcher
                     cyl = pyr.Cylinder()
                     center, axis, radius, inliers = cyl.fit(points_np, thresh=0.2)
                     
                     if radius < 50.0 and len(inliers) > 200:
                         # Snapping der Achse
                         axis = self._snap_normal(np.array(axis))
                         
                         logger.info(f"Bohrung erkannt: r={radius:.2f}mm, axis={axis}")
                         
                         # --- APPLY DRILL ---
                         # Zylinder erzeugen und abziehen
                         # Wir machen ihn lang genug
                         hole_tool = Cylinder(radius=radius, height=200, align=(Align.CENTER, Align.CENTER, Align.CENTER))
                         
                         # Ausrichten des Werkzeugs
                         # Rotation von (0,0,1) zur Zielachse berechnen
                         # Das ist in build123d etwas tricky, wir nutzen Locations
                         
                         # Einfacher Workaround: Wir nutzen OCP direkt für die Platzierung oder
                         # bauen den Zylinder entlang Z und rotieren ihn.
                         
                         # Annahme: Bohrung ist Z-Aligned (typisch für 3D Druck)
                         if abs(axis[2]) > 0.9:
                             with Locations((center[0], center[1], center[2])):
                                 base_solid -= Cylinder(radius=radius, height=dims[2]*3)
                         elif abs(axis[1]) > 0.9: # Y-Achse
                             with Locations((center[0], center[1], center[2])):
                                 base_solid -= Cylinder(radius=radius, height=dims[1]*3, rotation=(90,0,0))
                         elif abs(axis[0]) > 0.9: # X-Achse
                             with Locations((center[0], center[1], center[2])):
                                 base_solid -= Cylinder(radius=radius, height=dims[0]*3, rotation=(0,90,0))
                                 
                         # Punkte entfernen
                         mask = np.ones(len(points_np), dtype=bool)
                         mask[inliers] = False
                         points_np = points_np[mask]
                         if len(points_np) < 100: break
             except ImportError:
                 logger.warning("pyransac3d fehlt. Keine Bohrloch-Erkennung.")
             except Exception as e:
                 logger.warning(f"Bohren fehlgeschlagen: {e}")
        return base_solid