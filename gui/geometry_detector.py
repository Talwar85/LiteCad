import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

# Optional Imports
try:
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union, polygonize
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import pyvista as pv
    import vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False

@dataclass
class DetectedFace:
    id: int
    type: str  # 'sketch' oder 'body'
    
    # Für Sketch Faces
    sketch_id: Optional[str] = None
    shapely_poly: Any = None  # Das 2D Profil (inkl. Löcher)
    plane_origin: Tuple[float, float, float] = (0,0,0)
    plane_normal: Tuple[float, float, float] = (0,0,1)
    
    # Für Body Faces
    body_id: Optional[str] = None
    mesh_cells: List[int] = field(default_factory=list) # Indizes der Dreiecke im Body-Mesh
    center: Tuple[float, float, float] = (0,0,0)
    
    # Visualisierung
    display_mesh: Any = None # pv.PolyData für das Overlay im Viewport

@dataclass
class DetectedEdge:
    id: int
    body_id: str
    edge_type: str # 'boundary' oder 'feature'
    points: List[Tuple[float, float, float]]
    vtk_cells: List[int] # Referenz zur Linie im Edge-Mesh

class GeometryDetector:
    def __init__(self):
        self.faces: List[DetectedFace] = []
        self.edges: List[DetectedEdge] = []
        self._face_map = {} # Mapping für schnelles Picking
        self._counter = 0

    def clear(self):
        self.faces.clear()
        self.edges.clear()
        self._face_map.clear()
        self._counter = 0

    def process_sketch(self, sketch, plane_origin, plane_normal, plane_x_dir):
        """
        Analysiert einen Sketch und extrahiert geschlossene Profile (Faces).
        Behandelt verschachtelte Inseln (Löcher) korrekt.
        """
        if not HAS_SHAPELY: return
        
        # 1. Segmente sammeln (Code analog zu modeling.py, aber zentralisiert)
        lines = []
        def rnd(x): return round(x, 5)
        
        # ... (Hier die Sammel-Logik der Linien aus sketch.lines, arcs, etc. einfügen) ...
        # Vereinfacht für dieses Beispiel:
        from shapely.geometry import LineString
        for l in sketch.lines:
            if not l.construction:
                lines.append(LineString([(rnd(l.start.x), rnd(l.start.y)), (rnd(l.end.x), rnd(l.end.y))]))
        # ... (Arcs/Circles analog hinzufügen) ...
        
        if not lines: return

        try:
            # 2. Polygonize
            merged = unary_union(lines)
            raw_polys = list(polygonize(merged))
            
            # 3. Parent-Child Analyse für Löcher
            # Sortiere nach Fläche (groß zuerst)
            raw_polys.sort(key=lambda p: p.area, reverse=True)
            
            final_polys = []
            used_indices = set()
            
            for i, parent in enumerate(raw_polys):
                if i in used_indices: continue
                
                # Prüfe ob parent in einem noch größeren liegt (sollte nicht passieren durch Sortierung und Logik)
                hole_list = []
                
                for j, child in enumerate(raw_polys):
                    if i == j or j in used_indices: continue
                    
                    # Wenn child im parent liegt -> Es ist ein Loch
                    if parent.contains(child):
                        hole_list.append(child)
                        used_indices.add(j)
                
                # Erstelle finales Polygon mit Löchern
                shell = parent.exterior
                holes = [h.exterior for h in hole_list]
                final_poly = Polygon(shell, holes)
                final_polys.append(final_poly)

            # 4. In DetectedFace umwandeln & Triangulieren für Anzeige
            for poly in final_polys:
                if poly.area < 1e-6: continue
                
                # Triangulierung für den Viewport (Overlay)
                display_mesh = self._shapely_to_pv_mesh(poly, plane_origin, plane_normal, plane_x_dir)
                
                face = DetectedFace(
                    id=self._counter,
                    type='sketch',
                    sketch_id=sketch.id,
                    shapely_poly=poly,
                    plane_origin=plane_origin,
                    plane_normal=plane_normal,
                    center=self._transform_2d_3d(poly.centroid.x, poly.centroid.y, plane_origin, plane_normal, plane_x_dir),
                    display_mesh=display_mesh
                )
                self.faces.append(face)
                self._counter += 1
                
        except Exception as e:
            print(f"Detector Sketch Error: {e}")

    def process_body_mesh(self, body_id, vtk_mesh):
        """
        Extrahiert planare Flächen getrennt nach räumlicher Zusammengehörigkeit.
        Verhindert, dass separate Flächen mit gleicher Normale zusammen selektiert werden.
        """
        if not HAS_VTK or vtk_mesh is None: return
        
        # Sicherstellen, dass Zell-Normalen vorhanden sind
        if 'Normals' not in vtk_mesh.cell_data:
            vtk_mesh.compute_normals(cell_normals=True, inplace=True)
            
        normals = vtk_mesh.cell_data['Normals']
        rounded = np.round(normals, 2)
        unique_normals, indices = np.unique(rounded, axis=0, return_inverse=True)
        
        for group_idx, normal in enumerate(unique_normals):
            # 1. Alle Zellen mit dieser Normalen extrahieren
            cell_indices_in_group = np.where(indices == group_idx)[0]
            temp_mesh = vtk_mesh.extract_cells(cell_indices_in_group)
            
            # 2. Räumliche Trennung (Connectivity)
            # Teilt z.B. zwei separate Würfel-Oberseiten in unterschiedliche Regionen auf
            regions = temp_mesh.connectivity(extraction_mode='all')
            region_ids = regions.get_array("RegionId")
            n_regions = int(region_ids.max() + 1) if len(region_ids) > 0 else 0
            
            for r in range(n_regions):
                # Extrahiere die einzelne zusammenhängende Fläche (Region)
                region_mesh = regions.threshold([r, r], scalars="RegionId").extract_surface()
                
                if region_mesh.n_points == 0: continue
                
                # Zentrum für das Picking berechnen
                center = np.mean(region_mesh.points, axis=0)
                
                face = DetectedFace(
                    id=self._counter,
                    type='body',
                    body_id=body_id,
                    # Wir speichern die Zell-Indizes der Gruppe für den Kernel
                    mesh_cells=cell_indices_in_group.tolist(), 
                    center=tuple(center),
                    display_mesh=region_mesh,
                    plane_normal=tuple(normal)
                )
                self.faces.append(face)
                self._counter += 1

    def detect_edges(self, body_id, vtk_mesh):
        """Erkennt Kanten für Fillet/Chamfer"""
        if vtk_mesh is None: return
        
        feature_edges = vtk_mesh.extract_feature_edges(feature_angle=30)
        
        # Auch hier: Connectivity, um einzelne Kanten-Loops zu finden
        conn = feature_edges.connectivity(extraction_mode='all')
        
        # Wir speichern das gesamte Edge-Mesh, aber segmentiert wäre besser
        # Vereinfacht:
        edge = DetectedEdge(
            id=self._counter,
            body_id=body_id,
            edge_type='feature',
            points=feature_edges.points.tolist(),
            vtk_cells=[]
        )
        self.edges.append(edge)
        self._counter += 1

    def pick_face(self, ray_origin, ray_dir, candidates_indices=None) -> int:
        best_dist = float('inf')
        best_id = -1
        
        indices = candidates_indices if candidates_indices is not None else range(len(self.faces))
        
        start = np.array(ray_origin)
        end = start + np.array(ray_dir) * 10000.0
        
        for i in indices:
            face = self.faces[i]
            if face.display_mesh is None: continue
            
            # FIX: ray_trace funktioniert nur auf PolyData. 
            # Falls display_mesh ein UnstructuredGrid ist, konvertieren wir es zu PolyData.
            mesh_to_trace = face.display_mesh
            if not isinstance(mesh_to_trace, pv.PolyData):
                mesh_to_trace = mesh_to_trace.extract_surface()
            
            points, ind = mesh_to_trace.ray_trace(start, end)
            
            if len(points) > 0:
                dist = np.linalg.norm(points[0] - start)
                if dist < best_dist:
                    best_dist = dist
                    best_id = face.id # Nutze die gespeicherte ID
                    
        return best_id

    # --- Helper ---
    def _transform_2d_3d(self, x, y, o, n, x_dir=None):
        # ... (Transformation wie gehabt) ...
        # Nur Placeholder Logik
        return (o[0]+x, o[1]+y, o[2]) 

    def _shapely_to_pv_mesh(self, poly, o, n, x_dir):
        # Konvertiert Shapely Polygon (mit Löchern) via Earcut/Triangulation in pv.PolyData
        import shapely.ops
        try:
            tris = shapely.ops.triangulate(poly)
            # Filter triangles inside poly
            valid_tris = [t for t in tris if poly.contains(t.centroid)]
            
            points = []
            faces = []
            c = 0
            for t in valid_tris:
                xx, yy = t.exterior.coords.xy
                # Transform to 3D here...
                p1 = self._transform_2d_3d(xx[0], yy[0], o, n, x_dir)
                p2 = self._transform_2d_3d(xx[1], yy[1], o, n, x_dir)
                p3 = self._transform_2d_3d(xx[2], yy[2], o, n, x_dir)
                points.extend([p1, p2, p3])
                faces.extend([3, c, c+1, c+2])
                c += 3
            
            return pv.PolyData(points, faces)
        except: return None