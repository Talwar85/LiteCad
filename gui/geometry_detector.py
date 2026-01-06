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
class DetectedEdge:
    id: int
    body_id: str
    edge_type: str # 'boundary' oder 'feature'
    points: List[Tuple[float, float, float]]
    vtk_cells: List[int] # Referenz zur Linie im Edge-Mesh

@dataclass
class SelectionFace:
    id: int
    owner_id: str                # sketch_id oder body_id
    domain_type: str             # 'sketch_shell', 'sketch_hole', 'body_face'

    shapely_poly: Any = None
    plane_origin: Tuple[float, float, float] = (0,0,0)
    plane_normal: Tuple[float, float, float] = (0,0,1)
    plane_x: Tuple[float, float, float] = (1,0,0)
    plane_y: Tuple[float, float, float] = (0,1,0)

    pick_priority: int = 0
    display_mesh: Any = None


class GeometryDetector:
    class SelectionFilter:
        # FIX: "sketch_profile" hinzufügen!
        ALL = {"sketch_shell", "sketch_hole", "body_face", "sketch_profile"}
        
        # Profile (die Ringe) gehören logisch zu den Skizzen
        SKETCH = {"sketch_shell", "sketch_hole", "sketch_profile"}
        
        # Löcher bleiben Löcher
        HOLE = {"sketch_hole"}
        
        # Faces sind Shells, Profile (Ringe) und Body-Flächen
        FACE = {"sketch_shell", "body_face", "sketch_profile"}
        
    def __init__(self):
        self.selection_faces: List[SelectionFace] = []
        self._counter = 0
        

    def clear(self):
        self.selection_faces.clear()
        self._counter = 0
        
        
    
    def process_sketch(self, sketch, plane_origin, plane_normal, plane_x_dir, plane_y_dir=None):
        """
        Analysiert Sketch-Linien und erkennt verschachtelte Regionen (Inseln).
        Löst das Problem: Kreis im Hexagon -> Erkennt Ring und Kern separat.
        """
        if not HAS_SHAPELY: return

        from shapely.geometry import LineString, Polygon
        from shapely.ops import unary_union, polygonize

        if plane_y_dir is None:
             # Fallback Berechnung
             n = np.array(plane_normal)
             x = np.array(plane_x_dir)
             y = np.cross(n, x)
             plane_y_dir = tuple(y)

        # 1. Linien in Shapely konvertieren
        lines = []
        def rnd(val): return round(val, 5)

        # Lines
        for l in getattr(sketch, 'lines', []):
            if not l.construction:
                lines.append(LineString([(rnd(l.start.x), rnd(l.start.y)), (rnd(l.end.x), rnd(l.end.y))]))
        
        # Circles / Arcs / Splines müssen hier auch rein (verkürzt dargestellt, nimm deinen existierenden Code für die Punkt-Generierung)
        # ... (Füge hier die Logik für Circles/Arcs aus deinem alten Modeling-Code ein, um `lines` zu füllen) ...
        # WICHTIG: Um den Code kurz zu halten, nutze ich hier Platzhalter. 
        # Du musst sicherstellen, dass ALLE Geometrie-Typen in `lines` landen.
        for c in getattr(sketch, 'circles', []):
             if not c.construction:
                 pts = [(c.center.x + c.radius * math.cos(i*2*math.pi/64), c.center.y + c.radius * math.sin(i*2*math.pi/64)) for i in range(65)]
                 lines.append(LineString(pts))

        if not lines: return

        # 2. Polygonize -> Findet alle geschlossenen Loops
        try:
            merged = unary_union(lines)
            raw_polys = list(polygonize(merged))
        except Exception:
            return

        # 3. Containment Analyse (Wer liegt in wem?)
        # Wir sortieren nach Fläche (groß zuerst), um Parents vor Children zu finden
        raw_polys.sort(key=lambda p: p.area, reverse=True)
        
        # Struktur: poly_info = { index: {'poly': p, 'children': [], 'parent': None} }
        poly_info = {i: {'poly': p, 'children': [], 'parent': None} for i, p in enumerate(raw_polys)}

        for i in range(len(raw_polys)):
            parent = raw_polys[i]
            for j in range(len(raw_polys)):
                if i == j: continue
                child = raw_polys[j]
                
                # Wenn Child im Parent liegt
                if parent.contains(child):
                    # Checken ob es ein direkter Parent ist (kein anderer Parent dazwischen)
                    # Wir weisen es erstmal zu, und verfeinern später oder nutzen einfache Logik:
                    # Da wir sortiert haben, ist der erste Container der größte.
                    # Bessere Logik: Das kleinste Polygon, das mich enthält, ist mein Parent.
                    pass

        # Einfachere, robuste Methode für "Hexagon mit Kreis":
        # Wir erzeugen SelectionFaces für die "Differenz".
        
        processed_indices = set()
        
        # A. Finde Polygone, die andere enthalten (Parents)
        for i, parent in enumerate(raw_polys):
            if i in processed_indices: continue
            
            # Suche direkte Löcher (Polygone die IN diesem Parent liegen)
            holes = []
            for j, child in enumerate(raw_polys):
                if i == j: continue
                # Ein echtes geometrisches Loch
                if parent.contains(child):
                    # Prüfen ob dieses Child nicht schon in einem Loch dieses Parents liegt (Nested Holes)
                    # Für einfache CAD-Fälle (Level 1 Nesting) reicht dies:
                    is_nested = False
                    for existing_hole in holes:
                        if existing_hole.contains(child):
                            is_nested = True; break
                    if not is_nested:
                        holes.append(child)

            # B. Erzeuge die "Ring"-Fläche (Parent minus alle direkten Holes)
            shell_poly = parent
            for h in holes:
                shell_poly = shell_poly.difference(h)
            
            # Erzeuge SelectionFace für den Ring (Hexagon ohne Kreis)
            self.selection_faces.append(
                self._create_selection_face(
                    owner_id=sketch.id,
                    domain_type="sketch_profile", # Neu: Profile statt Shell
                    poly=shell_poly,
                    plane_origin=plane_origin,
                    plane_normal=plane_normal,
                    plane_x=plane_x_dir,
                    plane_y=plane_y_dir,
                    priority=10
                )
            )
            
            # C. Erzeuge SelectionFaces für die Löcher selbst (damit man den Kreis wieder füllen kann)
            # Das ist wichtig: Das Loch ist jetzt eine eigenständige, klickbare Fläche
            for h in holes:
                self.selection_faces.append(
                    self._create_selection_face(
                        owner_id=sketch.id,
                        domain_type="sketch_profile",
                        poly=h,
                        plane_origin=plane_origin,
                        plane_normal=plane_normal,
                        plane_x=plane_x_dir,
                        plane_y=plane_y_dir,
                        priority=20 # Höhere Prio für innere Teile
                    )
                )
                
            # Alle verarbeiteten markieren (eigentlich müssten wir holes markieren, aber 
            # durch die Logik oben werden Löcher auch als Parents geprüft. 
            # Da sie niemanden enthalten, fallen sie durch und werden im nächsten Loop als "Solid" erzeugt.
            # Um Duplikate zu vermeiden, müssen wir aufpassen.
            # FIX: Wenn wir Holes als Faces erzeugen, dürfen sie im Hauptloop nicht nochmal als Parent erzeugt werden?
            # Doch, wenn das Loch selbst wieder etwas enthält. 
            # Aber Shapely 'polygonize' liefert atomare Teile.
            # KORREKTUR: polygonize liefert NICHT atomare Teile bei Verschachtelung, es liefert "Filled" Polygons.
            
            # Daher: Wir müssen `holes` aus dem Hauptloop entfernen?
            # Nein, die einfache Logik ist: 
            # Wenn ein Polygon im Detector angelegt wird, ist es klickbar.
            # Wir haben oben Shell und Holes angelegt. 
            # Die Holes sind ja auch in `raw_polys`. Wenn der Loop bei einem Hole ankommt (als `parent`), 
            # findet er keine Kinder. Er würde das Hole nochmal als Shell anlegen. 
            # Das ist OK, solange wir Duplikate vermeiden.
            
            # Optimierung: Wir merken uns, welche `raw_polys` als Holes verwendet wurden
            # und überspringen diese im Hauptloop NICHT, sondern lassen sie durchlaufen,
            # ABER wir müssen verhindern, dass wir die Fläche doppelt addieren.
            
            # Da `shell_poly` (Ring) geometrisch neu ist, ist es ok.
            # Das `hole` (Kreis) ist identisch mit `raw_polys[j]`.
            
        # Um es simpel und stabil zu halten (Dein Wunsch):
        # Wir löschen die Faces vorher (in clear) und bauen sie hier auf.
        # Wir nutzen eine Hilfsliste, um Überlappung zu vermeiden.
        
        # RESET: Neue Logik, ganz sauber.
        # 1. Wir berechnen für jedes Polygon seinen "Level" der Verschachtelung.
        #    Level 0: Außen. Level 1: Im Außen (Loch). Level 2: Im Loch (Insel).
        # 2. Wir erzeugen Faces immer als (Poly - Direkte Kinder).
        
        pass # Der Code oben war "Denkprozess". Hier ist die Implementierung:
        
        # --- IMPLEMENTIERUNG ---
        self.selection_faces = [f for f in self.selection_faces if f.owner_id != sketch.id] # Alte löschen

        # Hierarchy Building
        # hierarchy[i] = [list of indices that are strictly inside i]
        hierarchy = {i: [] for i in range(len(raw_polys))}
        for i, p_out in enumerate(raw_polys):
            for j, p_in in enumerate(raw_polys):
                if i == j: continue
                if p_out.contains(p_in):
                    hierarchy[i].append(j)
        
        # Bestimme direkte Kinder (Direct Children)
        # Ein Kind K ist direktes Kind von P, wenn es keinen Zwischen-Parent Z gibt, der in P liegt und K enthält.
        direct_children = {i: [] for i in range(len(raw_polys))}
        
        for parent_idx, all_children in hierarchy.items():
            for child_idx in all_children:
                is_direct = True
                for other_child in all_children:
                    if child_idx == other_child: continue
                    if raw_polys[other_child].contains(raw_polys[child_idx]):
                        is_direct = False
                        break
                if is_direct:
                    direct_children[parent_idx].append(child_idx)

        # Nun erzeugen wir die "Ringe" (Profile)
        # Ein Profil ist immer: Polygon minus seine direkten Kinder.
        processed_indices = set()
        
        for i in range(len(raw_polys)):
            poly = raw_polys[i]
            children_indices = direct_children[i]
            
            # Subtrahiere alle direkten Kinder
            display_poly = poly
            for child_idx in children_indices:
                display_poly = display_poly.difference(raw_polys[child_idx])
            
            if not display_poly.is_empty and display_poly.area > 0.0001:
                self.selection_faces.append(
                    self._create_selection_face(
                        owner_id=sketch.id,
                        domain_type="sketch_profile",
                        poly=display_poly, # Das ist der Ring (oder der Kreis, wenn er keine Kinder hat)
                        plane_origin=plane_origin,
                        plane_normal=plane_normal,
                        plane_x=plane_x_dir,
                        plane_y=plane_y_dir,
                        priority=10 + len(children_indices) # Ringe bevorzugen
                    )
                )

    def _create_selection_face(self, owner_id, domain_type, poly, plane_origin, plane_normal, plane_x, plane_y, priority):
        # FIX: Hier fehlte die Weitergabe von plane_y an _shapely_to_pv_mesh
        display_mesh = self._shapely_to_pv_mesh(
            poly, 
            plane_origin, 
            plane_normal, 
            plane_x, 
            plane_y # <--- Neu übergeben
        )

        face = SelectionFace(
            id=self._counter,
            owner_id=owner_id,
            domain_type=domain_type,
            shapely_poly=poly,
            plane_origin=plane_origin,
            plane_normal=plane_normal,
            plane_x=plane_x,
            plane_y=plane_y,
            pick_priority=priority,
            display_mesh=display_mesh
        )
        self._counter += 1
        return face
    
    def pick(self, ray_origin, ray_dir, selection_filter=SelectionFilter.ALL):
        hits = []
        print(f"Pick Ray: {ray_origin} -> {ray_dir}")
        print(f"Active Filter: {selection_filter}")
        for face in self.selection_faces:
            if face.domain_type not in selection_filter:
                print(f"Ignoriere Face {face.id} ({face.domain_type}) wegen Filter")
                continue

            # Sketch-Faces → analytisch
            if face.domain_type.startswith("sketch"):
                hit = self._intersect_ray_plane(
                    ray_origin, ray_dir,
                    face.plane_origin,
                    face.plane_normal
                )
                if hit is None:
                    continue

                x, y = self._project_point_2d(
                    hit,
                    face.plane_origin,
                    face.plane_x,
                    face.plane_y
                )

                if face.shapely_poly.contains(Point(x, y)):
                    dist = np.linalg.norm(np.array(hit) - np.array(ray_origin))
                    hits.append((face.pick_priority, dist, face.id))

            # Body-Faces → Mesh-Ray
            elif face.domain_type == "body_face":
                mesh = face.display_mesh
                pts, _ = mesh.ray_trace(
                    ray_origin,
                    np.array(ray_origin) + np.array(ray_dir) * 10000
                )
                if len(pts) > 0:
                    dist = np.linalg.norm(pts[0] - ray_origin)
                    hits.append((face.pick_priority, dist, face.id))

        if not hits:
            return -1

        hits.sort(key=lambda h: (-h[0], h[1]))
        return hits[0][2]


    def _calculate_plane_axes(self, plane_normal):
        """
        Erzeugt ein stabiles, rechtshändiges Koordinatensystem
        (x_dir, y_dir) für eine Ebene mit gegebener Normalenrichtung.
        """

        n = np.array(plane_normal, dtype=float)
        n /= np.linalg.norm(n)

        # Wähle eine Hilfsachse, die nicht parallel zur Normalen ist
        if abs(n[2]) < 0.9:
            helper = np.array([0, 0, 1], dtype=float)
        else:
            helper = np.array([1, 0, 0], dtype=float)

        x_dir = np.cross(helper, n)
        x_len = np.linalg.norm(x_dir)
        if x_len < 1e-6:
            # Fallback (extremer Sonderfall)
            helper = np.array([0, 1, 0], dtype=float)
            x_dir = np.cross(helper, n)
            x_len = np.linalg.norm(x_dir)

        x_dir /= x_len
        y_dir = np.cross(n, x_dir)

        return tuple(x_dir), tuple(y_dir)

        
    def _intersect_ray_plane(self, ray_o, ray_d, plane_o, plane_n):
        ray_o = np.array(ray_o)
        ray_d = np.array(ray_d)
        plane_o = np.array(plane_o)
        plane_n = np.array(plane_n)

        denom = np.dot(plane_n, ray_d)
        if abs(denom) < 1e-6:
            return None

        t = np.dot(plane_o - ray_o, plane_n) / denom
        if t < 0:
            return None

        return ray_o + ray_d * t


    def _project_point_2d(self, pt, origin, x_dir, y_dir):
        v = np.array(pt) - np.array(origin)
        return np.dot(v, x_dir), np.dot(v, y_dir)
    
    
    def process_body_mesh(self, body_id, vtk_mesh):
        if not HAS_VTK or vtk_mesh is None:
            return

        if 'Normals' not in vtk_mesh.cell_data:
            vtk_mesh.compute_normals(cell_normals=True, inplace=True)

        if not vtk_mesh.is_all_triangles:
            vtk_mesh = vtk_mesh.triangulate()

        normals = np.round(vtk_mesh.cell_data['Normals'], 3)
        unique_normals, groups = np.unique(normals, axis=0, return_inverse=True)

        for group_idx, normal in enumerate(unique_normals):
            cell_ids = np.where(groups == group_idx)[0]
            
            if len(cell_ids) < 1: 
                continue

            # Das hier ist ein UnstructuredGrid
            group_mesh_ugrid = vtk_mesh.extract_cells(cell_ids)
            
            # WICHTIG: Sofort in PolyData (Surface) wandeln für ray_trace Support!
            group_mesh = group_mesh_ugrid.extract_surface()

            try:
                # Connectivity Check
                conn = group_mesh.connectivity(extraction_mode='all')
                n_regions = conn.n_regions
                
                if n_regions < 1:
                    # Fallback: Alles nehmen
                    if group_mesh.n_points >= 3:
                        center = np.mean(group_mesh.points, axis=0)
                        self._add_body_face(body_id, center, normal, group_mesh)
                    continue

                for i in range(n_regions):
                    region = conn.threshold([i, i], scalars='RegionId')
                    
                    # Auch hier sicherstellen, dass wir Surface haben
                    region_surf = region.extract_surface()
                    
                    if region_surf.n_points < 3:
                        continue
                    
                    center = np.mean(region_surf.points, axis=0)
                    self._add_body_face(body_id, center, normal, region_surf)

            except Exception as e:
                print(f"Connectivity Warning: {e}")
                # Fallback bei Fehler: Das bereits konvertierte group_mesh nutzen
                if group_mesh.n_points >= 3:
                    center = np.mean(group_mesh.points, axis=0)
                    self._add_body_face(body_id, center, normal, group_mesh)

    def _add_body_face(self, body_id, center, normal, mesh):
        face = SelectionFace(
            id=self._counter,
            owner_id=body_id,
            domain_type="body_face",
            plane_origin=tuple(center),
            plane_normal=tuple(normal),
            pick_priority=5,
            display_mesh=mesh 
        )
        self.selection_faces.append(face)
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

    

    # --- Helper ---
    def _transform_2d_3d(self, x, y, origin, x_dir, y_dir):
        """
        Wandelt 2D (x,y) der Skizze in globale 3D Koordinaten um.
        FIX: Nutzt direkt die Basisvektoren ohne Neuberechnung.
        """
        ox, oy, oz = origin
        ux, uy, uz = x_dir
        vx, vy, vz = y_dir
        
        # P_global = Origin + x * BasisX + y * BasisY
        px = ox + x * ux + y * vx
        py = oy + x * uy + y * vy
        pz = oz + x * uz + y * vz
        
        return (px, py, pz)

    def _shapely_to_pv_mesh(self, poly, o, n, x_dir, plane_y_dir=None): # <--- plane_y_dir hinzugefügt
        """
        Erstellt ein PyVista Mesh aus einem Shapely Polygon.
        FIX: Nutzt explizit übergebene X/Y Vektoren für exakte Deckungsgleichheit mit der Skizze.
        """
        if not HAS_VTK: return None
        
        # Vektoren vorbereiten
        import numpy as np
        
        # Wenn Y nicht übergeben wurde, berechnen (Fallback)
        if plane_y_dir is None:
             n_vec = np.array(n)
             x_vec = np.array(x_dir)
             y_vec = np.cross(n_vec, x_vec)
        else:
             x_vec = np.array(x_dir)
             y_vec = np.array(plane_y_dir) # <--- WICHTIG: Das übergebene Y nutzen!

        import shapely.ops
        try:
            # Triangulierung im 2D Raum
            tris = shapely.ops.triangulate(poly)
            # Nur Dreiecke behalten, die wirklich im Polygon liegen
            valid_tris = [t for t in tris if poly.contains(t.centroid)]
            
            if not valid_tris:
                return None
            
            points = []
            faces = []
            c = 0
            
            for t in valid_tris:
                # 2D Koordinaten des Dreiecks holen
                xx, yy = t.exterior.coords.xy
                
                # Die ersten 3 Punkte des Dreiecks transformieren
                # WICHTIG: Wir übergeben hier x_vec und y_vec explizit an _transform_2d_3d
                p1 = self._transform_2d_3d(xx[0], yy[0], o, x_vec, y_vec)
                p2 = self._transform_2d_3d(xx[1], yy[1], o, x_vec, y_vec)
                p3 = self._transform_2d_3d(xx[2], yy[2], o, x_vec, y_vec)
                
                points.extend([p1, p2, p3])
                # PyVista Face Format: [AnzahlPunkte, id1, id2, id3]
                faces.extend([3, c, c+1, c+2])
                c += 3
            
            # Mesh erstellen
            mesh = pv.PolyData(points, faces)
            return mesh
        except Exception as e:
            print(f"Mesh generation error: {e}")
            return None