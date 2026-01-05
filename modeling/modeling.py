"""
LiteCAD - 3D Modeling
Robust B-Rep Implementation with Build123d & Smart Failure Recovery
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from enum import Enum, auto
import math
import uuid
import sys
import os
import traceback
from loguru import logger # WICHTIG: Loguru nutzen

# ==================== IMPORTS ====================
HAS_BUILD123D = False
HAS_OCP = False

try:
    from build123d import (
        Box, Cylinder, Sphere, Solid, Shape,
        extrude, revolve, fillet, chamfer,
        Axis, Plane, Location, Vector,
        BuildPart, BuildSketch, BuildLine,
        Part, Sketch as B123Sketch, 
        Rectangle as B123Rect, Circle as B123Circle,
        Polyline, Polygon, make_face, Mode,
        export_stl, export_step,
        GeomType
    )
    HAS_BUILD123D = True
    logger.success("✓ build123d geladen (Modeling).")
except ImportError as e:
    logger.error(f"! build123d nicht gefunden: {e}")

# Fallback OCP Imports
if not HAS_BUILD123D:
    try:
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common
        from OCP.StlAPI import StlAPI_Writer
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        HAS_OCP = True
    except Exception:
        pass

# Projektpfad
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from sketcher import Sketch

# ==================== DATENSTRUKTUREN ====================

class FeatureType(Enum):
    SKETCH = auto()
    EXTRUDE = auto()
    REVOLVE = auto()
    FILLET = auto()
    CHAMFER = auto()

@dataclass
class Feature:
    type: FeatureType = None
    name: str = "Feature"
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    visible: bool = True
    suppressed: bool = False
    status: str = "OK" # OK, ERROR, WARNING

@dataclass
class ExtrudeFeature(Feature):
    sketch: Sketch = None
    distance: float = 10.0
    direction: int = 1 
    operation: str = "New Body"
    selector: list = None 
    
    def __post_init__(self):
        self.type = FeatureType.EXTRUDE
        if not self.name or self.name == "Feature": self.name = "Extrude"

@dataclass
class RevolveFeature(Feature):
    sketch: Sketch = None
    angle: float = 360.0
    axis: Tuple[float, float, float] = (0, 1, 0)
    operation: str = "New Body"
    
    def __post_init__(self):
        self.type = FeatureType.REVOLVE
        if not self.name or self.name == "Feature": self.name = "Revolve"

@dataclass 
class FilletFeature(Feature):
    radius: float = 2.0
    # Wir speichern hier Indizes oder Point-Selectors, um Kanten wiederzufinden
    # Für den Anfang: Liste von Kanten-Selektoren (z.B. Center Points) oder None für "Alle"
    edge_selectors: List = None 
    
    def __post_init__(self):
        self.type = FeatureType.FILLET
        if not self.name or self.name == "Feature": self.name = "Fillet"

@dataclass
class ChamferFeature(Feature):
    distance: float = 2.0
    edge_selectors: List = None
    
    def __post_init__(self):
        self.type = FeatureType.CHAMFER
        if not self.name or self.name == "Feature": self.name = "Chamfer"


# ==================== CORE LOGIC ====================

class Body:
    """
    3D-Körper (Body) mit RobustPartBuilder Logik.
    """
    
    def __init__(self, name: str = "Body"):
        self.name = name
        self.id = str(uuid.uuid4())[:8]
        self.features: List[Feature] = []
        
        # CAD Kernel Objekte
        self._build123d_solid = None  
        self.shape = None             
        
        # Visualisierungs-Daten (Mesh)
        self._mesh_vertices: List[Tuple[float, float, float]] = []
        self._mesh_triangles: List[Tuple[int, int, int]] = []
        self._mesh_normals = [] 
        self._mesh_edges = []
        
    def add_feature(self, feature: Feature):
        """Feature hinzufügen und Geometrie neu berechnen"""
        self.features.append(feature)
        self._rebuild()
    
    def remove_feature(self, feature: Feature):
        if feature in self.features:
            self.features.remove(feature)
            self._rebuild()
    
    def _safe_operation(self, op_name, op_func, fallback_func=None):
        """
        Wrapper für kritische CAD-Operationen.
        Fängt Crashes ab und erlaubt Fallbacks.
        """
        try:
            result = op_func()
            
            # Validierung des Ergebnisses
            if result is None:
                raise ValueError("Operation returned None")
            
            # Bei Build123d Objekten checken ob valid
            if hasattr(result, 'is_valid') and not result.is_valid():
                raise ValueError("Result geometry is invalid")
                
            return result, "OK"
            
        except Exception as e:
            logger.warning(f"Feature '{op_name}' fehlgeschlagen: {e}")
            # traceback.print_exc() # Optional für Debugging
            
            if fallback_func:
                logger.info(f"→ Versuche Fallback für '{op_name}'...")
                try:
                    res_fallback = fallback_func()
                    if res_fallback:
                        logger.success(f"✓ Fallback für '{op_name}' erfolgreich.")
                        return res_fallback, "WARNING" # Status Warning, weil nicht original
                except Exception as e2:
                    logger.error(f"✗ Auch Fallback fehlgeschlagen: {e2}")
            
            return None, "ERROR"

    def _rebuild(self):
        """
        Robuster Rebuild-Prozess (History-basiert).
        """
        logger.info(f"Rebuilding Body '{self.name}' ({len(self.features)} Features)...")
        
        # Reset visual data
        self._mesh_vertices.clear()
        self._mesh_triangles.clear()
        
        # Startzustand: Leer oder Base Object
        current_solid = None
        
        for i, feature in enumerate(self.features):
            if feature.suppressed:
                feature.status = "SUPPRESSED"
                continue
            
            new_solid = None
            status = "OK"
            
            # ================= EXTRUDE =================
            if isinstance(feature, ExtrudeFeature):
                def op_extrude():
                    return self._compute_extrude_part(feature)
                
                # Extrude hat noch keinen Fallback (könnte man z.B. ohne Selector probieren)
                part_geometry, status = self._safe_operation(f"Extrude_{i}", op_extrude)
                
                if part_geometry:
                    if current_solid is None or feature.operation == "New Body":
                        new_solid = part_geometry
                    else:
                        # Boolean Operationen
                        try:
                            if feature.operation == "Join":
                                new_solid = current_solid + part_geometry
                            elif feature.operation == "Cut":
                                new_solid = current_solid - part_geometry
                            elif feature.operation == "Intersect":
                                new_solid = current_solid & part_geometry
                        except Exception as e:
                            logger.error(f"Boolean {feature.operation} failed: {e}")
                            status = "ERROR"

            # ================= FILLET =================
            elif isinstance(feature, FilletFeature):
                if current_solid:
                    # Closure für Retry-Logik
                    def op_fillet(rad=feature.radius):
                        # Kanten finden (Logic TBD: Wir nehmen für jetzt ALLE oder Selektierte)
                        edges_to_fillet = self._resolve_edges(current_solid, feature.edge_selectors)
                        if not edges_to_fillet: raise ValueError("No edges selected")
                        return fillet(edges_to_fillet, radius=rad)
                    
                    def fallback_fillet():
                        # Smart Retry: Versuche 99% (oft Fix für Tangential-Probleme) oder 50%
                        try:
                            return op_fillet(feature.radius * 0.99)
                        except:
                            return op_fillet(feature.radius * 0.5)

                    new_solid, status = self._safe_operation(f"Fillet_{i}", op_fillet, fallback_fillet)
                    
                    # Wenn Fillet fehlschlägt, behalten wir den alten Solid!
                    if new_solid is None:
                        new_solid = current_solid 
                        # Feature markieren wir trotzdem als Error
                        status = "ERROR" 

            # ================= CHAMFER =================
            elif isinstance(feature, ChamferFeature):
                if current_solid:
                    def op_chamfer(dist=feature.distance):
                        edges = self._resolve_edges(current_solid, feature.edge_selectors)
                        if not edges: raise ValueError("No edges")
                        return chamfer(edges, length=dist)
                    
                    def fallback_chamfer():
                        return op_chamfer(feature.distance * 0.5)

                    new_solid, status = self._safe_operation(f"Chamfer_{i}", op_chamfer, fallback_chamfer)
                    if new_solid is None: new_solid = current_solid; status = "ERROR"
            
            # --- STATUS UPDATE ---
            feature.status = status
            
            # Wenn Feature erfolgreich (oder Fallback), Update current_solid
            # Wenn Fehler bei Boolean/Extrude, bleibt current_solid beim Alten (History Preservation)
            if new_solid is not None:
                current_solid = new_solid
                
        # Ende der History Chain
        if current_solid:
            self._build123d_solid = current_solid
            if hasattr(current_solid, 'wrapped'):
                self.shape = current_solid.wrapped 
            
            # Mesh erzeugen (nur einmal am Ende)
            self._update_mesh_from_solid(current_solid)
            # B-Rep Statistik abrufen
            n_faces = len(current_solid.faces())
            n_edges = len(current_solid.edges())
            logger.success(f"✓ {self.name}: BREP Valid ({n_faces} Faces, {n_edges} Edges)")
        else:
            logger.warning(f"Body '{self.name}' is empty after rebuild.")

    def _resolve_edges(self, solid, selectors):
        """
        Versucht, Kanten basierend auf Selektoren im aktuellen Solid zu finden.
        Parametric Robustness: Dies ist der schwierigste Teil (Topological Naming).
        
        Einfache Strategie für jetzt:
        - Wenn selectors None: Alle Kanten (Vorsicht!)
        - Sonst: Selektoren sind Punkte (Centers) im Raum. Wir suchen die nächste Kante.
        """
        if not selectors:
            return solid.edges() # Fallback: Alles
        
        found_edges = []
        all_edges = solid.edges()
        
        for sel in selectors:
            # Annahme: sel ist ein Tupel/Vector (x,y,z)
            # Finde Kante mit geringstem Abstand zum Punkt
            best_edge = None
            min_dist = float('inf')
            
            try:
                # Wir konvertieren den Selector in einen Vector
                p_sel = Vector(sel)
                
                for edge in all_edges:
                    # Edge Center berechnen
                    try:
                        # Build123d Edge hat .center()
                        dist = (edge.center() - p_sel).length
                        if dist < min_dist:
                            min_dist = dist
                            best_edge = edge
                    except: pass
                
                # Toleranz: Wenn Kante sich drastisch verschoben hat (> 20mm), nicht nehmen
                if best_edge and min_dist < 20.0:
                    found_edges.append(best_edge)
                    
            except Exception:
                pass
                
        return found_edges

    def _compute_extrude_part(self, feature: ExtrudeFeature):
        """Berechnet die Geometrie für eine Extrusion mit robuster Profil-Filterung"""
        if not HAS_BUILD123D or not feature.sketch: return None
        
        try:
            sketch = feature.sketch
            plane = self._get_plane_from_sketch(sketch)
            
            # Wichtig: Point importieren
            from shapely.geometry import Point, Polygon as ShapelyPoly
            
            with BuildPart() as part:
                with BuildSketch(plane):
                    profiles = sketch.find_closed_profiles()
                    created_any = False
                    
                    for p in profiles:
                        # Punkte extrahieren
                        points = [(float(l.start.x), float(l.start.y)) for l in p]
                        
                        if len(points) >= 3:
                            should_create = False
                            
                            # LOGIK: 
                            # None -> Alles
                            # Liste/Tupel -> Filtern
                            if feature.selector is None:
                                should_create = True
                            else:
                                try:
                                    poly = ShapelyPoly(points)
                                    # Normalisiere selector zu einer Liste von Punkten
                                    selectors = feature.selector
                                    if isinstance(selectors, tuple) and len(selectors) == 2 and isinstance(selectors[0], (int, float)):
                                        selectors = [selectors]
                                    
                                    # Prüfe ob EINER der Selector-Punkte passt
                                    for sel_pt in selectors:
                                        # Buffer(0) repariert oft invalid Geometrie
                                        pt = Point(sel_pt)
                                        if poly.contains(pt) or poly.distance(pt) < 1e-4:
                                            should_create = True
                                            break
                                except Exception as e:
                                    logger.warning(f"Selector check warning: {e}")
                                    should_create = True
                            
                            if should_create:
                                Polygon(*points, align=None)
                                created_any = True
                    
                    # Kreise
                    for c in sketch.circles:
                        should_create = False
                        if feature.selector is None:
                            should_create = True
                        else:
                            selectors = feature.selector
                            if isinstance(selectors, tuple) and len(selectors) == 2 and isinstance(selectors[0], (int, float)):
                                selectors = [selectors]
                                
                            for sel_pt in selectors:
                                dist = math.sqrt((c.center.x - sel_pt[0])**2 + (c.center.y - sel_pt[1])**2)
                                if dist <= c.radius:
                                    should_create = True
                                    break
                        
                        if should_create:
                            with Location((float(c.center.x), float(c.center.y))):
                                B123Circle(radius=float(c.radius))
                            created_any = True
                
                if created_any:
                    extrude(amount=feature.distance * feature.direction)
            
            return part.part 
            
        except Exception as e:
            logger.error(f"Extrude calc failed: {e}")
            raise e # Weiterwerfen für _safe_operation

    def _get_plane_from_sketch(self, sketch):
        """Erstellt Build123d Plane aus Sketch-Daten"""
        origin = getattr(sketch, 'plane_origin', (0,0,0))
        normal = getattr(sketch, 'plane_normal', (0,0,1))
        x_dir = getattr(sketch, 'plane_x_dir', None)
        
        if x_dir:
            return Plane(origin=origin, x_dir=x_dir, z_dir=normal)
        return Plane(origin=origin, z_dir=normal)

    def _update_mesh_from_solid(self, solid):
        """Generiert Mesh-Daten für die GUI (High Performance)"""
        import numpy as np
        
        # Reset
        self._mesh_vertices = []
        self._mesh_triangles = []
        self._mesh_normals = []
        self._mesh_edges = []

        # --- OPTION A: High Performance OCP Tessellation ---
        try:
            from ocp_tessellate.tessellator import tessellate
            
            # Einstellungen für Qualität
            shape = solid.wrapped
            cache_key = f"{id(shape)}" 
            deviation = 0.1
            
            result = tessellate(
                shape, cache_key, deviation, quality=0.1,
                angular_tolerance=0.2, compute_faces=True, 
                compute_edges=True, debug=False
            )

            # 1. Vertices & Normals
            verts_flat = result["vertices"]
            norms_flat = result["normals"]
            
            if isinstance(verts_flat, np.ndarray):
                self._mesh_vertices = verts_flat.reshape(-1, 3).tolist()
            else:
                self._mesh_vertices = [tuple(verts_flat[i:i+3]) for i in range(0, len(verts_flat), 3)]

            if isinstance(norms_flat, np.ndarray):
                 self._mesh_normals = norms_flat.reshape(-1, 3).tolist()
            else:
                 self._mesh_normals = [tuple(norms_flat[i:i+3]) for i in range(0, len(norms_flat), 3)]

            # 2. Triangles
            tris_flat = result["triangles"]
            if isinstance(tris_flat, np.ndarray):
                self._mesh_triangles = tris_flat.reshape(-1, 3).tolist()
            else:
                self._mesh_triangles = [tuple(tris_flat[i:i+3]) for i in range(0, len(tris_flat), 3)]

            # 3. Edges
            edges_flat = result["edges"]
            if edges_flat is not None and len(edges_flat) > 0:
                # OCP liefert Koordinaten [x,y,z, x,y,z...], diese werden direkt gespeichert
                # Main Window viewport muss wissen, dass das Koordinaten sind, keine Indizes!
                if isinstance(edges_flat, np.ndarray):
                    # Wir speichern es als separate Variable für OCP Linien
                    self._mesh_edge_lines = edges_flat.reshape(-1, 3).tolist()
                else:
                    self._mesh_edge_lines = [tuple(edges_flat[i:i+3]) for i in range(0, len(edges_flat), 3)]
            
            return 
            
        except ImportError:
            pass 
        except Exception as e:
            logger.warning(f"Tessellation warning: {e}")

        # --- OPTION B: Standard Build123d Fallback ---
        try:
            mesh = solid.tessellate(tolerance=0.05)
            self._mesh_vertices = [(v.X, v.Y, v.Z) for v in mesh[0]]
            self._mesh_triangles = []
            for face_indices in mesh[1]:
                if len(face_indices) == 3:
                    self._mesh_triangles.append(tuple(face_indices))
                elif len(face_indices) == 4:
                    self._mesh_triangles.append((face_indices[0], face_indices[1], face_indices[2]))
                    self._mesh_triangles.append((face_indices[0], face_indices[2], face_indices[3]))
                        
        except Exception as e:
            logger.error(f"CRITICAL MESHING ERROR: {e}")

    def export_stl(self, filename: str) -> bool:
        """STL Export"""
        if HAS_BUILD123D and self.shape is not None:
            try:
                export_stl(self._build123d_solid, filename)
                return True
            except Exception as e:
                logger.error(f"Build123d STL export failed: {e}")
        
        if self._mesh_vertices and self._mesh_triangles:
            return self._export_stl_simple(filename)
            
        return False

    def _export_stl_simple(self, filename: str) -> bool:
        """Primitiver STL Export aus Mesh-Daten"""
        try:
            with open(filename, 'w') as f:
                f.write(f"solid {self.name}\n")
                for tri in self._mesh_triangles:
                    v0 = self._mesh_vertices[tri[0]]
                    v1 = self._mesh_vertices[tri[1]]
                    v2 = self._mesh_vertices[tri[2]]
                    f.write(f"  facet normal 0 0 1\n")
                    f.write(f"    outer loop\n")
                    f.write(f"      vertex {v0[0]} {v0[1]} {v0[2]}\n")
                    f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
                    f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
                    f.write(f"    endloop\n")
                    f.write(f"  endfacet\n")
                f.write(f"endsolid {self.name}\n")
            return True
        except:
            return False

class Document:
    def __init__(self, name="Doc"):
        self.bodies: List[Body] = []
        self.sketches: List[Sketch] = []
        self.name = name
        self.active_body: Optional[Body] = None
        self.active_sketch: Optional[Sketch] = None
    
    def new_body(self, name=None):
        b = Body(name or f"Body{len(self.bodies)+1}")
        self.bodies.append(b)
        self.active_body = b
        return b
        
    def new_sketch(self, name=None):
        s = Sketch(name or f"Sketch{len(self.sketches)+1}")
        self.sketches.append(s)
        self.active_sketch = s
        return s