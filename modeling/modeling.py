"""
LiteCAD - 3D Modeling
Robust B-Rep Implementation with Build123d
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from enum import Enum, auto
import math
import uuid
import sys
import os

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
        export_stl, export_step
    )
    HAS_BUILD123D = True
    print("✓ build123d geladen (Modeling).")
except ImportError as e:
    print(f"! build123d nicht gefunden: {e}")

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

@dataclass
class ExtrudeFeature(Feature):
    sketch: Sketch = None
    distance: float = 10.0
    direction: int = 1 
    operation: str = "New Body"
    # ÄNDERUNG: Wir erlauben eine Liste von Punkten (oder None für "alles")
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
    edges: List = field(default_factory=list)
    
    def __post_init__(self):
        self.type = FeatureType.FILLET
        if not self.name or self.name == "Feature": self.name = "Fillet"

@dataclass
class ChamferFeature(Feature):
    distance: float = 2.0
    edges: List = field(default_factory=list)
    
    def __post_init__(self):
        self.type = FeatureType.CHAMFER
        if not self.name or self.name == "Feature": self.name = "Chamfer"


# ==================== CORE LOGIC ====================

class Body:
    """
    3D-Körper (Body)
    Speichert Build123d Solid (BREP) und Mesh für Anzeige.
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
    
    def _rebuild(self):
        """
        Der Kern der Parametrik:
        Durchläuft alle Features und baut den Solid Schritt für Schritt neu auf.
        """
        # Reset
        self._build123d_solid = None
        self.shape = None
        self._mesh_vertices.clear()
        self._mesh_triangles.clear()
        
        current_solid = None
        
        for feature in self.features:
            if feature.suppressed: continue
            
            # --- EXTRUDE ---
            if isinstance(feature, ExtrudeFeature):
                new_part = self._compute_extrude_part(feature)
                
                if new_part:
                    if current_solid is None or feature.operation == "New Body":
                        current_solid = new_part
                    elif feature.operation == "Join":
                        current_solid = current_solid + new_part
                    elif feature.operation == "Cut":
                        current_solid = current_solid - new_part
                    elif feature.operation == "Intersect":
                        current_solid = current_solid & new_part
            
            # --- REVOLVE (Platzhalter) ---
            elif isinstance(feature, RevolveFeature):
                pass 
                
        # Ergebnis speichern
        if current_solid:
            self._build123d_solid = current_solid
            if hasattr(current_solid, 'wrapped'):
                self.shape = current_solid.wrapped 
            
            # Mesh erzeugen
            self._update_mesh_from_solid(current_solid)

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
                                        # Es war nur ein einzelner Punkt (tuple)
                                        selectors = [selectors]
                                    
                                    # Prüfe ob EINER der Selector-Punkte passt
                                    for sel_pt in selectors:
                                        # Buffer(0) repariert oft invalid Geometrie, contains ist strikt
                                        # Wir nutzen distance < epsilon für Robustheit an Kanten
                                        pt = Point(sel_pt)
                                        if poly.contains(pt) or poly.distance(pt) < 1e-4:
                                            should_create = True
                                            break
                                except Exception as e:
                                    # Fallback bei Fehler: Erstellen
                                    print(f"Selector check warning: {e}")
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
            print(f"Extrude calc failed: {e}")
            import traceback
            traceback.print_exc()
            return None

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
            # Cache Key generieren (optional, aber gut für Performance bei ocp_tessellate intern)
            cache_key = f"{id(shape)}" 
            deviation = 0.1
            
            result = tessellate(
                shape,
                cache_key,      # key
                deviation,            # deviation (linear tolerance)
                quality=0.1,    # quality (same as deviation usually)
                angular_tolerance=0.2,
                compute_faces=True,
                compute_edges=True,  # WICHTIG: Kanten berechnen lassen!
                debug=False
            )

            # --- 1. VERTICES & NORMALS ---
            # ocp_tessellate gibt oft flache Arrays zurück [x,y,z,x,y,z...]
            
            verts_flat = result["vertices"]
            norms_flat = result["normals"]
            
            # Vertices konvertieren
            if isinstance(verts_flat, np.ndarray):
                self._mesh_vertices = verts_flat.reshape(-1, 3).tolist()
            else:
                self._mesh_vertices = [tuple(verts_flat[i:i+3]) for i in range(0, len(verts_flat), 3)]

            # Normals konvertieren (identisch zu Vertices)
            if isinstance(norms_flat, np.ndarray):
                 self._mesh_normals = norms_flat.reshape(-1, 3).tolist()
            else:
                 self._mesh_normals = [tuple(norms_flat[i:i+3]) for i in range(0, len(norms_flat), 3)]

            # --- 2. TRIANGLES ---
            tris_flat = result["triangles"]
            if isinstance(tris_flat, np.ndarray):
                # reshape zu [(i1,i2,i3), ...]
                self._mesh_triangles = tris_flat.reshape(-1, 3).tolist()
            else:
                self._mesh_triangles = [tuple(tris_flat[i:i+3]) for i in range(0, len(tris_flat), 3)]

            # --- 3. EDGES ---
            # Edges kommen als Liste von Indizes-Paaren [start, end, start, end...]
            # Wir wollen sie als Liste von Tupeln [(i1, i2), (i3, i4), ...]
            edges_flat = result["edges"]
            if edges_flat is not None and len(edges_flat) > 0:
                if isinstance(edges_flat, np.ndarray):
                    self._mesh_edges = edges_flat.reshape(-1, 2).tolist()
                else:
                    self._mesh_edges = [tuple(edges_flat[i:i+2]) for i in range(0, len(edges_flat), 2)]
            
            return # Erfolg!
            
        except ImportError:
            pass 
        except Exception as e:
            print(f"Tessellation warning: {e}")
            import traceback
            traceback.print_exc()

        # --- OPTION B: Standard Build123d Fallback (Robust) ---
        try:
            # Liefert (Liste[Vector], Liste[List[int]])
            mesh = solid.tessellate(tolerance=0.05)
            
            self._mesh_vertices = [(v.X, v.Y, v.Z) for v in mesh[0]]
            
            self._mesh_triangles = []
            for face_indices in mesh[1]:
                if len(face_indices) == 3:
                    self._mesh_triangles.append(tuple(face_indices))
                elif len(face_indices) == 4:
                    # Quad -> 2 Tris
                    self._mesh_triangles.append((face_indices[0], face_indices[1], face_indices[2]))
                    self._mesh_triangles.append((face_indices[0], face_indices[2], face_indices[3]))
                else:
                    # N-Gon Fan
                    v0 = face_indices[0]
                    for i in range(1, len(face_indices) - 1):
                        self._mesh_triangles.append((v0, face_indices[i], face_indices[i+1]))
                        
        except Exception as e:
            print(f"CRITICAL MESHING ERROR: {e}")

    def export_stl(self, filename: str) -> bool:
        """STL Export"""
        if HAS_BUILD123D and self.shape is not None:
            try:
                export_stl(self._build123d_solid, filename)
                return True
            except Exception as e:
                print(f"Build123d STL export failed: {e}")
        
        if HAS_OCP and self.shape is not None:
            try:
                mesh = BRepMesh_IncrementalMesh(self.shape, 0.1)
                mesh.Perform()
                writer = StlAPI_Writer()
                writer.Write(self.shape, filename)
                return True
            except Exception as e:
                print(f"OCP STL export failed: {e}")
                
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