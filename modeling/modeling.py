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
from loguru import logger 
try:
    from shapely.geometry import LineString, Polygon as ShapelyPoly, Point
    from shapely.ops import polygonize, unary_union
except ImportError:
    logger.warning("Shapely nicht gefunden. Komplexe Skizzen könnten fehlschlagen.")
    
    
# WICHTIG: Unser neuer Helper
from modeling.cad_tessellator import CADTessellator
from modeling.mesh_converter import MeshToBREPConverter # NEU


# ==================== IMPORTS ====================
HAS_BUILD123D = False
HAS_OCP = False

try:
    from build123d import (
        Box, Cylinder, Sphere, Solid, Shape,
        extrude, revolve, fillet, chamfer,
        Axis, Plane, Locations, Vector,
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
        
        # PyVista/VTK Objekte (Cache)
        self.vtk_mesh = None       # pv.PolyData (Faces)
        self.vtk_edges = None      # pv.PolyData (Edges)
        
        # Legacy Visualisierungs-Daten (Nur als Fallback)
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
            
    def convert_to_brep(self):
        """
        Versucht, die aktuellen Mesh-Daten in ein echtes CAD-BREP umzuwandeln.
        Dies ermöglicht Boolesche Operationen und präzise Schnitte.
        """
        if self._build123d_solid is not None:
            logger.info(f"Body '{self.name}' ist bereits BREP.")
            return True

        if self.vtk_mesh is None:
            logger.warning("Keine Mesh-Daten vorhanden.")
            return False

        converter = MeshToBREPConverter()
        
        # Wir zielen auf 2000-5000 Faces, damit OCP nicht abstürzt
        solid = converter.convert(self.vtk_mesh, target_faces=3000, smooth=True)

        if solid:
            self._build123d_solid = solid
            self.shape = solid.wrapped
            
            # Wichtig: Ein Feature hinzufügen, damit der Browser Bescheid weiß
            # Wir nutzen ein Dummy-Feature, damit Rebuilds funktionieren (oder wir löschen die Mesh-History)
            logger.success(f"Body '{self.name}' erfolgreich zu BREP konvertiert.")
            
            # Mesh neu berechnen (diesmal vom BREP abgeleitet für Konsistenz)
            self._update_mesh_from_solid(solid)
            return True
        else:
            return False
            
    def _safe_operation(self, op_name, op_func, fallback_func=None):
        """
        Wrapper für kritische CAD-Operationen.
        Fängt Crashes ab und erlaubt Fallbacks.
        """
        try:
            result = op_func()
            
            if result is None:
                raise ValueError("Operation returned None")
            
            if hasattr(result, 'is_valid') and not result.is_valid():
                raise ValueError("Result geometry is invalid")
                
            return result, "OK"
            
        except Exception as e:
            logger.warning(f"Feature '{op_name}' fehlgeschlagen: {e}")
            
            if fallback_func:
                logger.info(f"→ Versuche Fallback für '{op_name}'...")
                try:
                    res_fallback = fallback_func()
                    if res_fallback:
                        logger.success(f"✓ Fallback für '{op_name}' erfolgreich.")
                        return res_fallback, "WARNING"
                except Exception as e2:
                    logger.error(f"✗ Auch Fallback fehlgeschlagen: {e2}")
            
            return None, "ERROR"

    def _rebuild(self):
        """
        Robuster Rebuild-Prozess (History-basiert).
        """
        logger.info(f"Rebuilding Body '{self.name}' ({len(self.features)} Features)...")
        
        # Reset Cache
        self.vtk_mesh = None
        self.vtk_edges = None
        self._mesh_vertices.clear()
        self._mesh_triangles.clear()
        
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
                
                part_geometry, status = self._safe_operation(f"Extrude_{i}", op_extrude)
                
                if part_geometry:
                    if current_solid is None or feature.operation == "New Body":
                        new_solid = part_geometry
                    else:
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
                    def op_fillet(rad=feature.radius):
                        edges_to_fillet = self._resolve_edges(current_solid, feature.edge_selectors)
                        if not edges_to_fillet: raise ValueError("No edges selected")
                        return fillet(edges_to_fillet, radius=rad)
                    
                    def fallback_fillet():
                        try: return op_fillet(feature.radius * 0.99)
                        except: return op_fillet(feature.radius * 0.5)

                    new_solid, status = self._safe_operation(f"Fillet_{i}", op_fillet, fallback_fillet)
                    if new_solid is None:
                        new_solid = current_solid 
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
            
            feature.status = status
            
            if new_solid is not None:
                current_solid = new_solid
                
        if current_solid:
            self._build123d_solid = current_solid
            if hasattr(current_solid, 'wrapped'):
                self.shape = current_solid.wrapped 
            
            # UPDATE MESH via Helper
            self._update_mesh_from_solid(current_solid)
            
            n_faces = len(current_solid.faces())
            logger.success(f"✓ {self.name}: BREP Valid ({n_faces} Faces)")
        else:
            logger.warning(f"Body '{self.name}' is empty after rebuild.")

    def _resolve_edges(self, solid, selectors):
        """Versucht, Kanten basierend auf Selektoren zu finden (Topological Naming Fallback)."""
        if not selectors:
            return solid.edges() 
        
        found_edges = []
        all_edges = solid.edges()
        
        for sel in selectors:
            best_edge = None
            min_dist = float('inf')
            
            try:
                p_sel = Vector(sel)
                for edge in all_edges:
                    try:
                        dist = (edge.center() - p_sel).length
                        if dist < min_dist:
                            min_dist = dist
                            best_edge = edge
                    except: pass
                
                if best_edge and min_dist < 20.0:
                    found_edges.append(best_edge)
                    
            except Exception:
                pass
                
        return found_edges

    def _compute_extrude_part(self, feature: ExtrudeFeature):
        """
        Berechnet die Geometrie für eine Extrusion.
        FIX: 
        1. Area-Check (Loch muss kleiner sein als Outer).
        2. Konzentrisch-Check (fixt Loch im Hexagon).
        3. Robuster Extrude-Aufruf mit Keywords.
        """
        if not HAS_BUILD123D or not feature.sketch: return None
        
        try:
            from shapely.geometry import LineString, Point, Polygon as ShapelyPoly
            from shapely.ops import unary_union, polygonize
            from build123d import make_face, Vector, Wire, extrude, Compound, Shape
            
            logger.info(f"--- Starte Extrusion: {feature.name} ---")
            
            sketch = feature.sketch
            plane = self._get_plane_from_sketch(sketch)
            
            # --- 1. Segmente sammeln ---
            all_segments = []
            def rnd(val): return round(val, 5)
            
            for l in sketch.lines:
                if not l.construction:
                    all_segments.append(LineString([(rnd(l.start.x), rnd(l.start.y)), (rnd(l.end.x), rnd(l.end.y))]))
            for c in sketch.circles:
                if not c.construction:
                    pts = [(rnd(c.center.x + c.radius * math.cos(i * 2 * math.pi / 64)), 
                            rnd(c.center.y + c.radius * math.sin(i * 2 * math.pi / 64))) for i in range(65)]
                    all_segments.append(LineString(pts))
            for arc in sketch.arcs:
                 if not arc.construction:
                    pts = []
                    start, end = arc.start_angle, arc.end_angle
                    sweep = end - start
                    if sweep < 0.1: sweep += 360
                    steps = max(12, int(sweep / 5))
                    for i in range(steps + 1):
                        t = math.radians(start + sweep * i / steps)
                        x = arc.center.x + arc.radius * math.cos(t)
                        y = arc.center.y + arc.radius * math.sin(t)
                        pts.append((rnd(x), rnd(y)))
                    if len(pts) >= 2: all_segments.append(LineString(pts))
            for spline in getattr(sketch, 'splines', []):
                 if not getattr(spline, 'construction', False):
                     pts_raw = []
                     if hasattr(spline, 'get_curve_points'):
                         pts_raw = spline.get_curve_points(segments_per_span=16)
                     elif hasattr(spline, 'to_lines'):
                         lines = spline.to_lines(segments_per_span=16)
                         if lines:
                             pts_raw.append((lines[0].start.x, lines[0].start.y))
                             for ln in lines: pts_raw.append((ln.end.x, ln.end.y))
                     pts = [(rnd(p[0]), rnd(p[1])) for p in pts_raw]
                     if len(pts) >= 2: all_segments.append(LineString(pts))

            if not all_segments: return None

            # --- 2. Polygonize & Deduplizierung ---
            try:
                merged = unary_union(all_segments)
                raw_candidates = list(polygonize(merged))
                
                candidates = []
                for rc in raw_candidates:
                    clean_poly = rc.buffer(0) # Reparatur
                    is_dup = False
                    for existing in candidates:
                        if abs(clean_poly.area - existing.area) < 1e-4 and clean_poly.centroid.distance(existing.centroid) < 1e-4:
                            is_dup = True
                            break
                    if not is_dup:
                        candidates.append(clean_poly)
                
                logger.info(f"Kandidaten (Unique): {len(candidates)}")
            except Exception as e:
                logger.error(f"Polygonize failed: {e}")
                return None
            
            if not candidates: return None

            # --- 3. Selektion ---
            selected_indices = set()
            if feature.selector:
                selectors = feature.selector
                if isinstance(selectors, tuple) and len(selectors) == 2 and isinstance(selectors[0], (int, float)):
                    selectors = [selectors]
                
                for sel_pt in selectors:
                    pt = Point(sel_pt)
                    matches = []
                    for i, poly in enumerate(candidates):
                        if poly.contains(pt) or poly.distance(pt) < 1e-2:
                            matches.append(i)
                    if matches:
                        best = min(matches, key=lambda i: candidates[i].area)
                        selected_indices.add(best)
            else:
                selected_indices = set(range(len(candidates)))

            if not selected_indices: return None

            # --- 4. Faces bauen ---
            faces_to_extrude = []

            def to_3d_wire(shapely_poly):
                try:
                    pts_2d = list(shapely_poly.exterior.coords[:-1])
                    if len(pts_2d) < 3: return None
                    pts_3d = [plane.from_local_coords((p[0], p[1])) for p in pts_2d]
                    return Wire.make_polygon(pts_3d)
                except: return None

            for outer_idx in selected_indices:
                try:
                    outer_poly = candidates[outer_idx]
                    outer_wire = to_3d_wire(outer_poly)
                    if not outer_wire: continue
                    
                    main_face = make_face(outer_wire)
                    
                    # Löcher suchen
                    for i, potential_hole in enumerate(candidates):
                        if i == outer_idx: continue
                        
                        # WICHTIG 1: Ein Loch muss kleiner sein!
                        if potential_hole.area >= outer_poly.area * 0.99:
                            continue

                        # Check: Ist es drinnen?
                        is_inside = False
                        reason = ""
                        
                        try:
                            # A) Konzentrisch (Sehr starkes Indiz für Loch)
                            if outer_poly.centroid.distance(potential_hole.centroid) < 1e-3:
                                is_inside = True; reason = "Concentric"
                            
                            # B) Intersection
                            elif not is_inside:
                                intersect = outer_poly.intersection(potential_hole)
                                ratio = intersect.area / potential_hole.area if potential_hole.area > 0 else 0
                                if ratio > 0.9: 
                                    is_inside = True; reason = f"Overlap {ratio:.2f}"
                                
                            # C) Centroid Check
                            if not is_inside:
                                if outer_poly.contains(potential_hole.centroid):
                                    is_inside = True; reason = "Centroid"
                        except: pass

                        if is_inside:
                            # Nur schneiden, wenn nicht selbst ausgewählt
                            if i not in selected_indices:
                                logger.info(f"  -> Schneide Loch #{i} ({reason})")
                                hole_wire = to_3d_wire(potential_hole)
                                if hole_wire:
                                    try:
                                        hole_face = make_face(hole_wire)
                                        main_face = main_face - hole_face
                                    except Exception as e:
                                        logger.warning(f"Cut failed: {e}")

                    faces_to_extrude.append(main_face)
                except Exception as e:
                    logger.error(f"Face construction error: {e}")

            if not faces_to_extrude: return None

            # --- 5. Extrudieren ---
            solids = []
            amount = feature.distance * feature.direction
            direction_vec = plane.z_dir 
            
            for f in faces_to_extrude:
                try:
                    # Versuche globale Funktion mit Keywords
                    s = extrude(f, amount=amount, dir=direction_vec)
                    if s and s.is_valid():
                        solids.append(s)
                    else:
                        raise ValueError("Invalid result")
                except Exception:
                    # Fallback: Methoden-Aufruf mit expliziten Keywords
                    try:
                        # Manche Versionen verlangen 'amount' und 'dir' explizit
                        if hasattr(f, 'extrude'):
                             s = f.extrude(amount=amount, dir=direction_vec)
                             if s: solids.append(s)
                    except Exception as ex2:
                        logger.error(f"Extrude failed for face: {ex2}")

            if not solids: 
                logger.warning("Keine Solids erzeugt!")
                return None
            
            logger.success(f"Extrusion OK: {len(solids)} Solids erzeugt.")

            if len(solids) == 1:
                return solids[0]
            else:
                return Compound(children=solids)
            
        except Exception as e:
            logger.error(f"Extrude CRASH: {e}")
            raise e





        
    
    
    def _get_plane_from_sketch(self, sketch):
        origin = getattr(sketch, 'plane_origin', (0,0,0))
        normal = getattr(sketch, 'plane_normal', (0,0,1))
        x_dir = getattr(sketch, 'plane_x_dir', None)
        if x_dir:
            return Plane(origin=origin, x_dir=x_dir, z_dir=normal)
        return Plane(origin=origin, z_dir=normal)

    def _update_mesh_from_solid(self, solid):
        """
        Generiert Mesh-Daten via Tessellator.
        Ersetzt den alten komplexen Code durch einen einfachen Helper-Aufruf.
        """
        if not solid: return

        # 1. High-Performance Tessellierung mit Cache
        # Dies delegiert die Arbeit an cad_tessellator.py
        self.vtk_mesh, self.vtk_edges = CADTessellator.tessellate(solid)
        
        # 2. Legacy Support leeren (spart Speicher), da wir jetzt vtk_mesh nutzen.
        # Falls du reine Listen brauchst (z.B. für Debugging), müsstest du sie hier aus
        # self.vtk_mesh extrahieren, aber für den Viewport ist das nicht mehr nötig.
        self._mesh_vertices = []
        self._mesh_triangles = []

    def export_stl(self, filename: str) -> bool:
        """STL Export: Versucht Build123d, dann VTK, dann Legacy."""
        
        # 1. build123d Export (Analytisch sauber)
        if HAS_BUILD123D and self.shape is not None:
            try:
                export_stl(self._build123d_solid, filename)
                return True
            except Exception as e:
                logger.error(f"Build123d STL export failed, trying mesh fallback: {e}")
        
        # 2. VTK Mesh Export (Schnell und robust)
        if self.vtk_mesh is not None:
            try:
                self.vtk_mesh.save(filename)
                return True
            except Exception as e:
                logger.error(f"VTK STL export failed: {e}")

        # 3. Legacy Fallback
        if self._mesh_vertices and self._mesh_triangles:
            return self._export_stl_simple(filename)
            
        return False

    def _export_stl_simple(self, filename: str) -> bool:
        """Primitiver STL Export aus Mesh-Daten (Letzter Ausweg)"""
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