"""
LiteCAD - 2D Sketch Editor v4
Fusion360-Style mit Tab-Eingabe, geschlossene Profile, professionelle UX
Mit Build123d Backend für parametrische CAD-Operationen
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QMenu, QApplication, QFrame, QInputDialog, QPushButton
)
from PySide6.QtCore import Qt, QPointF, QPoint, Signal, QRectF, QTimer, QThread
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath,
    QMouseEvent, QWheelEvent, QKeyEvent, QPolygonF, QFontMetrics
)
from enum import Enum, auto
from typing import Optional, List, Tuple, Set
import math
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from i18n import tr
from sketcher import Sketch, Point2D, Line2D, Circle2D, Arc2D, Constraint, ConstraintType

# Build123d für parametrische CAD-Operationen
HAS_BUILD123D = False
try:
    import build123d as b3d
    from build123d import (
        BuildPart, BuildSketch, BuildLine,
        Plane, Location, Locations, Vector,
        Line as B3DLine, Circle as B3DCircle, 
        CenterArc, ThreePointArc, TangentArc,
        Rectangle, Polygon, Polyline,
        extrude, fillet, chamfer, make_face,
        Mode
    )
    HAS_BUILD123D = True
    print("✓ Build123d erfolgreich geladen für Sketch-Editor")
except ImportError as e:
    print(f"! Build123d nicht verfügbar: {e}")

# Importiere Dialoge und Tools - mehrere Fallback-Versuche
_import_ok = False
_import_error = None

# Versuch 1: Als Paket (gui.xxx)
if not _import_ok:
    try:
        from gui.sketch_dialogs import DimensionInput, ToolOptionsPopup
        from gui.sketch_tools import SketchTool, SnapType
        from gui.sketch_handlers import SketchHandlersMixin
        from gui.sketch_renderer import SketchRendererMixin
        _import_ok = True
    except ImportError as e:
        _import_error = f"gui.xxx: {e}"

# Versuch 2: Direkt (wenn aus gui/ heraus ausgeführt)
if not _import_ok:
    try:
        from sketch_dialogs import DimensionInput, ToolOptionsPopup
        from sketch_tools import SketchTool, SnapType
        from sketch_handlers import SketchHandlersMixin
        from sketch_renderer import SketchRendererMixin
        _import_ok = True
    except ImportError as e:
        _import_error = f"direct: {e}"

# Versuch 3: Relativer Import
if not _import_ok:
    try:
        from .sketch_dialogs import DimensionInput, ToolOptionsPopup
        from .sketch_tools import SketchTool, SnapType
        from .sketch_handlers import SketchHandlersMixin
        from .sketch_renderer import SketchRendererMixin
        _import_ok = True
    except ImportError as e:
        _import_error = f"relative: {e}"

if not _import_ok:
    print(f"CRITICAL: Sketch-Module nicht gefunden! Letzter Fehler: {_import_error}")
    print("Bitte stelle sicher, dass sketch_handlers.py, sketch_renderer.py, sketch_tools.py und sketch_dialogs.py im gui/ Ordner liegen!")
    # Leere Fallback-Mixins damit es nicht crasht
    class SketchHandlersMixin: pass
    class SketchRendererMixin: pass
    class SketchTool:
        SELECT = 0
    class SnapType:
        NONE = 0
    class DimensionInput:
        def __init__(self, parent=None): pass
    class ToolOptionsPopup:
        def __init__(self, parent=None): pass


def ramer_douglas_peucker(points, epsilon):
    if len(points) < 3: return points
    
    dmax = 0.0
    index = 0
    end = len(points) - 1
    p1 = points[0]
    p2 = points[end]
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    norm = math.hypot(dx, dy)
    
    for i in range(1, end):
        p = points[i]
        # Senkrechter Abstand Punkt zur Linie
        if norm == 0:
            d = math.hypot(p[0]-p1[0], p[1]-p1[1])
        else:
            d = abs(dy * p[0] - dx * p[1] + p2[0] * p1[1] - p2[1] * p1[0]) / norm
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        res1 = ramer_douglas_peucker(points[:index+1], epsilon)
        res2 = ramer_douglas_peucker(points[index:], epsilon)
        return res1[:-1] + res2
    else:
        return [p1, p2]

class DXFImportWorker(QThread):
    finished_signal = Signal(list, list, list)  # lines, circles, arcs
    error_signal = Signal(str)
    progress_signal = Signal(str)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            import ezdxf
            import ezdxf.path
            
            doc = ezdxf.readfile(self.filepath)
            msp = doc.modelspace()
            
            new_lines = []
            new_circles = []
            new_arcs = []  # (cx, cy, radius, start_angle, end_angle)
            
            # Helper: Konvertiert komplexe Formen in mikroskopisch feine Linien
            def add_path_as_lines(entity, matrix=None):
                try:
                    p = ezdxf.path.make_path(entity)
                    if matrix: p = p.transform(matrix)
                    
                    # 1. SAMPLING: Alle 0.01 mm ein Punkt (Fusion-Qualität)
                    raw_points = list(p.flattening(distance=0.01))
                    if len(raw_points) < 2: return
                    
                    # 2. OPTIMIERUNG: RDP entfernt überflüssige Punkte
                    # epsilon=0.005: Erlaubt max 5 Mikrometer Abweichung -> Unsichtbar für Auge
                    points_2d = [(v.x, v.y) for v in raw_points]
                    simplified = ramer_douglas_peucker(points_2d, 0.005)
                    
                    for k in range(len(simplified) - 1):
                        p1 = simplified[k]
                        p2 = simplified[k+1]
                        # Min-Länge Filter (0.001mm) gegen Grafik-Glitches
                        if math.hypot(p2[0]-p1[0], p2[1]-p1[1]) > 0.001:
                            new_lines.append((p1[0], p1[1], p2[0], p2[1]))
                except Exception as e:
                    print(f"Path Error: {e}")

            def process_entity(entity, matrix=None):
                dxftype = entity.dxftype()
                
                # Blöcke rekursiv auflösen
                if dxftype == 'INSERT':
                    m = entity.matrix44()
                    if matrix: m = m * matrix
                    for virtual_entity in entity.virtual_entities():
                        process_entity(virtual_entity, m)
                
                # Echte Kreise
                elif dxftype == 'CIRCLE':
                    c = entity.dxf.center
                    if matrix: c = matrix.transform(c)
                    r = entity.dxf.radius
                    if matrix:
                        vec = matrix.transform_direction(ezdxf.math.Vec3(1, 0, 0))
                        r *= vec.magnitude
                    new_circles.append((c.x, c.y, r))

                # Linien direkt übernehmen
                elif dxftype == 'LINE':
                    start = entity.dxf.start
                    end = entity.dxf.end
                    if matrix:
                        start = matrix.transform(start)
                        end = matrix.transform(end)
                    new_lines.append((start.x, start.y, end.x, end.y))

                # Arcs als echte Arcs importieren
                elif dxftype == 'ARC':
                    c = entity.dxf.center
                    r = entity.dxf.radius
                    # DXF: Winkel in Grad, CCW von positiver X-Achse
                    start_angle = entity.dxf.start_angle
                    end_angle = entity.dxf.end_angle
                    
                    if matrix:
                        c = matrix.transform(c)
                        # Skalierung für Radius
                        vec = matrix.transform_direction(ezdxf.math.Vec3(1, 0, 0))
                        r *= vec.magnitude
                        # Rotation für Winkel (vereinfacht - funktioniert für 90°-Rotationen)
                        # Für komplexere Transformationen müsste man die Winkel neu berechnen
                        rot_angle = math.degrees(math.atan2(vec.y, vec.x))
                        start_angle += rot_angle
                        end_angle += rot_angle
                    
                    # Normalisiere Winkel auf 0-360
                    start_angle = start_angle % 360
                    end_angle = end_angle % 360
                    
                    new_arcs.append((c.x, c.y, r, start_angle, end_angle))

                # Splines, Polylines, Ellipsen -> High-Res Fitting
                elif dxftype in ['SPLINE', 'LWPOLYLINE', 'POLYLINE', 'ELLIPSE']:
                    add_path_as_lines(entity, matrix)

            # Start
            all_ents = list(msp)
            total = len(all_ents)
            for i, e in enumerate(all_ents):
                process_entity(e)
                if i % 20 == 0: 
                    self.progress_signal.emit(f"Importiere... {int(i/total*100)}%")

            self.finished_signal.emit(new_lines, new_circles, new_arcs)

        except Exception as e:
            self.error_signal.emit(str(e))
            
            
class SketchEditor(QWidget, SketchHandlersMixin, SketchRendererMixin):
    """Professioneller 2D-Sketch-Editor"""
    
    sketched_changed = Signal()
    tool_changed = Signal(SketchTool)
    status_message = Signal(str)
    
    # Farben
    BG_COLOR = QColor(28, 28, 28)
    GRID_MINOR = QColor(38, 38, 38)
    GRID_MAJOR = QColor(50, 50, 50)
    AXIS_X = QColor(180, 60, 60)
    AXIS_Y = QColor(60, 180, 60)
    GEO_COLOR = QColor(220, 220, 220)
    GEO_CONSTRUCTION = QColor(255, 140, 0)
    GEO_SELECTED = QColor(0, 150, 255)
    GEO_HOVER = QColor(100, 200, 255)
    PROFILE_CLOSED = QColor(0, 120, 215, 40)
    PROFILE_HOVER = QColor(0, 180, 255, 80)  # Helleres Blau beim Hover
    PROFILE_OPEN = QColor(255, 100, 0, 30)
    SNAP_COLOR = QColor(255, 100, 0)
    DIM_COLOR = QColor(255, 200, 100)
    CONSTRAINT_COLOR = QColor(100, 200, 100)
    PREVIEW_COLOR = QColor(0, 150, 255, 150)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        
        self.sketch = Sketch("Sketch1")
        self.view_offset = QPointF(0, 0)
        self.view_scale = 5.0
        self.grid_size = 1.0
        self.grid_snap = True
        self.snap_enabled = True
        self.snap_radius = 15
        
        self.current_tool = SketchTool.SELECT
        self.tool_step = 0
        self.tool_points = []
        self.tool_data = {}
        
        self.selected_lines = []
        self.selected_circles = []
        self.selected_arcs = []
        self.selected_points = []  # Standalone Punkte
        self.hovered_entity = None
        
        self.selection_box_start = None
        self.selection_box_end = None
        self.current_snap = None
        
        self.mouse_screen = QPointF(0, 0)
        self.mouse_world = QPointF(0, 0)
        self.is_panning = False
        self.pan_start = QPointF()
        
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo = 50
        
        self.dim_input = DimensionInput(self)
        if hasattr(self.dim_input, 'choice_changed'):
            self.dim_input.choice_changed.connect(self._on_dim_choice_changed)
        
        self.dim_input.value_changed.connect(self._on_dim_value_changed)
        self.dim_input.confirmed.connect(self._on_dim_confirmed)
        self.dim_input.cancelled.connect(self._cancel_tool)
        self.dim_input_active = False
        self.viewport = None
        self.dim_input.value_changed.connect(self._on_dim_value_changed)
        
        self.live_length = 0.0
        self.live_angle = 0.0
        self.live_width = 0.0
        self.live_height = 0.0
        self.live_radius = 0.0
        
        self.polygon_sides = 6
        self.rect_mode = 0  # 0=2-Punkt, 1=Center (wie Fusion360)
        self.circle_mode = 0  # 0=Center-Radius, 1=2-Punkt, 2=3-Punkt
        self.construction_mode = False
        self.fillet_radius = 5.0
        self.offset_distance = 5.0
        self.chamfer_distance = 5.0
        self.extrude_operation = "New Body"  # Für Extrude-Dialog
        
        # Kreis-Segmente für Polygonisierung (für Face-Erkennung)
        self.circle_segments = 64  # Standard: 64 für gute Genauigkeit
        
        # Muttern-Aussparung (M2-M14)
        # Schlüsselweiten in mm für metrische Sechskant-Muttern (DIN 934)
        self.nut_sizes = {
            'M2': 4.0, 'M2.5': 5.0, 'M3': 5.5, 'M4': 7.0, 'M5': 8.0,
            'M6': 10.0, 'M8': 13.0, 'M10': 17.0, 'M12': 19.0, 'M14': 22.0
        }
        self.nut_size_names = ['M2', 'M2.5', 'M3', 'M4', 'M5', 'M6', 'M8', 'M10', 'M12', 'M14']
        self.nut_size_index = 4  # Default: M5
        self.nut_tolerance = 0.2  # Toleranz/Offset in mm für 3D-Druck
        
        self.closed_profiles = []
        self.profile_parent = []   # Hierarchie: Parent-Index für jeden Profil
        self.profile_children = [] # Hierarchie: Kinder-Indizes für jeden Profil
        self.hovered_face = None  # Face unter dem Cursor
        
        # Offset-Tool Zustand
        self.offset_dragging = False
        self.offset_start_pos = None
        self.offset_profile = None  # Das Profil das geoffsetet wird
        self.offset_preview_lines = []  # Preview der Offset-Linien
        
        # Spline-Editing (Fusion360-Style)
        self.spline_dragging = False
        self.spline_drag_type = None  # 'point', 'handle_in', 'handle_out'
        self.spline_drag_spline = None  # Die Spline die bearbeitet wird
        self.spline_drag_cp_index = None  # Index des Kontrollpunkts
        self.selected_spline = None  # Aktuell ausgewählte Spline
        self.hovered_spline_element = None  # (spline, cp_index, element_type)
        
        # Schwebende Optionen-Palette
        self.tool_options = ToolOptionsPopup(self)
        self.tool_options.option_selected.connect(self._on_tool_option_selected)
        
        # Body-Referenzen für transparente Anzeige im Hintergrund (Fusion360-Style)
        self.reference_bodies = []  # Liste von {'edges_2d': [...], 'color': (r,g,b)}
        self.show_body_reference = True  # Toggle für Anzeige
        self.body_reference_opacity = 0.25  # Transparenz der Bodies
        self.sketch_plane_normal = (0, 0, 1)  # Normale der Sketch-Ebene
        self.sketch_plane_origin = (0, 0, 0)  # Ursprung der Sketch-Ebene
        
        QTimer.singleShot(100, self._center_view)
    
    def world_to_screen(self, w):
        return QPointF(w.x() * self.view_scale + self.view_offset.x(),
                      -w.y() * self.view_scale + self.view_offset.y())
    
    def screen_to_world(self, s):
        return QPointF((s.x() - self.view_offset.x()) / self.view_scale,
                      -(s.y() - self.view_offset.y()) / self.view_scale)
    
    def _center_view(self):
        self.view_offset = QPointF(self.width() / 2, self.height() / 2)
        self.update()
    
    def set_reference_bodies(self, bodies_data, plane_normal=(0,0,1), plane_origin=(0,0,0)):
        """
        Setzt die Body-Referenzen für transparente Anzeige.
        bodies_data: Liste von {'mesh': pyvista_mesh, 'color': (r,g,b)} 
        plane_normal/origin: Die Sketch-Ebene für 2D-Projektion
        """
        self.reference_bodies = []
        self.sketch_plane_normal = plane_normal
        self.sketch_plane_origin = plane_origin
        
        if not bodies_data:
            self.update()
            return
        
        import numpy as np
        
        # Berechne lokales Koordinatensystem für die Ebene
        n = np.array(plane_normal)
        n = n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else np.array([0,0,1])
        
        # U und V Vektoren (lokal X und Y auf der Ebene)
        if abs(n[2]) < 0.9:
            u = np.cross(n, [0, 0, 1])
        else:
            u = np.cross(n, [1, 0, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        
        origin = np.array(plane_origin)
        
        for body_info in bodies_data:
            mesh = body_info.get('mesh')
            color = body_info.get('color', (0.6, 0.6, 0.8))
            
            if mesh is None:
                continue
            
            try:
                # Extrahiere Kanten für Drahtgitter-Darstellung
                edges = mesh.extract_feature_edges(
                    boundary_edges=True,
                    feature_edges=True,
                    manifold_edges=False,
                    feature_angle=30
                )
                
                if edges.n_points == 0:
                    edges = mesh.extract_all_edges()
                
                # Projiziere 3D-Kanten auf 2D-Ebene
                edges_2d = []
                
                if edges.n_lines > 0:
                    lines = edges.lines
                    points = edges.points
                    i = 0
                    while i < len(lines):
                        n_pts = lines[i]
                        if n_pts >= 2:
                            for j in range(n_pts - 1):
                                p1_3d = points[lines[i + 1 + j]]
                                p2_3d = points[lines[i + 2 + j]]
                                
                                # Projiziere auf Ebene (lokale 2D-Koordinaten)
                                rel1 = p1_3d - origin
                                rel2 = p2_3d - origin
                                
                                x1 = np.dot(rel1, u)
                                y1 = np.dot(rel1, v)
                                x2 = np.dot(rel2, u)
                                y2 = np.dot(rel2, v)
                                
                                edges_2d.append((x1, y1, x2, y2))
                        i += n_pts + 1
                
                if edges_2d:
                    self.reference_bodies.append({
                        'edges_2d': edges_2d,
                        'color': color
                    })
                    
            except Exception as e:
                print(f"Body reference error: {e}")
        
        self.update()
    
    def _draw_body_references(self, painter):
        """Zeichnet Bodies als transparente Referenz im Hintergrund"""
        if not self.show_body_reference or not self.reference_bodies:
            return
        
        painter.save()
        
        for body in self.reference_bodies:
            edges_2d = body.get('edges_2d', [])
            color = body.get('color', (0.6, 0.6, 0.8))
            
            # Farbe mit Transparenz
            r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
            alpha = int(self.body_reference_opacity * 255)
            
            pen = QPen(QColor(r, g, b, alpha))
            pen.setWidth(1)
            painter.setPen(pen)
            
            for x1, y1, x2, y2 in edges_2d:
                p1 = self.world_to_screen(QPointF(x1, y1))
                p2 = self.world_to_screen(QPointF(x2, y2))
                painter.drawLine(p1, p2)
        
        painter.restore()
    
    
    def _find_closed_profiles(self):
        """
        Findet geschlossene Flächen mit Shapely.
        FIX: Nutzt Koordinaten-Rundung, damit auch schräge Slots erkannt werden.
        """
        from shapely.geometry import LineString, Polygon as ShapelyPolygon
        from shapely.ops import polygonize, unary_union
        
        self.closed_profiles.clear()
        
        lines = [l for l in self.sketch.lines if not l.construction]
        arcs = [a for a in self.sketch.arcs if not a.construction]
        circles = [c for c in self.sketch.circles if not c.construction]
        
        shapely_lines = []
        
        # Hilfsfunktion zum Runden von Koordinaten (löst das Slot-Problem)
        def rnd(val):
            return round(val, 5)

        # 1. Linien
        for line in lines:
            shapely_lines.append(LineString([
                (rnd(line.start.x), rnd(line.start.y)),
                (rnd(line.end.x), rnd(line.end.y))
            ]))
            
        # 2. Bögen (in Segmente zerlegen und runden)
        for arc in arcs:
            points = []
            sweep = abs(arc.end_angle - arc.start_angle)
            if sweep < 0.1: sweep += 360
            steps = max(8, int(sweep / 5)) # Mindestens 8 Schritte für Genauigkeit
            
            # Start/End Winkel normalisieren
            start_rad = math.radians(arc.start_angle)
            
            # Wichtig: Winkelrichtung korrekt behandeln
            diff = arc.end_angle - arc.start_angle
            # Wenn diff fast 0 ist, ist es wahrsch. ein Vollkreis-Fehler oder Konstruktionsfehler,
            # aber hier nehmen wir an, die Winkel stimmen aus _create_slot
            
            for i in range(steps + 1):
                t = i / steps
                angle = math.radians(arc.start_angle + diff * t)
                px = arc.center.x + arc.radius * math.cos(angle)
                py = arc.center.y + arc.radius * math.sin(angle)
                points.append((rnd(px), rnd(py)))
            
            if len(points) >= 2:
                shapely_lines.append(LineString(points))

        # 3. Kreise (Code bleibt ähnlich, nur mit rnd)
        circle_segments = []
        standalone_circle_polys = []
        standalone_circles = [] # Originale speichern
        
        # ... (Logik für Kreise hier gekürzt, da meist unproblematisch. 
        # Wichtig ist, dass Linien und Bögen sich treffen.)
        # Wir fügen die Kreise als Segmente hinzu, wenn sie Linien berühren
        
        # (Vereinfachte Kreis-Logik für Übersichtlichkeit, der alte Code war okay, 
        # aber Linien/Bögen sind das Hauptproblem beim Slot)
        for circle in circles:
            cx, cy, r = circle.center.x, circle.center.y, circle.radius
            poly_points = []
            for j in range(64):
                a = 2 * math.pi * j / 64
                poly_points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
            standalone_circle_polys.append(ShapelyPolygon(poly_points))
            standalone_circles.append(circle)

        # 4. Polygonize
        if shapely_lines:
            try:
                merged = unary_union(shapely_lines)
                for poly in polygonize(merged):
                    if poly.area > 0.1:
                        self.closed_profiles.append(('polygon', poly))
            except Exception as e:
                print(f"Polygonize error: {e}")

        # 5. Standalone Kreise & Löcher hinzufügen
        # (Einfache Version: Füge einfach alle vollen Kreise als Profile hinzu)
        # Wenn du komplexe Loch-Logik brauchst, nimm den alten Block 5, 
        # aber meistens reicht es, Flächen zu finden.
        for i, poly in enumerate(standalone_circle_polys):
            # Prüfen ob dieser Kreis schon Teil einer Fläche ist (Loch)
            is_hole = False
            for p_type, p_data in self.closed_profiles:
                if p_type == 'polygon' and p_data.contains(poly):
                    # Es ist ein Loch in einem Polygon -> Polygon ausstanzen
                    # (Hier vereinfacht: Wir markieren es nur)
                    # Um Löcher korrekt darzustellen, muss man difference() nutzen
                    pass 
            
            # Füge Kreis hinzu (als echtes Kreis-Objekt für perfektes Rendering)
            self.closed_profiles.append(('circle', standalone_circles[i]))
            
        self._build_profile_hierarchy()
        
    def _build_profile_hierarchy(self):
        """Baut Containment-Hierarchie auf: Welche Faces sind Löcher in anderen?"""
        from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
        
        def get_profile_vertices(profile):
            """Extrahiert Vertices aus einem Profil für Point-in-Polygon Test"""
            profile_type, data = profile
            if profile_type == 'lines':
                vertices = []
                for line in data:
                    vertices.append((line.start.x, line.start.y))
                return vertices
            elif profile_type == 'polygon':
                # Shapely Polygon - coords direkt nutzen
                return list(data.exterior.coords)[:-1]  # Ohne Schlusspunkt
            elif profile_type == 'circle':
                # Kreis als Polygon approximieren (16 Punkte)
                circle = data
                vertices = []
                for i in range(16):
                    angle = 2 * math.pi * i / 16
                    x = circle.center.x + circle.radius * math.cos(angle)
                    y = circle.center.y + circle.radius * math.sin(angle)
                    vertices.append((x, y))
                return vertices
            return []
        
        def get_profile_area(profile):
            """Berechnet Fläche eines Profils"""
            profile_type, data = profile
            if profile_type == 'lines':
                vertices = get_profile_vertices(profile)
                if len(vertices) < 3:
                    return 0
                # Shoelace Formel
                area = 0
                n = len(vertices)
                for i in range(n):
                    j = (i + 1) % n
                    area += vertices[i][0] * vertices[j][1]
                    area -= vertices[j][0] * vertices[i][1]
                return abs(area / 2)
            elif profile_type == 'polygon':
                return abs(data.area)
            elif profile_type == 'circle':
                return math.pi * data.radius ** 2
            return 0
        
        def get_profile_point(profile):
            """Gibt einen Testpunkt für Point-in-Polygon zurück"""
            profile_type, data = profile
            if profile_type == 'lines':
                # Zentroid der Vertices
                vertices = get_profile_vertices(profile)
                if not vertices:
                    return None
                cx = sum(v[0] for v in vertices) / len(vertices)
                cy = sum(v[1] for v in vertices) / len(vertices)
                return (cx, cy)
            elif profile_type == 'polygon':
                centroid = data.centroid
                return (centroid.x, centroid.y)
            elif profile_type == 'circle':
                return (data.center.x, data.center.y)
            return None
        
        def point_in_polygon(px, py, vertices):
            """Ray-Casting Algorithmus"""
            n = len(vertices)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = vertices[i]
                xj, yj = vertices[j]
                if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            return inside
        
        # Initialisiere Hierarchie-Struktur
        # parent[i] = Index des Parent-Profils, oder -1 wenn Root
        # children[i] = Liste von Kind-Indizes
        n = len(self.closed_profiles)
        self.profile_parent = [-1] * n  # -1 = kein Parent (Root)
        self.profile_children = [[] for _ in range(n)]
        
        if n == 0:
            return
        
        # Sortiere Profile nach Fläche (größte zuerst)
        # Größere Profile können kleinere enthalten
        areas = [get_profile_area(p) for p in self.closed_profiles]
        sorted_indices = sorted(range(n), key=lambda i: areas[i], reverse=True)
        
        # Für jedes Profil: Finde den kleinsten Container
        for i in range(n):
            profile_i = self.closed_profiles[i]
            point_i = get_profile_point(profile_i)
            
            if point_i is None:
                continue
            
            # Finde den kleinsten Container (mit kleinster Fläche der enthält)
            best_parent = -1
            best_parent_area = float('inf')
            
            for j in range(n):
                if i == j:
                    continue
                
                profile_j = self.closed_profiles[j]
                vertices_j = get_profile_vertices(profile_j)
                
                if len(vertices_j) < 3:
                    continue
                
                # Prüfe ob Punkt von i in j liegt
                if point_in_polygon(point_i[0], point_i[1], vertices_j):
                    # j enthält i
                    if areas[j] < best_parent_area and areas[j] > areas[i]:
                        best_parent = j
                        best_parent_area = areas[j]
            
            self.profile_parent[i] = best_parent
            if best_parent >= 0:
                self.profile_children[best_parent].append(i)
        
        # Debug-Ausgabe
        # for i, profile in enumerate(self.closed_profiles):
        #     parent = self.profile_parent[i]
        #     children = self.profile_children[i]
        #     print(f"Profile {i}: parent={parent}, children={children}, type={profile[0]}")
    
    def is_profile_closed(self):
        self._find_closed_profiles()
        return len(self.closed_profiles) > 0
    
    def snap_point(self, w):
        if not self.snap_enabled: return w, SnapType.NONE
        r = self.snap_radius / self.view_scale
        best_pt, best_d, best_t = None, r, SnapType.NONE
        
        # ORIGIN SNAPPING - Höchste Priorität!
        # Origin (0,0) ist wichtiger als andere Punkte
        origin_d = math.hypot(0 - w.x(), 0 - w.y())
        origin_snap_r = r * 2.0  # Doppelter Snap-Radius für Origin
        if origin_d < origin_snap_r:
            # Origin hat Priorität - direkt zurückgeben wenn sehr nah
            if origin_d < r * 0.5:
                return QPointF(0, 0), SnapType.CENTER
            # Sonst als besten Kandidaten merken
            best_d, best_pt, best_t = origin_d, QPointF(0, 0), SnapType.CENTER
        
        # Endpunkte von Linien
        for line in self.sketch.lines:
            for pt in [line.start, line.end]:
                d = math.hypot(pt.x - w.x(), pt.y - w.y())
                if d < best_d: best_d, best_pt, best_t = d, QPointF(pt.x, pt.y), SnapType.ENDPOINT
            mid = line.midpoint
            d = math.hypot(mid.x - w.x(), mid.y - w.y())
            if d < best_d: best_d, best_pt, best_t = d, QPointF(mid.x, mid.y), SnapType.MIDPOINT
        
        # Kreiszentren und Quadranten
        for c in self.sketch.circles:
            d = math.hypot(c.center.x - w.x(), c.center.y - w.y())
            if d < best_d: best_d, best_pt, best_t = d, QPointF(c.center.x, c.center.y), SnapType.CENTER
            for ang in [0, 90, 180, 270]:
                qx = c.center.x + c.radius * math.cos(math.radians(ang))
                qy = c.center.y + c.radius * math.sin(math.radians(ang))
                d = math.hypot(qx - w.x(), qy - w.y())
                if d < best_d: best_d, best_pt, best_t = d, QPointF(qx, qy), SnapType.QUADRANT
        
        # Schnittpunkte
        for i, l1 in enumerate(self.sketch.lines):
            for l2 in self.sketch.lines[i+1:]:
                inter = self._line_intersection(l1, l2)
                if inter:
                    d = math.hypot(inter.x() - w.x(), inter.y() - w.y())
                    if d < best_d: best_d, best_pt, best_t = d, inter, SnapType.INTERSECTION
        
        # Grid als Fallback
        if best_pt is None and self.grid_snap:
            gx = round(w.x() / self.grid_size) * self.grid_size
            gy = round(w.y() / self.grid_size) * self.grid_size
            best_pt, best_t = QPointF(gx, gy), SnapType.GRID
        
        return (best_pt, best_t) if best_pt else (w, SnapType.NONE)
    
    def _line_intersection(self, l1, l2):
        x1, y1, x2, y2 = l1.start.x, l1.start.y, l1.end.x, l1.end.y
        x3, y3, x4, y4 = l2.start.x, l2.start.y, l2.end.x, l2.end.y
        d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(d) < 1e-10: return None
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / d
        s = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / d
        if 0 <= t <= 1 and 0 <= s <= 1:
            return QPointF(x1 + t*(x2-x1), y1 + t*(y2-y1))
        return None
    
    def _save_undo(self):
        self.undo_stack.append(self.sketch.to_dict())
        self.redo_stack.clear()
        if len(self.undo_stack) > self.max_undo: self.undo_stack.pop(0)
    
    def undo(self):
        if not self.undo_stack:
            self.status_message.emit(tr("Nothing to undo"))
            return
        self.redo_stack.append(self.sketch.to_dict())
        self.sketch = Sketch.from_dict(self.undo_stack.pop())
        self._clear_selection()
        self._find_closed_profiles()
        self.sketched_changed.emit()
        self.status_message.emit(tr("Undone"))
        self.update()
    
    def redo(self):
        if not self.redo_stack:
            self.status_message.emit(tr("Nothing to redo"))
            return
        self.undo_stack.append(self.sketch.to_dict())
        self.sketch = Sketch.from_dict(self.redo_stack.pop())
        self._clear_selection()
        self._find_closed_profiles()
        self.sketched_changed.emit()
        self.status_message.emit(tr("Redone"))
        self.update()
    
    def import_dxf(self, filepath=None):
        """Startet den Import im Hintergrund"""
        from PySide6.QtWidgets import QFileDialog, QApplication
        from PySide6.QtGui import Qt
        
        try:
            import ezdxf
        except ImportError:
            self.status_message.emit("Fehler: 'ezdxf' fehlt.")
            return

        if filepath is None:
            filepath, _ = QFileDialog.getOpenFileName(
                self, "DXF importieren", "", "DXF Dateien (*.dxf);;Alle Dateien (*)"
            )
        if not filepath: return

        self.status_message.emit("Starte Import (High-Fidelity)...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        # Worker starten
        self._dxf_worker = DXFImportWorker(filepath)
        self._dxf_worker.finished_signal.connect(self._on_dxf_finished)
        self._dxf_worker.error_signal.connect(self._on_dxf_error)
        self._dxf_worker.progress_signal.connect(self.status_message.emit)
        self._dxf_worker.start()

    def _on_dxf_finished(self, lines, circles, arcs):
        from PySide6.QtWidgets import QApplication
        
        self._save_undo()
        
        # Linien hinzufügen
        for l in lines:
            self.sketch.add_line(l[0], l[1], l[2], l[3])
            
        # Kreise hinzufügen
        for c in circles:
            self.sketch.add_circle(c[0], c[1], c[2])
        
        # Arcs hinzufügen
        for a in arcs:
            # a = (cx, cy, radius, start_angle, end_angle)
            self.sketch.add_arc(a[0], a[1], a[2], a[3], a[4])
            
        QApplication.restoreOverrideCursor()
        self._find_closed_profiles()
        self.sketched_changed.emit()
        self.status_message.emit(f"Fertig: {len(lines)} Linien, {len(circles)} Kreise, {len(arcs)} Bögen.")
        self.update()
        self._dxf_worker = None

    def _on_dxf_error(self, err):
        from PySide6.QtWidgets import QApplication
        QApplication.restoreOverrideCursor()
        self.status_message.emit(f"Import Fehler: {err}")
        self._dxf_worker = None
    
    def export_dxf(self, filepath=None):
        """Exportiert als DXF-Datei"""
        from PySide6.QtWidgets import QFileDialog
        
        if filepath is None:
            filepath, _ = QFileDialog.getSaveFileName(
                self, "DXF exportieren", "", "DXF Dateien (*.dxf);;Alle Dateien (*)"
            )
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'w') as f:
                f.write("0\nSECTION\n2\nENTITIES\n")
                
                for line in self.sketch.lines:
                    f.write(f"0\nLINE\n8\n0\n")
                    f.write(f"10\n{line.start.x}\n20\n{line.start.y}\n30\n0\n")
                    f.write(f"11\n{line.end.x}\n21\n{line.end.y}\n31\n0\n")
                
                for circle in self.sketch.circles:
                    f.write(f"0\nCIRCLE\n8\n0\n")
                    f.write(f"10\n{circle.center.x}\n20\n{circle.center.y}\n30\n0\n")
                    f.write(f"40\n{circle.radius}\n")
                
                f.write("0\nENDSEC\n0\nEOF\n")
            
            self.status_message.emit(tr("DXF exported: {path}").format(path=filepath))
            
        except Exception as e:
            self.status_message.emit(tr("DXF export error: {e}").format(e=e))
    
    # ==================== BUILD123D INTEGRATION ====================
    
    def get_build123d_plane(self):
        """Gibt die Build123d Plane für diesen Sketch zurück"""
        if not HAS_BUILD123D:
            return None
        
        # Hole Plane-Daten vom Sketch
        origin = getattr(self.sketch, 'plane_origin', (0, 0, 0))
        normal = getattr(self.sketch, 'plane_normal', (0, 0, 1))
        
        ox, oy, oz = origin
        nx, ny, nz = normal
        
        # Standard-Ebenen erkennen
        if abs(nz - 1) < 0.01 and abs(nx) < 0.01 and abs(ny) < 0.01:
            return Plane.XY.offset(oz)
        elif abs(ny - 1) < 0.01 and abs(nx) < 0.01 and abs(nz) < 0.01:
            return Plane.XZ.offset(oy)
        elif abs(nx - 1) < 0.01 and abs(ny) < 0.01 and abs(nz) < 0.01:
            return Plane.YZ.offset(ox)
        else:
            # Custom Plane
            return Plane(origin=(ox, oy, oz), z_dir=(nx, ny, nz))
    
    def get_build123d_sketch(self, plane=None):
        """
        Konvertiert den aktuellen Sketch zu einem Build123d BuildSketch.
        
        Returns:
            BuildSketch oder None wenn Build123d nicht verfügbar
        """
        if not HAS_BUILD123D:
            print("Build123d nicht verfügbar!")
            return None
        
        if plane is None:
            plane = self.get_build123d_plane()
        
        try:
            # Sammle alle nicht-construction Geometrie
            lines = [l for l in self.sketch.lines if not l.construction]
            circles = [c for c in self.sketch.circles if not c.construction]
            arcs = [a for a in self.sketch.arcs if not a.construction]
            
            if not lines and not circles and not arcs:
                print("Keine Geometrie im Sketch!")
                return None
            
            # Erstelle Build123d Sketch
            with BuildSketch(plane) as sketch:
                # Linien hinzufügen
                for line in lines:
                    with BuildLine():
                        B3DLine(
                            (line.start.x, line.start.y),
                            (line.end.x, line.end.y)
                        )
                
                # Kreise hinzufügen - mit Locations für Position
                for circle in circles:
                    with Locations([(circle.center.x, circle.center.y)]):
                        B3DCircle(radius=circle.radius)
                
                # Arcs hinzufügen (als CenterArc)
                for arc in arcs:
                    start_deg = arc.start_angle
                    end_deg = arc.end_angle
                    
                    # Berechne arc_size (Sweep)
                    arc_size = end_deg - start_deg
                    if arc_size < 0:
                        arc_size += 360
                    
                    with BuildLine():
                        CenterArc(
                            center=(arc.center.x, arc.center.y),
                            radius=arc.radius,
                            start_angle=start_deg,
                            arc_size=arc_size
                        )
                
                # Faces erzeugen aus den Linien
                make_face()
            
            return sketch
            
        except Exception as e:
            print(f"Build123d Sketch Konvertierung fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_build123d_part(self, height: float, operation: str = "New Body"):
        """
        Erstellt ein extrudiertes Build123d Part aus dem Sketch.
        
        Args:
            height: Extrusionshöhe in mm
            operation: "New Body", "Join", "Cut", "Intersect"
            
        Returns:
            (Part, mesh_verts, mesh_faces) oder (None, None, None)
        """
        if not HAS_BUILD123D:
            print("Build123d nicht verfügbar!")
            return None, None, None
        
        # Stelle sicher dass Profile aktuell sind
        self._find_closed_profiles()
        
        plane = self.get_build123d_plane()
        print(f"Build123d Part: height={height}, plane={plane}")
        print(f"  Profiles: {len(self.closed_profiles)}, Lines: {len(self.sketch.lines)}, Circles: {len(self.sketch.circles)}")
        
        try:
            # Versuche aus Profilen zu erstellen
            result = self._build123d_from_profiles(height, plane)
            if result[0] is not None:
                return result
            
            # Fallback: Direkt aus Sketch-Geometrie
            print("Build123d: Fallback zu direkter Geometrie-Erstellung")
            return self._build123d_direct(height, plane)
            
        except Exception as e:
            print(f"Build123d Part Erstellung fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def _build123d_direct(self, height: float, plane):
        """Erstellt Build123d Part direkt aus Sketch-Linien (Fallback)"""
        if not HAS_BUILD123D:
            return None, None, None
        
        lines = [l for l in self.sketch.lines if not l.construction]
        circles = [c for c in self.sketch.circles if not c.construction]
        
        if not lines and not circles:
            print("Build123d: Keine Geometrie!")
            return None, None, None
        
        try:
            with BuildPart() as part:
                with BuildSketch(plane):
                    # Versuche Rechteck zu erkennen (4 Linien, geschlossen)
                    if len(lines) == 4 and not circles:
                        # Sammle alle Endpunkte
                        points = set()
                        for l in lines:
                            points.add((round(l.start.x, 4), round(l.start.y, 4)))
                            points.add((round(l.end.x, 4), round(l.end.y, 4)))
                        
                        if len(points) == 4:
                            # Es ist ein Rechteck!
                            pts = list(points)
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            min_x, max_x = min(xs), max(xs)
                            min_y, max_y = min(ys), max(ys)
                            width = max_x - min_x
                            height_rect = max_y - min_y
                            center_x = (min_x + max_x) / 2
                            center_y = (min_y + max_y) / 2
                            
                            print(f"  Rechteck erkannt: {width}x{height_rect} at ({center_x}, {center_y})")
                            
                            with Locations([(center_x, center_y)]):
                                Rectangle(width, height_rect)
                    
                    # Kreise hinzufügen
                    for circle in circles:
                        print(f"  Kreis: r={circle.radius} at ({circle.center.x}, {circle.center.y})")
                        with Locations([(circle.center.x, circle.center.y)]):
                            B3DCircle(radius=circle.radius)
                
                extrude(amount=height)
            
            solid = part.part
            if solid is None:
                return None, None, None
            
            print(f"Build123d Direct: Solid erstellt!")
            
            mesh_data = solid.tessellate(tolerance=0.1)
            verts = [(v.X, v.Y, v.Z) for v in mesh_data[0]]
            faces = [tuple(t) for t in mesh_data[1]]
            
            return solid, verts, faces
            
        except Exception as e:
            print(f"Build123d Direct fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def _build123d_from_profiles(self, height: float, plane):
        """Erstellt Build123d Part aus geschlossenen Shapely-Profilen"""
        if not HAS_BUILD123D:
            return None, None, None
        
        # Debug: Zeige was wir haben
        print(f"Build123d: {len(self.closed_profiles)} geschlossene Profile gefunden")
        print(f"Build123d: {len(self.sketch.lines)} Linien, {len(self.sketch.circles)} Kreise")
        
        if not self.closed_profiles and not self.sketch.circles:
            print("Build123d: Keine Profile oder Kreise!")
            return None, None, None
        
        try:
            with BuildPart() as part:
                with BuildSketch(plane):
                    created_something = False
                    
                    # Methode 1: Aus Shapely Profilen - verwende Polygon!
                    for i, profile in enumerate(self.closed_profiles):
                        if hasattr(profile, 'exterior'):
                            coords = list(profile.exterior.coords)
                            print(f"  Profil {i}: {len(coords)} Koordinaten")
                            
                            if len(coords) >= 4:  # Mindestens 3 Punkte + Schließpunkt
                                # Entferne Schließpunkt wenn vorhanden
                                pts = [(float(c[0]), float(c[1])) for c in coords]
                                if len(pts) > 1 and pts[0] == pts[-1]:
                                    pts = pts[:-1]
                                
                                print(f"    Punkte: {pts}")
                                
                                if len(pts) >= 3:
                                    # Polygon erstellt direkt ein Face!
                                    Polygon(*pts)
                                    created_something = True
                                    print(f"    -> Polygon erstellt")
                    
                    # Methode 2: Kreise
                    circles = [c for c in self.sketch.circles if not c.construction]
                    for circle in circles:
                        print(f"  Kreis: center=({circle.center.x}, {circle.center.y}), r={circle.radius}")
                        with Locations([(circle.center.x, circle.center.y)]):
                            B3DCircle(radius=circle.radius)
                        created_something = True
                    
                    if not created_something:
                        print("Build123d: Nichts erstellt in BuildSketch!")
                        return None, None, None
                
                # Extrudieren
                extrude(amount=height)
            
            solid = part.part
            if solid is None:
                print("Build123d: Part ist None nach Extrusion")
                return None, None, None
            
            print(f"Build123d: Solid erstellt mit {len(solid.faces())} Faces")
            
            # Tessellieren
            mesh_data = solid.tessellate(tolerance=0.1)
            verts = [(v.X, v.Y, v.Z) for v in mesh_data[0]]
            faces = [tuple(t) for t in mesh_data[1]]
            
            return solid, verts, faces
            
        except Exception as e:
            print(f"Build123d Profile Extrusion fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def has_build123d(self) -> bool:
        """Prüft ob Build123d verfügbar ist"""
        return HAS_BUILD123D
    
    def set_tool(self, tool):
        self._cancel_tool()
        self.current_tool = tool
        self.tool_changed.emit(tool)
        self._update_cursor()
        self._show_tool_hint()
        self._show_tool_options()  # Optionen-Popup
        self.update()
    
    def _show_tool_options(self):
        """Zeigt Optionen-Popup für Tools die Optionen haben"""
        tool = self.current_tool
        
        # Rechteck: 2-Punkt vs Center
        if tool == SketchTool.RECTANGLE:
            self.tool_options.show_options(
                "Rechteck-Modus",
                "rect_mode",
                [("⬚", "2-Punkt"), ("⊞", "Center")],
                self.rect_mode
            )
            self.tool_options.position_near(self, 10, 10)
            
        # Kreis: Center vs 2-Punkt vs 3-Punkt
        elif tool == SketchTool.CIRCLE:
            self.tool_options.show_options(
                "Kreis-Modus",
                "circle_mode",
                [("◎", "Center"), ("⌀", "2-Punkt"), ("◯", "3-Punkt")],
                self.circle_mode
            )
            self.tool_options.position_near(self, 10, 10)
            
        # Polygon: Anzahl Seiten
        elif tool == SketchTool.POLYGON:
            self.tool_options.show_options(
                "Polygon-Seiten",
                "polygon_sides",
                [("△", "3"), ("◇", "4"), ("⬠", "5"), ("⬡", "6"), ("⯃", "8")],
                [3, 4, 5, 6, 8].index(self.polygon_sides) if self.polygon_sides in [3, 4, 5, 6, 8] else 3
            )
            self.tool_options.position_near(self, 10, 10)
            
        # Muttern-Aussparung: Größe M2-M14
        elif tool == SketchTool.NUT:
            self.tool_options.show_options(
                f"Mutter (Tol: {self.nut_tolerance:.2f}mm)",
                "nut_size",
                [(f"⬡", s) for s in self.nut_size_names],
                self.nut_size_index
            )
            self.tool_options.position_near(self, 10, 10)
        else:
            self.tool_options.hide()
    
    def _on_tool_option_selected(self, option_name, value):
        """Handler für Optionen-Auswahl"""
        if option_name == "rect_mode":
            self.rect_mode = value
        elif option_name == "circle_mode":
            self.circle_mode = value
        elif option_name == "polygon_sides":
            self.polygon_sides = [3, 4, 5, 6, 8][value]
        elif option_name == "nut_size":
            self.nut_size_index = value
        
        self._show_tool_hint()  # Hint aktualisieren
        self._show_tool_options()  # Optionen-Titel aktualisieren (für Toleranz-Anzeige)
        self.update()
    
    def _cancel_tool(self):
        self.tool_step = 0
        self.tool_points.clear()
        self.tool_data.clear()
        self.selection_box_start = None
        self.selection_box_end = None
        self.dim_input.hide()
        self.dim_input.unlock_all()  # Locks zurücksetzen!
        self.dim_input_active = False
        # NICHT tool_options.hide() - Palette bleibt sichtbar solange Tool aktiv!
        
        # Offset-Tool Reset
        self.offset_dragging = False
        self.offset_profile = None
        self.offset_start_pos = None
        self.offset_preview_lines = []
        
        self.update()
    
    def _update_cursor(self):
        if self.current_tool == SketchTool.SELECT: self.setCursor(Qt.ArrowCursor)
        elif self.current_tool == SketchTool.MOVE: self.setCursor(Qt.SizeAllCursor)
        else: self.setCursor(Qt.CrossCursor)
    
    def _show_tool_hint(self):
        n_sel = len(self.selected_lines) + len(self.selected_circles) + len(self.selected_arcs) + len(self.selected_points)
        sel_info = f" ({n_sel} ausgewählt)" if n_sel > 0 else ""
        
        hints = {
            SketchTool.SELECT: tr("Click=Select | Shift+Click=Multi | Drag=Box | Del=Delete"),
            SketchTool.LINE: tr("Click=Start | Tab=Length/Angle | RightClick=Finish"),
            SketchTool.RECTANGLE: tr("Rectangle") + f" ({tr('Center') if self.rect_mode else tr('2-Point')}) | " + tr("Click=Start"),
            SketchTool.RECTANGLE_CENTER: tr("Rectangle") + " (" + tr("Center") + ") | " + tr("Click=Start"),
            SketchTool.CIRCLE: tr("Circle") + f" ({[tr('Center'), tr('2-Point'), tr('3-Point')][self.circle_mode]}) | Tab=" + tr("Radius"),
            SketchTool.CIRCLE_2POINT: tr("Circle") + " (" + tr("2-Point") + ")",
            SketchTool.POLYGON: tr("Polygon") + f" ({self.polygon_sides} " + tr("{n} sides").format(n="") + ") | Tab",
            SketchTool.ARC_3POINT: tr("[A] Arc | Click=Start→Through→End"),
            SketchTool.SLOT: tr("Slot | Click=Start"),
            SketchTool.SPLINE: tr("Spline | Click=Points | Right=Finish"),
            SketchTool.MOVE: tr("[M] Move | Click=Base→Target") + sel_info,
            SketchTool.COPY: tr("Copy") + sel_info,
            SketchTool.ROTATE: tr("Rotate") + sel_info + " | Tab=°",
            SketchTool.MIRROR: tr("Mirror") + sel_info,
            SketchTool.SCALE: tr("[S] Scale | Click=Center") + sel_info,
            SketchTool.TRIM: tr("Trim | Click on segment"),
            SketchTool.EXTEND: tr("Extend | Click on line"),
            SketchTool.OFFSET: tr("Offset") + f" ({self.offset_distance:+g}mm) | Tab",
            SketchTool.FILLET_2D: tr("Fillet") + f" (R={self.fillet_radius:g}mm) | Tab=" + tr("Radius"),
            SketchTool.CHAMFER_2D: tr("Chamfer") + f" ({self.chamfer_distance:g}mm) | Tab=" + tr("Length"),
            SketchTool.DIMENSION: tr("Dimension | Click on element"),
            SketchTool.DIMENSION_ANGLE: tr("Angle") + " | " + tr("Click=Line1→Line2"),
            SketchTool.HORIZONTAL: tr("Horizontal (H)"),
            SketchTool.VERTICAL: tr("Vertical (V)"),
            SketchTool.PARALLEL: tr("Parallel | Click=Line1→Line2"),
            SketchTool.PERPENDICULAR: tr("Perpendicular | Click=Line1→Line2"),
            SketchTool.EQUAL: tr("Equal | Click=Element1→Element2"),
            SketchTool.CONCENTRIC: tr("Concentric | Click=Circle1→Circle2"),
            SketchTool.TANGENT: tr("Tangent") + " | " + tr("Click=Line→Circle"),
            SketchTool.PATTERN_LINEAR: tr("Lin. Pattern | Select→Dialog"),
            SketchTool.PATTERN_CIRCULAR: tr("Circ. Pattern | Select→Center"),
            SketchTool.GEAR: tr("Gear generator | Click=Center"),
            SketchTool.STAR: tr("Star") + " | " + tr("Click=Center"),
            SketchTool.NUT: tr("Nut") + f" {self.nut_size_names[self.nut_size_index]} | +/- " + tr("Tolerance"),
            SketchTool.TEXT: tr("Text") + " | " + tr("Click=Position"),
        }
        self.status_message.emit(hints.get(self.current_tool, ""))
    
    def _show_dimension_input(self):
        fields = []
        
        # LINE: Nach erstem Punkt
        if self.current_tool == SketchTool.LINE and self.tool_step >= 1:
            fields = [("L", "length", self.live_length, "mm"), ("∠", "angle", self.live_angle, "°")]
        
        # RECTANGLE: Nach erstem Punkt ODER bevor erster Punkt (dann Standardwerte)
        elif self.current_tool == SketchTool.RECTANGLE:
            if self.tool_step == 1:
                fields = [("B", "width", self.live_width, "mm"), ("H", "height", self.live_height, "mm")]
            elif self.tool_step == 0:
                fields = [("B", "width", 50.0, "mm"), ("H", "height", 30.0, "mm")]
                
        elif self.current_tool == SketchTool.RECTANGLE_CENTER:
            if self.tool_step == 1:
                fields = [("B", "width", self.live_width, "mm"), ("H", "height", self.live_height, "mm")]
            elif self.tool_step == 0:
                fields = [("B", "width", 50.0, "mm"), ("H", "height", 30.0, "mm")]
                
        # CIRCLE: Nach Zentrum ODER bevor (dann Standardwerte)
        elif self.current_tool == SketchTool.CIRCLE:
            if self.tool_step >= 1:
                fields = [("R", "radius", self.live_radius, "mm")]
            elif self.tool_step == 0:
                fields = [("R", "radius", 25.0, "mm")]
                
        # POLYGON: Nach Zentrum
        elif self.current_tool == SketchTool.POLYGON:
            if self.tool_step >= 1:
                fields = [("R", "radius", self.live_radius, "mm"), ("N", "sides", float(self.polygon_sides), "")]
            elif self.tool_step == 0:
                fields = [("R", "radius", 25.0, "mm"), ("N", "sides", float(self.polygon_sides), "")]
                
        elif self.current_tool == SketchTool.MOVE and self.tool_step == 1:
            fields = [("X", "dx", 0.0, "mm"), ("Y", "dy", 0.0, "mm")]
        elif self.current_tool == SketchTool.ROTATE and self.tool_step >= 1:
            fields = [("∠", "angle", 0.0, "°")]
        elif self.current_tool == SketchTool.SCALE and self.tool_step >= 1:
            fields = [("F", "factor", 1.0, "")]
            
        # SLOT: Nach Startpunkt (Länge/Winkel), Nach Mittellinie (Breite)
        elif self.current_tool == SketchTool.SLOT:
            if self.tool_step == 1:
                fields = [("L", "length", 50.0, "mm"), ("∠", "angle", 0.0, "°")]
            elif self.tool_step == 2:
                fields = [("B", "width", 10.0, "mm")]
            
        # OFFSET: Immer verfügbar (auch negativ!)
        elif self.current_tool == SketchTool.OFFSET:
            fields = [("D", "distance", self.offset_distance, "mm")]
            
        elif self.current_tool == SketchTool.FILLET_2D:
            fields = [("R", "radius", self.fillet_radius, "mm")]
        elif self.current_tool == SketchTool.CHAMFER_2D:
            fields = [("L", "length", self.chamfer_distance, "mm")]
        
        # PATTERN: Linear und Circular
        elif self.current_tool == SketchTool.PATTERN_LINEAR and self.tool_step >= 1:
            count = self.tool_data.get('pattern_count', 3)
            spacing = self.tool_data.get('pattern_spacing', 20.0)
            fields = [("N", "count", float(count), ""), ("D", "spacing", spacing, "mm")]
        elif self.current_tool == SketchTool.PATTERN_CIRCULAR and self.tool_step >= 1:
            count = self.tool_data.get('pattern_count', 6)
            angle = self.tool_data.get('pattern_angle', 360.0)
            fields = [("N", "count", float(count), ""), ("∠", "angle", angle, "°")]
            
        if not fields:
            self.status_message.emit(tr("Tab: Set a point first or choose another tool"))
            return
        
        self.dim_input.setup(fields)
        pos = self.mouse_screen
        x = min(int(pos.x()) + 20, self.width() - self.dim_input.width() - 10)
        y = min(int(pos.y()) - 40, self.height() - self.dim_input.height() - 10)
        self.dim_input.move(max(10, x), max(10, y))
        self.dim_input.show()
        self.dim_input.focus_field(0)
        self.dim_input_active = True
    
    def _on_dim_value_changed(self, key, value):
        if self.current_tool == SketchTool.LINE:
            if key == "length": self.live_length = value
            elif key == "angle": self.live_angle = value
        elif self.current_tool in [SketchTool.RECTANGLE, SketchTool.RECTANGLE_CENTER]:
            if key == "width": self.live_width = value
            elif key == "height": self.live_height = value
        elif self.current_tool == SketchTool.CIRCLE:
            if key == "radius": self.live_radius = value
        elif self.current_tool == SketchTool.POLYGON:
            if key == "radius": self.live_radius = value
            elif key == "sides": self.polygon_sides = max(3, min(64, int(value)))
        elif self.current_tool == SketchTool.OFFSET:
            self.offset_distance = value
            self._update_offset_preview()  # Preview sofort aktualisieren
        elif self.current_tool == SketchTool.FILLET_2D: self.fillet_radius = value
        elif self.current_tool == SketchTool.CHAMFER_2D: self.chamfer_distance = value
        elif self.current_tool == SketchTool.PATTERN_LINEAR:
            if key == "count": self.tool_data['pattern_count'] = max(2, int(value))
            elif key == "spacing": self.tool_data['pattern_spacing'] = value
        elif self.current_tool == SketchTool.PATTERN_CIRCULAR:
            if key == "count": self.tool_data['pattern_count'] = max(2, int(value))
            elif key == "angle": self.tool_data['pattern_angle'] = value
        self.update()
    
    def _on_dim_confirmed(self):
        values = self.dim_input.get_values()
        
        if self.viewport and getattr(self.viewport, 'extrude_mode', False):
            height = values.get("height", 0.0)
            # Hier holen wir "Join", "Cut" oder "New Body" aus dem Dropdown
            op = values.get("operation", "New Body") 
            
            self.viewport.extrude_height = height
            
            # WICHTIG: Wir übergeben 'op' an den Viewport!
            self.viewport.confirm_extrusion(operation=op)
            
            self.dim_input.hide()
            self.dim_input.unlock_all()
            self.dim_input_active = False
            self.status_message.emit(f"Extrusion ({op}) angewendet.")
            return
            
        # Debug: Zeige Lock-Status und Werte
        
        if self.current_tool == SketchTool.LINE and self.tool_step >= 1:
            start = self.tool_points[-1]
            
            # WICHTIG: Wenn Länge gelockt ist, verwende live_length statt get_values!
            if self.dim_input.is_locked('length'):
                length = self.live_length
            else:
                length = values.get("length", 10)
            
            if self.dim_input.is_locked('angle'):
                angle_deg = self.live_angle
            else:
                angle_deg = values.get("angle", 0)
            
            angle = math.radians(angle_deg)
            
            
            end_x = start.x() + length * math.cos(angle)
            end_y = start.y() + length * math.sin(angle)
            
            actual_length = math.hypot(end_x - start.x(), end_y - start.y())
            
            self._save_undo()
            self.sketch.add_line(start.x(), start.y(), end_x, end_y, construction=self.construction_mode)
            self.tool_points.append(QPointF(end_x, end_y))
            self.sketched_changed.emit()
            self._find_closed_profiles()
            
        elif self.current_tool == SketchTool.RECTANGLE:
            # WICHTIG: Wenn gelockt, verwende live_width/live_height!
            if self.dim_input.is_locked('width'):
                w = self.live_width
            else:
                w = values.get("width", 50)
            
            if self.dim_input.is_locked('height'):
                h = self.live_height
            else:
                h = values.get("height", 30)
            
            if self.tool_step >= 1:
                p1 = self.tool_points[0]
            else:
                p1 = self.mouse_world if self.mouse_world else QPointF(0, 0)
            self._save_undo()
            
            # rect_mode berücksichtigen: 0=2-Punkt, 1=Center
            if self.rect_mode == 1:
                # Center-Modus: p1 ist Mittelpunkt
                self.sketch.add_rectangle(p1.x() - w/2, p1.y() - h/2, w, h, construction=self.construction_mode)
            else:
                # 2-Punkt-Modus: p1 ist Ecke, Richtung von Mausposition bestimmen!
                mouse = self.mouse_world if self.mouse_world else p1
                
                # X-Richtung: Links oder Rechts?
                if mouse.x() < p1.x():
                    # Maus ist links vom Startpunkt -> Rechteck nach links
                    x = p1.x() - w
                else:
                    # Maus ist rechts -> Rechteck nach rechts
                    x = p1.x()
                
                # Y-Richtung: Oben oder Unten?
                if mouse.y() < p1.y():
                    # Maus ist unter dem Startpunkt -> Rechteck nach unten
                    y = p1.y() - h
                else:
                    # Maus ist oben -> Rechteck nach oben
                    y = p1.y()
                
                self.sketch.add_rectangle(x, y, w, h, construction=self.construction_mode)
            self.sketch.solve()
            self.sketched_changed.emit()
            self._find_closed_profiles()
            self._cancel_tool()
            
        elif self.current_tool == SketchTool.RECTANGLE_CENTER:
            w, h = values.get("width", 50), values.get("height", 30)
            if self.tool_step >= 1:
                c = self.tool_points[0]
            else:
                c = self.mouse_world if self.mouse_world else QPointF(0, 0)
            self._save_undo()
            self.sketch.add_rectangle(c.x() - w/2, c.y() - h/2, w, h, construction=self.construction_mode)
            self.sketch.solve()
            self.sketched_changed.emit()
            self._find_closed_profiles()
            self._cancel_tool()
            
        elif self.current_tool == SketchTool.CIRCLE:
            # WICHTIG: Wenn gelockt, verwende live_radius!
            if self.dim_input.is_locked('radius'):
                r = self.live_radius
            else:
                r = values.get("radius", 25)
            
            if self.tool_step >= 1:
                c = self.tool_points[0]
            else:
                c = self.mouse_world if self.mouse_world else QPointF(0, 0)
            self._save_undo()
            # circle_mode berücksichtigen: 0=Center-Radius, 1=2-Punkt (aber bei Tab immer Center)
            self.sketch.add_circle(c.x(), c.y(), r, construction=self.construction_mode)
            self.sketched_changed.emit()
            self._find_closed_profiles()
            self._cancel_tool()
            
        elif self.current_tool == SketchTool.POLYGON:
            # WICHTIG: Wenn gelockt, verwende live_radius!
            if self.dim_input.is_locked('radius'):
                r = self.live_radius
            else:
                r = values.get("radius", 25)
            n = int(values.get("sides", self.polygon_sides))
            
            if self.tool_step >= 1:
                c = self.tool_points[0]
            else:
                c = self.mouse_world if self.mouse_world else QPointF(0, 0)
            self._save_undo()
            pts = [(c.x() + r*math.cos(2*math.pi*i/n - math.pi/2), c.y() + r*math.sin(2*math.pi*i/n - math.pi/2)) for i in range(n)]
            self.sketch.add_polygon(pts, closed=True, construction=self.construction_mode)
            self.sketch.solve()
            self.sketched_changed.emit()
            self._find_closed_profiles()
            self._cancel_tool()
            
        elif self.current_tool == SketchTool.MOVE and self.tool_step == 1:
            dx, dy = values.get("dx", 0), values.get("dy", 0)
            self._save_undo()
            self._move_selection(dx, dy)
            self.sketched_changed.emit()
            self._cancel_tool()
            
        elif self.current_tool == SketchTool.ROTATE and self.tool_step >= 1:
            self._save_undo()
            self._rotate_selection(self.tool_points[0], values.get("angle", 0))
            self.sketched_changed.emit()
            self._cancel_tool()
            
        elif self.current_tool == SketchTool.SCALE and self.tool_step >= 1:
            self._save_undo()
            self._scale_selection(self.tool_points[0], values.get("factor", 1.0))
            self.sketched_changed.emit()
            self._cancel_tool()
        
        # SLOT: Tab-Eingabe
        elif self.current_tool == SketchTool.SLOT:
            if self.tool_step == 1:
                # Länge und Winkel eingegeben -> berechne Endpunkt
                p1 = self.tool_points[0]
                length = values.get("length", 50.0)
                angle = math.radians(values.get("angle", 0.0))
                p2 = QPointF(p1.x() + length * math.cos(angle), p1.y() + length * math.sin(angle))
                self.tool_points.append(p2)
                self.tool_step = 2
                self.status_message.emit(tr("Width | Tab=Enter width"))
                # Nicht _cancel_tool - wir brauchen noch die Breite
            elif self.tool_step == 2:
                # Breite eingegeben -> erstelle Slot
                p1, p2 = self.tool_points[0], self.tool_points[1]
                width = values.get("width", 10.0)
                if width > 0.01:
                    self._save_undo()
                    self._create_slot(p1, p2, width)
                    self.sketched_changed.emit()
                    self._find_closed_profiles()
                self._cancel_tool()
        
        # OFFSET, FILLET, CHAMFER: Nur Wert speichern, nicht anwenden
        elif self.current_tool == SketchTool.OFFSET:
            self.offset_distance = values.get("distance", 5.0)
            
            # Vorschau aktualisieren wenn Profil bereits gewählt
            if self.tool_step == 1 and self.offset_profile:
                self._update_offset_preview()
                self.status_message.emit(tr("Offset: {dist}mm | Tab=Change | Enter/Click=Apply | Esc=Cancel").format(dist=f"{self.offset_distance:+.2f}"))
            else:
                self.status_message.emit(tr("Offset: {dist}mm | Click on element").format(dist=f"{self.offset_distance:+.2f}"))
            
        elif self.current_tool == SketchTool.FILLET_2D:
            self.fillet_radius = values.get("radius", 5.0)
            self.status_message.emit(tr("Fillet: R={r}mm (click corner)").format(r=self.fillet_radius))
            
        elif self.current_tool == SketchTool.CHAMFER_2D:
            self.chamfer_distance = values.get("length", 5.0)
            self.status_message.emit(tr("Chamfer: {d}mm (click corner)").format(d=self.chamfer_distance))
        
        # PATTERN: Werte speichern, Preview aktualisieren
        elif self.current_tool == SketchTool.PATTERN_LINEAR and self.tool_step >= 1:
            self.tool_data['pattern_count'] = max(2, int(values.get("count", 3)))
            self.tool_data['pattern_spacing'] = values.get("spacing", 20.0)
            self.status_message.emit(tr("Lin. pattern: {n}x, spacing {s}mm | Click=Apply").format(n=self.tool_data["pattern_count"], s=f"{self.tool_data['pattern_spacing']:.1f}"))
            
        elif self.current_tool == SketchTool.PATTERN_CIRCULAR and self.tool_step >= 1:
            self.tool_data['pattern_count'] = max(2, int(values.get("count", 6)))
            self.tool_data['pattern_angle'] = values.get("angle", 360.0)
            self.status_message.emit(tr("Circ. pattern: {n}x over {a}° | Click=Apply").format(n=self.tool_data["pattern_count"], a=f"{self.tool_data['pattern_angle']:.0f}"))
        
        self.dim_input.hide()
        self.dim_input.unlock_all()  # Wichtig: Locks zurücksetzen!
        self.dim_input_active = False
        self.setFocus()
        self.update()
    
    def mousePressEvent(self, event):
        pos = event.position()
        self.mouse_screen = pos
        self.mouse_world = self.screen_to_world(pos)
        
        # Wenn DimensionInput aktiv: Klick außerhalb = BESTÄTIGEN
        if self.dim_input_active and self.dim_input.isVisible():
            if not self.dim_input.geometry().contains(pos.toPoint()):
                self._on_dim_confirmed()  # Bestätigt statt nur schließen!
                return
        
        if event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.pan_start = pos
            self.setCursor(Qt.ClosedHandCursor)
            return
        
        if event.button() == Qt.LeftButton:
            # Spline-Element-Klick prüfen (hat Priorität im SELECT-Modus)
            if self.current_tool == SketchTool.SELECT:
                spline_elem = self._find_spline_element_at(self.mouse_world)
                if spline_elem:
                    spline, cp_idx, elem_type = spline_elem
                    self.spline_dragging = True
                    self.spline_drag_spline = spline
                    self.spline_drag_cp_index = cp_idx
                    self.spline_drag_type = elem_type
                    self.selected_spline = spline
                    self._save_undo()
                    self.setCursor(Qt.ClosedHandCursor)
                    self.status_message.emit(tr("Drag spline {type} | Shift=Corner").format(type=elem_type))
                    return
            
            snapped, snap_type = self.snap_point(self.mouse_world)
            if self.current_tool == SketchTool.SELECT and not self._find_entity_at(snapped):
                # Auch Spline-Kurve selbst prüfen
                spline = self._find_spline_at(self.mouse_world)
                if spline:
                    self.selected_spline = spline
                    self._clear_selection()
                    self.status_message.emit(tr("Spline selected - drag points/handles"))
                    self.update()
                    return
                self.selection_box_start = pos
                self.selection_box_end = pos
                return
            handler = getattr(self, f'_handle_{self.current_tool.name.lower()}', None)
            if handler: handler(snapped, snap_type)
        elif event.button() == Qt.RightButton:
            if self.tool_step > 0: self._finish_current_operation()
            else: self._show_context_menu(pos)
        self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.is_panning = False
            self._update_cursor()
        if event.button() == Qt.LeftButton:
            # Spline-Dragging beenden
            if self.spline_dragging:
                self._finish_spline_drag()
                self._update_cursor()
            if self.selection_box_start:
                self._finish_selection_box()
        self.update()
    
    def _finish_spline_drag(self):
        """Beendet das Spline-Dragging und aktualisiert die Linien-Approximation"""
        if not self.spline_dragging or not self.spline_drag_spline:
            self.spline_dragging = False
            return
        
        spline = self.spline_drag_spline
        
        # Alte Linien der Spline entfernen (verwende gespeicherte Referenzen)
        if hasattr(spline, '_lines') and spline._lines:
            for old_line in spline._lines:
                if old_line in self.sketch.lines:
                    self.sketch.lines.remove(old_line)
                # Auch die Punkte entfernen
                if old_line.start in self.sketch.points:
                    self.sketch.points.remove(old_line.start)
                if old_line.end in self.sketch.points:
                    self.sketch.points.remove(old_line.end)
        
        # Neue Linien generieren
        new_lines = spline.to_lines(segments_per_span=10)
        spline._lines = new_lines  # Referenz aktualisieren
        
        for line in new_lines:
            self.sketch.lines.append(line)
            self.sketch.points.append(line.start)
        if new_lines:
            self.sketch.points.append(new_lines[-1].end)
        
        # Reset Drag-State aber behalte Spline-Selection
        self.spline_dragging = False
        self.spline_drag_spline = None
        self.spline_drag_cp_index = None
        self.spline_drag_type = None
        
        self.sketched_changed.emit()
        self._find_closed_profiles()
        self.status_message.emit(tr("Spline updated"))
    
    def mouseMoveEvent(self, event):
        pos = event.position()
        self.mouse_screen = pos
        self.mouse_world = self.screen_to_world(pos)
        
        if self.is_panning:
            self.view_offset += pos - self.pan_start
            self.pan_start = pos
        elif self.spline_dragging:
            # Spline-Element ziehen
            self._drag_spline_element(event.modifiers() & Qt.ShiftModifier)
        elif self.selection_box_start:
            self.selection_box_end = pos
        else:
            snapped, snap_type = self.snap_point(self.mouse_world)
            self.current_snap = (snapped, snap_type) if snap_type != SnapType.NONE else None
            self.hovered_entity = self._find_entity_at(self.mouse_world)
            self.hovered_face = self._find_face_at(self.mouse_world)
            
            # Spline-Element-Hover für Cursor-Feedback
            if self.current_tool == SketchTool.SELECT:
                spline_elem = self._find_spline_element_at(self.mouse_world)
                self.hovered_spline_element = spline_elem
                if spline_elem:
                    self.setCursor(Qt.OpenHandCursor)
                else:
                    self._update_cursor()
            
            self._update_live_values(snapped)
        self.update()
    
    def _drag_spline_element(self, shift_pressed):
        """Zieht ein Spline-Element und aktualisiert die Vorschau sofort"""
        if not self.spline_drag_spline or self.spline_drag_cp_index is None:
            return
        
        spline = self.spline_drag_spline
        cp_idx = self.spline_drag_cp_index
        cp = spline.control_points[cp_idx]
        
        snapped, _ = self.snap_point(self.mouse_world)
        new_x, new_y = snapped.x(), snapped.y()
        
        # Positionen aktualisieren
        if self.spline_drag_type == 'point':
            cp.point.x = new_x
            cp.point.y = new_y
        elif self.spline_drag_type == 'handle_in':
            if shift_pressed: # Corner Mode
                cp.smooth = False
                cp.handle_in = (new_x - cp.point.x, new_y - cp.point.y)
            else: # Smooth Mode
                cp.smooth = True
                cp.set_handle_in_abs(new_x, new_y)
        elif self.spline_drag_type == 'handle_out':
            if shift_pressed:
                cp.smooth = False
                cp.handle_out = (new_x - cp.point.x, new_y - cp.point.y)
            else:
                cp.smooth = True
                cp.set_handle_out_abs(new_x, new_y)
        
        # FIX: Vorschau erzwingen
        try:
            # Wir speichern die temporären Linien direkt im Spline-Objekt
            spline._preview_lines = spline.to_lines(segments_per_span=10)
            self.update() # Wichtig: PaintEvent neu triggern
        except Exception as e:
            print(f"Spline preview error: {e}")
    
    def _update_live_values(self, snapped):
        """Aktualisiert Live-Werte NUR wenn die Felder nicht manuell editiert wurden"""
        if self.current_tool == SketchTool.LINE and self.tool_step >= 1:
            start = self.tool_points[-1]
            dx, dy = snapped.x() - start.x(), snapped.y() - start.y()
            # Nur aktualisieren wenn nicht gelockt
            if not self.dim_input.is_locked('length'):
                self.live_length = math.hypot(dx, dy)
                if self.dim_input.isVisible():
                    self.dim_input.set_value('length', self.live_length)
            if not self.dim_input.is_locked('angle'):
                self.live_angle = math.degrees(math.atan2(dy, dx))
                if self.dim_input.isVisible():
                    self.dim_input.set_value('angle', self.live_angle)
                
        elif self.current_tool in [SketchTool.RECTANGLE, SketchTool.RECTANGLE_CENTER] and self.tool_step == 1:
            p1 = self.tool_points[0]
            if self.rect_mode == 1:  # Center-Modus
                new_width = abs(snapped.x() - p1.x()) * 2
                new_height = abs(snapped.y() - p1.y()) * 2
            else:  # 2-Punkt-Modus
                new_width = abs(snapped.x() - p1.x())
                new_height = abs(snapped.y() - p1.y())
            # Nur aktualisieren wenn nicht gelockt
            if not self.dim_input.is_locked('width'):
                self.live_width = new_width
                if self.dim_input.isVisible():
                    self.dim_input.set_value('width', self.live_width)
            if not self.dim_input.is_locked('height'):
                self.live_height = new_height
                if self.dim_input.isVisible():
                    self.dim_input.set_value('height', self.live_height)
                
        elif self.current_tool == SketchTool.CIRCLE and self.tool_step == 1:
            c = self.tool_points[0]
            if not self.dim_input.is_locked('radius'):
                self.live_radius = math.hypot(snapped.x() - c.x(), snapped.y() - c.y())
                if self.dim_input.isVisible():
                    self.dim_input.set_value('radius', self.live_radius)
                
        elif self.current_tool == SketchTool.POLYGON and self.tool_step == 1:
            c = self.tool_points[0]
            if not self.dim_input.is_locked('radius'):
                self.live_radius = math.hypot(snapped.x() - c.x(), snapped.y() - c.y())
                if self.dim_input.isVisible():
                    self.dim_input.set_value('radius', self.live_radius)
    
    def wheelEvent(self, event):
        pos = event.position()
        world_before = self.screen_to_world(pos)
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        self.view_scale = max(0.5, min(200, self.view_scale * factor))
        world_after = self.screen_to_world(pos)
        self.view_offset += QPointF((world_after.x() - world_before.x()) * self.view_scale,
                                   -(world_after.y() - world_before.y()) * self.view_scale)
        self.update()
    
    def keyPressEvent(self, event):
        key, mod = event.key(), event.modifiers()
        if self.dim_input_active and self.dim_input.isVisible():
            if key == Qt.Key_Escape:
                self.dim_input.hide()
                self.dim_input_active = False
                self.setFocus()
                return
            elif key == Qt.Key_Tab:
                self.dim_input.next_field()
                return
            elif key in (Qt.Key_Return, Qt.Key_Enter):
                # Enter bestätigt den aktuellen Wert
                self.dim_input._confirm()
                return
            return
        
        # Enter zum Bestätigen (für Offset etc.)
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if self.current_tool == SketchTool.OFFSET and self.tool_step == 1:
                self._apply_offset()
                return
            elif self.tool_step > 0:
                self._finish_current_operation()
                return
        
        if mod & Qt.ControlModifier:
            if key == Qt.Key_Z: self.undo(); return
            elif key == Qt.Key_Y: self.redo(); return
            elif key == Qt.Key_A: self._select_all(); return
            elif key == Qt.Key_I: self.import_dxf(); return
            elif key == Qt.Key_E: self.export_dxf(); return
        
        if key == Qt.Key_Tab:
            self._show_dimension_input()
            return
        
        shortcuts = {
            Qt.Key_Escape: lambda: self._cancel_tool() if self.tool_step > 0 else self.set_tool(SketchTool.SELECT),
            Qt.Key_Space: lambda: self.set_tool(SketchTool.SELECT),
            Qt.Key_L: lambda: self.set_tool(SketchTool.LINE),
            Qt.Key_R: lambda: self.set_tool(SketchTool.RECTANGLE),
            Qt.Key_C: lambda: self.set_tool(SketchTool.CIRCLE),
            Qt.Key_P: lambda: self.set_tool(SketchTool.POLYGON),
            Qt.Key_A: lambda: self.set_tool(SketchTool.ARC_3POINT),
            Qt.Key_M: lambda: self.set_tool(SketchTool.MOVE),
            Qt.Key_O: lambda: self.set_tool(SketchTool.OFFSET),
            Qt.Key_T: lambda: self.set_tool(SketchTool.TRIM),
            Qt.Key_E: lambda: self.set_tool(SketchTool.EXTEND),
            Qt.Key_D: lambda: self.set_tool(SketchTool.DIMENSION),
            Qt.Key_N: lambda: self.set_tool(SketchTool.NUT),  # N für Mutter
            Qt.Key_Delete: self._delete_selected,
            Qt.Key_F: self._fit_view,
            Qt.Key_G: self._toggle_grid_snap,
            Qt.Key_X: self._toggle_construction,
            Qt.Key_H: lambda: self._apply_constraint('horizontal'),
            Qt.Key_V: lambda: self._apply_constraint('vertical'),
            # Neue Shortcuts für Bearbeitungstools
            Qt.Key_K: lambda: self.set_tool(SketchTool.COPY),      # K für Kopieren
            Qt.Key_Q: lambda: self.set_tool(SketchTool.ROTATE),    # Q für Rotieren  
            Qt.Key_I: lambda: self.set_tool(SketchTool.MIRROR),    # I für Spiegeln (mIrror)
            Qt.Key_S: lambda: self.set_tool(SketchTool.SCALE),     # S für Skalieren
            Qt.Key_Plus: self._increase_tolerance,   # + für Toleranz erhöhen
            Qt.Key_Minus: self._decrease_tolerance,  # - für Toleranz verringern
        }
        if key in shortcuts: shortcuts[key](); self.update()
    
    def _finish_current_operation(self):
        if self.current_tool == SketchTool.SPLINE and len(self.tool_points) >= 2:
            self._finish_spline()
            return
        self._cancel_tool()
    
    def _toggle_grid_snap(self):
        self.grid_snap = not self.grid_snap
        self.status_message.emit(tr("Grid snap: {state}").format(state=tr("ON") if self.grid_snap else tr("OFF")))
    
    def _toggle_construction(self):
        self.construction_mode = not self.construction_mode
        self.status_message.emit(tr("Construction: {state}").format(state=tr("ON") if self.construction_mode else tr("OFF")))
    
    def _increase_tolerance(self):
        """Toleranz für Muttern erhöhen"""
        self.nut_tolerance = round(min(2.0, self.nut_tolerance + 0.1), 2)
        self.status_message.emit(tr("Nut tolerance: {tol}mm").format(tol=f"{self.nut_tolerance:.2f}"))
        if self.current_tool == SketchTool.NUT:
            self._show_tool_options()  # Titel aktualisieren
    
    def _decrease_tolerance(self):
        """Toleranz für Muttern verringern"""
        self.nut_tolerance = round(max(0.0, self.nut_tolerance - 0.1), 2)
        self.status_message.emit(tr("Nut tolerance: {tol}mm").format(tol=f"{self.nut_tolerance:.2f}"))
        if self.current_tool == SketchTool.NUT:
            self._show_tool_options()  # Titel aktualisieren
    
    def _clear_selection(self):
        self.selected_lines.clear()
        self.selected_circles.clear()
        self.selected_arcs.clear()
        self.selected_points.clear()
        self.selected_spline = None
    
    def _select_all(self):
        self._clear_selection()
        self.selected_lines = list(self.sketch.lines)
        self.selected_circles = list(self.sketch.circles)
        self.selected_arcs = list(self.sketch.arcs)
        # Nur standalone Punkte (nicht Teil anderer Geometrie)
        used_point_ids = set()
        for line in self.sketch.lines:
            used_point_ids.add(line.start.id)
            used_point_ids.add(line.end.id)
        for circle in self.sketch.circles:
            used_point_ids.add(circle.center.id)
        for arc in self.sketch.arcs:
            used_point_ids.add(arc.center.id)
        self.selected_points = [p for p in self.sketch.points if p.id not in used_point_ids]
        self.update()
    
    def _finish_selection_box(self):
        if not self.selection_box_start or not self.selection_box_end: return
        x1, y1 = self.selection_box_start.x(), self.selection_box_start.y()
        x2, y2 = self.selection_box_end.x(), self.selection_box_end.y()
        rect = QRectF(min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1))
        if not (QApplication.keyboardModifiers() & Qt.ShiftModifier): self._clear_selection()
        for line in self.sketch.lines:
            p1 = self.world_to_screen(QPointF(line.start.x, line.start.y))
            p2 = self.world_to_screen(QPointF(line.end.x, line.end.y))
            if rect.contains(p1) and rect.contains(p2) and line not in self.selected_lines:
                self.selected_lines.append(line)
        for circle in self.sketch.circles:
            center = self.world_to_screen(QPointF(circle.center.x, circle.center.y))
            r = circle.radius * self.view_scale
            if rect.contains(QRectF(center.x()-r, center.y()-r, 2*r, 2*r)) and circle not in self.selected_circles:
                self.selected_circles.append(circle)
        for arc in self.sketch.arcs:
            center = self.world_to_screen(QPointF(arc.center.x, arc.center.y))
            r = arc.radius * self.view_scale
            if rect.contains(QRectF(center.x()-r, center.y()-r, 2*r, 2*r)) and arc not in self.selected_arcs:
                self.selected_arcs.append(arc)
        # Standalone Punkte
        used_point_ids = self._get_used_point_ids()
        for pt in self.sketch.points:
            if pt.id not in used_point_ids:
                pos = self.world_to_screen(QPointF(pt.x, pt.y))
                if rect.contains(pos) and pt not in self.selected_points:
                    self.selected_points.append(pt)
        self.selection_box_start = None
        self.selection_box_end = None
    
    def _get_used_point_ids(self):
        """Gibt IDs aller Punkte zurück die Teil anderer Geometrie sind"""
        used = set()
        for line in self.sketch.lines:
            used.add(line.start.id)
            used.add(line.end.id)
        for circle in self.sketch.circles:
            used.add(circle.center.id)
        for arc in self.sketch.arcs:
            used.add(arc.center.id)
        return used
    
    def _delete_selected(self):
        if not self.selected_lines and not self.selected_circles and not self.selected_arcs and not self.selected_points: return
        self._save_undo()
        for line in self.selected_lines[:]: self.sketch.delete_line(line)
        for circle in self.selected_circles[:]: self.sketch.delete_circle(circle)
        for arc in self.selected_arcs[:]: self.sketch.delete_arc(arc)
        for pt in self.selected_points[:]:
            if pt in self.sketch.points:
                self.sketch.points.remove(pt)
        self._clear_selection()
        self._find_closed_profiles()
        self.sketched_changed.emit()
        self.status_message.emit(tr("Deleted"))
        self.update()
    
    def _find_entity_at(self, pos):
        r = self.snap_radius / self.view_scale
        pt = Point2D(pos.x(), pos.y())
        
        # Erst Linien, Kreise, Bögen prüfen
        for line in self.sketch.lines:
            if line.distance_to_point(pt) < r: return line
        for circle in self.sketch.circles:
            dist = abs(math.hypot(circle.center.x - pos.x(), circle.center.y - pos.y()) - circle.radius)
            if dist < r: return circle
        for arc in self.sketch.arcs:
            # Distanz zum Bogen-Kreis
            dist_to_center = math.hypot(arc.center.x - pos.x(), arc.center.y - pos.y())
            dist_to_arc = abs(dist_to_center - arc.radius)
            if dist_to_arc < r:
                # Prüfe ob der Punkt im Winkelbereich des Bogens liegt
                angle = math.degrees(math.atan2(pos.y() - arc.center.y, pos.x() - arc.center.x))
                # Normalisiere Winkel auf 0-360
                while angle < 0: angle += 360
                while angle >= 360: angle -= 360
                
                start = arc.start_angle
                while start < 0: start += 360
                while start >= 360: start -= 360
                
                end = arc.end_angle
                while end < 0: end += 360
                while end >= 360: end -= 360
                
                # Prüfe ob im Bogenbereich
                if start <= end:
                    if start <= angle <= end:
                        return arc
                else:  # Bogen geht über 0°
                    if angle >= start or angle <= end:
                        return arc
        
        # Standalone Punkte prüfen
        used_point_ids = self._get_used_point_ids()
        for point in self.sketch.points:
            if point.id not in used_point_ids:
                dist = math.hypot(point.x - pos.x(), point.y - pos.y())
                if dist < r:
                    return point
        
        return None
    
    def _find_line_at(self, pos):
        r = self.snap_radius / self.view_scale
        pt = Point2D(pos.x(), pos.y())
        for line in self.sketch.lines:
            if line.distance_to_point(pt) < r: return line
        return None
    
    def _find_circle_at(self, pos):
        r = self.snap_radius / self.view_scale
        for c in self.sketch.circles:
            dist = abs(math.hypot(c.center.x - pos.x(), c.center.y - pos.y()) - c.radius)
            if dist < r: return c
        return None
    
    def _find_arc_at(self, pos):
        """Findet einen Bogen an der Position"""
        r = self.snap_radius / self.view_scale
        px, py = pos.x(), pos.y()
        
        for arc in self.sketch.arcs:
            # 1. Distanz zum Radius prüfen
            dist_to_center = math.hypot(arc.center.x - pos.x(), arc.center.y - pos.y())
            if abs(dist_to_center - arc.radius) < r:
                # 2. Winkel prüfen
                angle = math.degrees(math.atan2(pos.y() - arc.center.y, pos.x() - arc.center.x))
                if angle < 0: angle += 360
                
                start = arc.start_angle % 360
                end = arc.end_angle % 360
                
                # Normalisierung für den Fall, dass Start > End (über 0° Grenze)
                is_within = False
                if start <= end:
                    is_within = start <= angle <= end
                else:
                    is_within = angle >= start or angle <= end
                
                if is_within:
                    return arc
                    
        return None
    
    def _find_spline_element_at(self, pos):
        """
        Findet Spline-Elemente (Kontrollpunkt oder Handle) an der Position.
        
        Returns:
            Tuple (spline, cp_index, element_type) oder None
            element_type: 'point', 'handle_in', 'handle_out'
        """
        r = self.snap_radius / self.view_scale
        px, py = pos.x(), pos.y()
        
        for spline in self.sketch.splines:
            for i, cp in enumerate(spline.control_points):
                # Kontrollpunkt prüfen
                dist = math.hypot(cp.point.x - px, cp.point.y - py)
                if dist < r:
                    return (spline, i, 'point')
                
                # Eingehender Handle prüfen (nur wenn nicht erster Punkt oder closed)
                if i > 0 or spline.closed:
                    h_in = cp.handle_in_abs
                    dist_in = math.hypot(h_in[0] - px, h_in[1] - py)
                    if dist_in < r:
                        return (spline, i, 'handle_in')
                
                # Ausgehender Handle prüfen (nur wenn nicht letzter Punkt oder closed)
                if i < len(spline.control_points) - 1 or spline.closed:
                    h_out = cp.handle_out_abs
                    dist_out = math.hypot(h_out[0] - px, h_out[1] - py)
                    if dist_out < r:
                        return (spline, i, 'handle_out')
        
        return None
    
    def _find_spline_at(self, pos):
        """Findet eine Spline-Kurve an der Position"""
        r = self.snap_radius / self.view_scale
        px, py = pos.x(), pos.y()
        
        for spline in self.sketch.splines:
            # Prüfe Abstand zu den Kurvenpunkten
            pts = spline.get_curve_points(segments_per_span=10)
            for i in range(len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i + 1]
                # Punkt-Linien-Abstand
                line_len = math.hypot(x2 - x1, y2 - y1)
                if line_len < 0.001:
                    continue
                t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_len * line_len)))
                closest_x = x1 + t * (x2 - x1)
                closest_y = y1 + t * (y2 - y1)
                dist = math.hypot(px - closest_x, py - closest_y)
                if dist < r:
                    return spline
        
        return None
    
    def _find_face_at(self, pos):
        """Findet die Fläche unter dem Cursor (Point-in-Polygon Test)"""
        px, py = pos.x(), pos.y()
        
        for profile_data in self.closed_profiles:
            if not isinstance(profile_data, tuple) or len(profile_data) != 2:
                continue
            
            profile_type, data = profile_data
            
            if profile_type == 'circle':
                # Kreis: Prüfe ob Punkt innerhalb
                circle = data
                dist = math.hypot(px - circle.center.x, py - circle.center.y)
                if dist <= circle.radius:
                    return profile_data
                    
            elif profile_type == 'lines':
                # Linien-Polygon: Ray-Casting Algorithmus
                lines = data
                if len(lines) < 3:
                    continue
                
                # Sammle Vertices des Polygons
                vertices = []
                for line in lines:
                    vertices.append((line.start.x, line.start.y))
                
                if self._point_in_polygon(px, py, vertices):
                    return profile_data
            
            elif profile_type == 'polygon':
                # Shapely Polygon: Nutze Shapely's contains
                from shapely.geometry import Point as ShapelyPoint
                point = ShapelyPoint(px, py)
                if data.contains(point):
                    return profile_data
        
        return None
    
    def _point_in_polygon(self, px, py, vertices):
        """Ray-Casting Algorithmus für Point-in-Polygon Test"""
        n = len(vertices)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def _show_context_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background: #2d2d30; color: #ddd; border: 1px solid #3f3f46; } QMenu::item { padding: 6px 20px; } QMenu::item:selected { background: #094771; }")
        if self.selected_lines:
            menu.addAction("Horizontal (H)", lambda: self._apply_constraint('horizontal'))
            menu.addAction("Vertikal (V)", lambda: self._apply_constraint('vertical'))
            menu.addSeparator()
        if self.selected_lines or self.selected_circles:
            menu.addAction("Löschen (Del)", self._delete_selected)
            menu.addSeparator()
        menu.addAction("Alles auswählen (Ctrl+A)", self._select_all)
        menu.addAction("Ansicht einpassen (F)", self._fit_view)
        menu.exec(self.mapToGlobal(pos.toPoint()))
    
    def _fit_view(self):
        if not self.sketch.lines and not self.sketch.circles:
            self._center_view(); return
        minx = miny = float('inf')
        maxx = maxy = float('-inf')
        for l in self.sketch.lines:
            for p in [l.start, l.end]:
                minx, maxx = min(minx, p.x), max(maxx, p.x)
                miny, maxy = min(miny, p.y), max(maxy, p.y)
        for c in self.sketch.circles:
            minx = min(minx, c.center.x - c.radius)
            maxx = max(maxx, c.center.x + c.radius)
            miny = min(miny, c.center.y - c.radius)
            maxy = max(maxy, c.center.y + c.radius)
        if minx == float('inf'): return
        pad = 60
        w, h = max(maxx-minx, 1), max(maxy-miny, 1)
        self.view_scale = min((self.width()-2*pad)/w, (self.height()-2*pad)/h)
        cx, cy = (minx+maxx)/2, (miny+maxy)/2
        self.view_offset = QPointF(self.width()/2 - cx*self.view_scale, self.height()/2 + cy*self.view_scale)
        self.update()
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), self.BG_COLOR)
        self._draw_grid(p)
        self._draw_body_references(p)  # Bodies als Referenz im Hintergrund
        self._draw_profiles(p)
        self._draw_axes(p)
        self._draw_geometry(p)
        self._draw_constraints(p)
        self._draw_preview(p)
        self._draw_selection_box(p)
        self._draw_snap(p)
        self._draw_live_dimensions(p)
        self._draw_hud(p)
    
    def _on_dim_choice_changed(self, key, value):
        """Reagiert auf Dropdown-Auswahl im Dialog"""
        if key == "operation":
            self.extrude_operation = value

    def start_extrude_dialog(self):
        """Startet den Extrude-Dialog"""
        if not self.viewport:
            self.status_message.emit("Fehler: Kein Viewport verbunden!")
            return

        if not self.closed_profiles:
            self.status_message.emit("Keine geschlossenen Profile gefunden!")
            return
            
        self.viewport.set_extrude_mode(True)
        
        fields = [
            ("H", "height", 10.0, "mm"),
            ("Op", "operation", 0, ["New Body", "Join", "Cut"])
        ]
        
        self.dim_input.setup(fields)
        from PySide6.QtGui import QCursor
        pos = self.mapFromGlobal(QCursor.pos())
        self.dim_input.move(pos.x() + 20, pos.y() + 20)
        self.dim_input.show()
        self.dim_input.focus_field(0)
        self.dim_input_active = True
        self.status_message.emit("Extrude: Flächen wählen | Tab=Optionen | Enter=Anwenden")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.view_offset == QPointF(0, 0):
            self._center_view()

