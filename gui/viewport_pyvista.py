"""
LiteCAD - PyVista 3D Viewport
COMPLETE BUILD: FXAA, Matte Look, Robust Picking, No crashes
"""

import math
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from PySide6.QtWidgets import QWidget, QVBoxLayout, QFrame, QLabel, QToolButton
from PySide6.QtCore import Qt, Signal, QTimer, QEvent, QPoint
from PySide6.QtGui import QCursor

# ==================== IMPORTS ====================
HAS_PYVISTA = False
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    import vtk 
    HAS_PYVISTA = True
    print("✓ PyVista & VTK erfolgreich geladen.")
except ImportError as e:
    print(f"! PyVista Import-Fehler: {e}")

HAS_BUILD123D = False
try:
    import build123d
    HAS_BUILD123D = True
except ImportError:
    pass

HAS_SHAPELY = False
try:
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, unary_union, triangulate
    HAS_SHAPELY = True
except ImportError:
    pass


class OverlayHomeButton(QToolButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("🏠")
        self.setFixedSize(32, 32)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Standardansicht (Home)")
        self.setStyleSheet("""
            QToolButton {
                background-color: rgba(60, 60, 60, 180);
                color: #e0e0e0;
                border: 1px solid rgba(100, 100, 100, 100);
                border-radius: 4px;
                font-size: 16px;
            }
            QToolButton:hover {
                background-color: rgba(0, 120, 212, 230);
                border: 1px solid #0078d4;
                color: white;
            }
        """)

class PyVistaViewport(QWidget):
    view_changed = Signal()
    plane_clicked = Signal(str)
    custom_plane_clicked = Signal(tuple, tuple)
    extrude_requested = Signal(list, float, str)
    height_changed = Signal(float)
    face_selected = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._viewcube_created = False  # VOR _setup_plotter initialisieren
        
        # Dunkler Hintergrund für das Widget selbst
        self.setStyleSheet("background-color: #1e1e1e;")
        self.setAutoFillBackground(True)
        
        self._setup_ui()
        
        if HAS_PYVISTA:
            self._setup_plotter()
            self._setup_scene()
        
        # State
        self.sketches = []
        self.bodies = {} 
        self.detected_faces = []
        self.selected_faces = set()
        self.hovered_face = -1
        self.hovered_body_face = None  # (body_id, cell_id, normal, position)
        self.body_face_extrude = None  # Für Extrusion von Body-Flächen
        
        # Modes
        self.plane_select_mode = False
        self.extrude_mode = False
        self.extrude_height = 0.0
        self.is_dragging = False
        self.drag_start_pos = QPoint()
        self.drag_start_height = 0.0
        
        # Tracking
        self._sketch_actors = []
        self._body_actors = {}
        self._plane_actors = {}
        self._face_actors = []
        self._preview_actor = None
        self.last_highlighted_plane = None
        
        self.setFocusPolicy(Qt.StrongFocus)
    
    def _setup_ui(self):
        # Direktes Layout ohne zusätzlichen Frame
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        if not HAS_PYVISTA:
            self.main_layout.addWidget(QLabel("PyVista fehlt! Installiere: pip install pyvista pyvistaqt"))
            return
    
    def _setup_plotter(self):
        # QtInteractor direkt zum Widget-Layout hinzufügen (ohne extra Frame)
        self.plotter = QtInteractor(self)
        self.plotter.interactor.setStyleSheet("background-color: #262626;")
        self.main_layout.addWidget(self.plotter.interactor)
        
        self.plotter.interactor.setMouseTracking(True)
        self.plotter.interactor.installEventFilter(self)
        
        # Gradient Background - dunkler
        self.plotter.set_background('#1e1e1e', top='#2d2d30')
        self.plotter.enable_trackball_style()
        
        try: self.plotter.enable_anti_aliasing('fxaa')
        except: pass
        
        # WICHTIG: Entferne alle Standard-Widgets die PyVista automatisch erstellt
        try: self.plotter.hide_axes()
        except: pass
        
        # NUR das große Camera Orientation Widget (ViewCube mit klickbaren Kugeln)
        try:
            widget = self.plotter.add_camera_orientation_widget()
            if widget:
                # Versuch Labels zu setzen, aber stürzt nicht ab wenn es fehlschlägt
                try:
                    rep = widget.GetRepresentation()
                    if hasattr(rep, 'SetLabelText'):
                        rep.SetLabelText(0, "RECHTS")
                        rep.SetLabelText(1, "LINKS")
                        rep.SetLabelText(2, "HINTEN")
                        rep.SetLabelText(3, "VORNE")
                        rep.SetLabelText(4, "OBEN")
                        rep.SetLabelText(5, "UNTEN")
                except:
                    pass # Labels werden ignoriert, wenn nicht unterstützt
                self._cam_widget = widget
        except Exception as e:
            print(f"ViewCube creation warning: {e}")
        
        # Home-Button als Overlay
        self.btn_home = OverlayHomeButton(self)
        self.btn_home.clicked.connect(self._reset_camera_animated)
        self.btn_home.move(20, 20)
        self.btn_home.raise_()
        self.btn_home.show()
        
        self._viewcube_created = True
        
        try:
            if hasattr(self.plotter, 'iren') and self.plotter.iren:
                self.plotter.iren.AddObserver('EndInteractionEvent', lambda o,e: self.view_changed.emit())
        except: pass

    def _reset_camera_animated(self):
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.view_changed.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Home-Button immer oben links halten
        if hasattr(self, 'btn_home'):
            self.btn_home.move(20, 20)
            self.btn_home.raise_()  # Nach vorne bringen

    def _setup_scene(self):
        self._draw_grid(200)
        self._draw_axes(50)
        self.plotter.camera_position = 'iso'
        self.plotter.reset_camera()
    
    def _draw_grid(self, size=200, spacing=10):
        try: self.plotter.remove_actor('grid_main')
        except: pass
        n_lines = int(size/spacing)+1
        lines = []
        h = size/2
        for i in range(n_lines):
            p = -h + i*spacing
            lines.append(pv.Line((-h, p, 0), (h, p, 0)))
            lines.append(pv.Line((p, -h, 0), (p, h, 0)))
        if lines:
            grid = pv.MultiBlock(lines).combine()
            self.plotter.add_mesh(grid, color='#3a3a3a', line_width=1, name='grid_main', pickable=False)

    def _draw_axes(self, length=50):
        try:
            self.plotter.remove_actor('axis_x_org')
            self.plotter.remove_actor('axis_y_org')
            self.plotter.remove_actor('axis_z_org')
        except: pass
        self.plotter.add_mesh(pv.Line((0,0,0),(length,0,0)), color='#ff4444', line_width=3, name='axis_x_org')
        self.plotter.add_mesh(pv.Line((0,0,0),(0,length,0)), color='#44ff44', line_width=3, name='axis_y_org')
        self.plotter.add_mesh(pv.Line((0,0,0),(0,0,length)), color='#4444ff', line_width=3, name='axis_z_org')

    # ==================== EVENT FILTER ====================
    def eventFilter(self, obj, event):
        if not HAS_PYVISTA: return False
        
        if event.type() == QEvent.MouseMove:
            pos = event.position() if hasattr(event, 'position') else event.pos()
            x, y = int(pos.x()), int(pos.y())
            if self.plane_select_mode:
                self._highlight_plane_at_position(x, y)
                return False
            if self.is_dragging and self.extrude_mode:
                # Extrude-Drag mit linker Maustaste
                delta = self._calculate_extrude_delta(pos)
                new_height = self.drag_start_height + delta
                
                if abs(new_height - self.extrude_height) > 0.5:
                    self.show_extrude_preview(new_height)
                    self.height_changed.emit(self.extrude_height)
                return True  # Event konsumieren
            if self.extrude_mode and not self.is_dragging:
                # Hover-Highlight für Faces
                self._pick_face_at_position(x, y, hover_only=True)
                return False  # Kamera-Rotation erlauben!
                
        elif event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                pos = event.position() if hasattr(event, 'position') else event.pos()
                x, y = int(pos.x()), int(pos.y())
                
                if self.parent(): self.parent().setFocus()
                self.setFocus()
                
                if self.plane_select_mode: 
                    self._pick_plane_at_position(x, y)
                elif self.extrude_mode:
                    # Prüfe ob wir auf eine Face klicken
                    clicked_face = self._pick_face_at_position(x, y)
                    if clicked_face:
                        # Nur Drag starten wenn bereits Faces ausgewählt sind
                        if self.selected_faces:
                            self.is_dragging = True
                            self.drag_start_pos = pos
                            self.drag_start_height = self.extrude_height
                            self._cache_drag_direction()
                            return True  # Event konsumieren
                    # Wenn keine Face getroffen, Kamera-Interaktion erlauben
                    return False
                        
        elif event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton and self.is_dragging and self.extrude_mode:
                self.is_dragging = False
                return True
                
        return False
    
    def _cache_drag_direction(self):
        """Cached die Drag-Richtung basierend auf der ausgewählten Face-Normale"""
        self._drag_screen_dir = None
        if not self.selected_faces:
            return
        
        try:
            # Hole erste ausgewählte Face
            idx = list(self.selected_faces)[0]
            if idx >= len(self.detected_faces):
                return
            
            face = self.detected_faces[idx]
            normal = np.array(face.get('normal', (0, 0, 1)))
            
            # Hole Face-Zentrum
            if face.get('type') == 'body_face':
                center = np.array(face.get('center_3d', (0, 0, 0)))
            else:
                c2 = face['center_2d']
                center = np.array(self._transform_2d_to_3d(c2[0], c2[1], face['normal'], face['origin']))
            
            # Projiziere Normale auf Bildschirm
            cam_pos = np.array(self.plotter.camera_position[0])
            cam_focal = np.array(self.plotter.camera_position[1])
            cam_up = np.array(self.plotter.camera_position[2])
            
            # View-Richtung
            view_dir = cam_focal - cam_pos
            view_dir = view_dir / np.linalg.norm(view_dir)
            
            # Dot-Product: Wenn Normale zur Kamera zeigt, ist Drag nach oben = positiv
            dot = np.dot(normal, -view_dir)
            
            # Speichere Vorzeichen für Drag-Richtung
            self._drag_direction_sign = 1 if dot > 0 else -1
            
        except Exception as e:
            self._drag_direction_sign = 1
    
    def _calculate_extrude_delta(self, current_pos):
        """Berechnet die Extrusions-Höhenänderung basierend auf Mausbewegung"""
        dx = current_pos.x() - self.drag_start_pos.x()
        dy = self.drag_start_pos.y() - current_pos.y()  # Y invertiert
        
        # Kombiniere beide Achsen
        if abs(dy) > abs(dx):
            delta = dy * 0.3
        else:
            delta = dx * 0.3
        
        # Richtung INVERTIERT (war vorher * sign, jetzt * -sign)
        sign = getattr(self, '_drag_direction_sign', 1)
        return delta * (-sign)

    # ==================== PICKING ====================
    def _highlight_plane_at_position(self, x, y):
        if not self._plane_actors: return
        try:
            height = self.plotter.interactor.height()
            
            # Erst Standard-Ebenen prüfen
            picker = vtk.vtkPropPicker()
            picker.Pick(x, height-y, 0, self.plotter.renderer)
            actor = picker.GetActor()
            found = None
            if actor:
                for k, v in self._plane_actors.items():
                    if self.plotter.renderer.actors.get(v) == actor: found = k; break
            
            # Standard-Ebenen Highlight
            if found != self.last_highlighted_plane:
                self.last_highlighted_plane = found
                for k in ['xy','xz','yz']:
                    self._set_opacity(k, 0.7 if k == found else 0.25)
            
            # Body-Flächen Highlight
            if found is None and self.bodies:
                cell_picker = vtk.vtkCellPicker()
                cell_picker.SetTolerance(0.01)
                if cell_picker.Pick(x, height-y, 0, self.plotter.renderer):
                    cell_id = cell_picker.GetCellId()
                    if cell_id != -1:
                        pos = cell_picker.GetPickPosition()
                        normal = cell_picker.GetPickNormal()
                        new_hover = (tuple(pos), tuple(normal))
                        if getattr(self, '_plane_hover', None) != new_hover:
                            self._plane_hover = new_hover
                            self._draw_plane_hover_highlight(pos, normal)
                        return
            
            # Kein Body-Face getroffen - Highlight entfernen
            if getattr(self, '_plane_hover', None) is not None:
                self._plane_hover = None
                self._clear_plane_hover_highlight()
            
            self.plotter.render()
        except: pass
    
    def _draw_plane_hover_highlight(self, pos, normal):
        """Zeichnet Highlight für Body-Fläche im Plane-Select-Modus"""
        self._clear_plane_hover_highlight()
        try:
            center = np.array(pos)
            n = np.array(normal)
            n = n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else np.array([0,0,1])
            
            # Erstelle einen Kreis senkrecht zur Normalen
            radius = 8.0
            if abs(n[2]) < 0.9:
                u = np.cross(n, [0, 0, 1])
            else:
                u = np.cross(n, [1, 0, 0])
            u = u / np.linalg.norm(u)
            v = np.cross(n, u)
            
            points = []
            for i in range(33):
                angle = i * 2 * math.pi / 32
                p = center + radius * (math.cos(angle) * u + math.sin(angle) * v)
                points.append(p)
            
            pts = np.array(points)
            lines = pv.lines_from_points(pts)
            self.plotter.add_mesh(lines, color='lime', line_width=3, name='plane_hover')
            
            # Pfeil für Normale
            arrow = pv.Arrow(start=center, direction=n, scale=10)
            self.plotter.add_mesh(arrow, color='lime', name='plane_hover_arrow')
            
            self.plotter.update()
        except: pass
    
    def _clear_plane_hover_highlight(self):
        """Entfernt Plane-Hover-Highlight"""
        try:
            self.plotter.remove_actor('plane_hover')
        except: pass
        try:
            self.plotter.remove_actor('plane_hover_arrow')
        except: pass

    def _set_opacity(self, key, val):
        try: 
            self.plotter.renderer.actors.get(self._plane_actors[key]).GetProperty().SetOpacity(val)
        except: pass

    def _pick_plane_at_position(self, x, y):
        try:
            height = self.plotter.interactor.height()
            picker = vtk.vtkPropPicker()
            picker.Pick(x, height-y, 0, self.plotter.renderer)
            actor = picker.GetActor()
            
            if actor:
                for name, actor_name in self._plane_actors.items():
                    if self.plotter.renderer.actors.get(actor_name) == actor:
                        self.plane_clicked.emit(name)
                        return

            cell_picker = vtk.vtkCellPicker()
            cell_picker.SetTolerance(0.005)
            if cell_picker.Pick(x, height-y, 0, self.plotter.renderer):
                pos = cell_picker.GetPickPosition()
                normal = cell_picker.GetPickNormal()
                if pos and normal:
                    nx, ny, nz = normal
                    if abs(abs(nx)-1) < 0.05: normal = (1 if nx>0 else -1, 0, 0)
                    elif abs(abs(ny)-1) < 0.05: normal = (0, 1 if ny>0 else -1, 0)
                    elif abs(abs(nz)-1) < 0.05: normal = (0, 0, 1 if nz>0 else -1)
                    self.custom_plane_clicked.emit(tuple(pos), tuple(normal))
        except: pass

    def _pick_face_at_position(self, x, y, hover_only=False):
        if not self.detected_faces: 
            return False
        try:
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.01)  # Größere Toleranz für besseres Picking
            picker.Pick(x, self.plotter.interactor.height()-y, 0, self.plotter.renderer)
            pos = picker.GetPickPosition()
            if pos != (0.0,0.0,0.0):
                self._on_face_clicked(pos, hover_only)
                return not hover_only  # Bei Klick True, bei Hover False
            elif hover_only and self.hovered_face != -1:
                self.hovered_face = -1
                self._draw_selectable_faces()
        except Exception as e:
            print(f"Pick error: {e}")
        return False

    def _on_face_clicked(self, point, hover_only=False):
        best_dist = float('inf')
        best_idx = -1
        
        for i, face in enumerate(self.detected_faces):
            # Body-Face oder Sketch-Face?
            if face.get('type') == 'body_face':
                c3 = face.get('center_3d', (0, 0, 0))
            else:
                c2 = face['center_2d']
                c3 = self._transform_2d_to_3d(c2[0], c2[1], face['normal'], face['origin'])
            
            dist = math.sqrt(sum((point[k]-c3[k])**2 for k in range(3)))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        if best_idx >= 0 and best_dist < 200:
            if hover_only:
                # Nur neu zeichnen wenn sich hover geändert hat
                if self.hovered_face != best_idx:
                    self.hovered_face = best_idx
                    self._draw_selectable_faces()
            else:
                # Klick: Toggle Auswahl
                if best_idx in self.selected_faces: 
                    self.selected_faces.remove(best_idx)
                else: 
                    self.selected_faces.add(best_idx)
                self._draw_selectable_faces()
                self.face_selected.emit(best_idx)
        elif hover_only and self.hovered_face != -1:
            # Nichts getroffen - hover zurücksetzen
            self.hovered_face = -1
            self._draw_selectable_faces()

    def _transform_2d_to_3d(self, x, y, normal, origin):
        ox, oy, oz = origin
        if isinstance(normal, list): normal = tuple(normal)
        if normal == (0, 0, 1): return (ox + x, oy + y, oz)
        if normal == (0, 1, 0): return (ox + x, oy, oz - y)
        if normal == (1, 0, 0): return (ox, oy + x, oz - y)
        return (ox + x, oy + y, oz)

    def _apply_boolean_operation(self, new_mesh, operation, target_body_id=None):
        """Führt Boolean Ops durch: Union, Difference, New Body"""
        import vtk
        
        # 1. Neuer Körper
        if operation == "New Body" or not self.bodies or target_body_id is None:
            new_id = max(self.bodies.keys(), default=0) + 1
            self.add_body(new_id, f"Body {new_id}", new_mesh.points, [], color=None)
            # Hinweis: add_body erwartet verts/faces, aber wir können PyVista mesh direkt nutzen wenn wir add_body anpassen
            # Besserer Weg hier: Direkt mesh speichern
            self._add_mesh_as_body(new_id, new_mesh)
            return

        # 2. Bestehenden Körper holen
        # Wir nehmen vereinfacht an, es gibt einen 'aktiven' Körper oder den zuletzt erstellten
        target_id = target_body_id if target_body_id else list(self.bodies.keys())[-1]
        target_mesh = self.bodies[target_id]['mesh']

        try:
            # PyVista Boolean Operations (benötigt triangulierte, saubere Meshes)
            # Sicherstellen, dass alles Dreiecke sind
            if not target_mesh.is_all_triangles: target_mesh = target_mesh.triangulate()
            if not new_mesh.is_all_triangles: new_mesh = new_mesh.triangulate()
            
            result_mesh = None
            
            if operation == "Join":
                result_mesh = target_mesh.boolean_union(new_mesh)
            elif operation == "Cut":
                result_mesh = target_mesh.boolean_difference(new_mesh)
            elif operation == "Intersect":
                result_mesh = target_mesh.boolean_intersection(new_mesh)
                
            if result_mesh and result_mesh.n_points > 0:
                # Körper aktualisieren
                self._add_mesh_as_body(target_id, result_mesh, color=self.bodies[target_id]['color'])
                print(f"Boolean {operation} success.")
            else:
                print("Boolean Operation failed (empty result).")
                # Fallback: Als neuen Körper hinzufügen, damit Arbeit nicht verloren geht
                self._apply_boolean_operation(new_mesh, "New Body")

        except Exception as e:
            print(f"Boolean Error: {e}")
            self._apply_boolean_operation(new_mesh, "New Body")
            
    def _add_mesh_as_body(self, bid, mesh, color=None):
        """Interne Hilfsfunktion um ein PyVista Mesh direkt als Body zu speichern"""
        # Alten Actor entfernen
        if bid in self._body_actors:
            for n in self._body_actors[bid]: 
                try: self.plotter.remove_actor(n)
                except: pass
        
        mesh = mesh.clean() # Wichtig für Rendering
        mesh.compute_normals(inplace=True)
        
        col = color or "lightblue"
        n1 = f"body_{bid}_m"
        # PBR (Physically Based Rendering) für Metall-Look
        self.plotter.add_mesh(mesh, color=col, name=n1, show_edges=False, 
                              smooth_shading=True, pbr=True, metallic=0.5, roughness=0.4)
        
        n2 = f"body_{bid}_e"
        edges = mesh.extract_feature_edges(feature_angle=45)
        self.plotter.add_mesh(edges, color="black", line_width=2, name=n2)
        
        self._body_actors[bid] = (n1, n2)
        self.bodies[bid] = {'mesh': mesh, 'color': col}
        self.plotter.render()

    # ==================== PUBLIC API ====================
    def set_plane_select_mode(self, enabled):
        self.plane_select_mode = enabled
        self.last_highlighted_plane = None
        self._plane_hover = None
        if enabled: 
            self._show_selection_planes()
            self.plotter.render()
        else: 
            self._hide_selection_planes()
            self._clear_plane_hover_highlight()

    def set_extrude_mode(self, enabled):
        self.extrude_mode = enabled
        self.selected_faces.clear()
        self.hovered_face = -1
        self.hovered_body_face = None
        self.body_face_extrude = None
        self.is_dragging = False; self.extrude_height = 0.0
        if enabled: 
            self._detect_faces()          # Sketch-Faces
            self._detect_body_faces()     # Body-Faces hinzufügen!
            self._draw_selectable_faces()
            self._clear_body_face_highlight()
        else: 
            self._clear_face_actors(); self._clear_preview()
            self._restore_body_colors()   # Farben wiederherstellen
            self._clear_body_face_highlight()
            try:
                self.plotter.remove_actor('body_face_selection')
            except: pass
    
    def set_edge_select_mode(self, enabled):
        """Aktiviert/deaktiviert den Edge-Selection-Modus für Fillet/Chamfer"""
        self.edge_select_mode = getattr(self, 'edge_select_mode', False)
        self.edge_select_mode = enabled
        self.selected_edges = getattr(self, 'selected_edges', set())
        
        if enabled:
            # Highlight alle Kanten
            self._highlight_all_edges()
        else:
            # Entferne Edge-Highlights
            self._clear_edge_highlights()
    
    def _highlight_all_edges(self):
        """Zeigt alle Kanten der Bodies als selektierbar an"""
        try:
            for bid, body_data in self.bodies.items():
                mesh = body_data.get('mesh')
                if mesh:
                    edges = mesh.extract_feature_edges(feature_angle=30)
                    name = f"edges_highlight_{bid}"
                    self.plotter.add_mesh(edges, color='yellow', line_width=2, 
                                         name=name, opacity=0.7)
            self.plotter.update()
        except Exception as e:
            print(f"Edge highlight error: {e}")
    
    def _clear_edge_highlights(self):
        """Entfernt alle Edge-Highlights"""
        try:
            for bid in self.bodies.keys():
                name = f"edges_highlight_{bid}"
                try:
                    self.plotter.remove_actor(name)
                except:
                    pass
            self.plotter.update()
        except:
            pass
    
    def _restore_body_colors(self):
        """Stellt Original-Farben aller Bodies wieder her"""
        for bid, (mesh_name, edge_name) in self._body_actors.items():
            try:
                if mesh_name in self.plotter.renderer.actors:
                    actor = self.plotter.renderer.actors[mesh_name]
                    prop = actor.GetProperty()
                    if bid in self.bodies:
                        col = self.bodies[bid].get('color', (0.6, 0.6, 0.8))
                        prop.SetColor(*col)
                        prop.SetOpacity(1.0)
            except:
                pass
        self.plotter.render()
    
    def _detect_body_faces(self):
        """Erkennt planare Flächen von 3D-Bodies und fügt sie zu detected_faces hinzu"""
        if not self.bodies:
            return
            
        for bid, body_data in self.bodies.items():
            mesh = body_data.get('mesh')
            if mesh is None:
                continue
            
            try:
                # Normalen berechnen falls nicht vorhanden
                if 'Normals' not in mesh.cell_data:
                    mesh.compute_normals(cell_normals=True, inplace=True)
                
                cell_normals = mesh.cell_data.get('Normals', [])
                if len(cell_normals) == 0:
                    continue
                
                # Gruppiere Zellen nach Normale (für planare Flächen)
                face_groups = {}  # normal_key -> list of cell_ids
                
                for cell_id in range(mesh.n_cells):
                    normal = cell_normals[cell_id]
                    # Runde Normale für Gruppierung
                    nkey = (round(normal[0], 2), round(normal[1], 2), round(normal[2], 2))
                    if nkey not in face_groups:
                        face_groups[nkey] = []
                    face_groups[nkey].append(cell_id)
                
                # Für jede Gruppe eine selektierbare Fläche erstellen
                for normal_key, cell_ids in face_groups.items():
                    # Berechne Zentrum und Bounds der Fläche
                    centers = []
                    all_points = []
                    
                    for cid in cell_ids:
                        cell = mesh.get_cell(cid)
                        pts = cell.points
                        all_points.extend(pts)
                        centers.append(np.mean(pts, axis=0))
                    
                    if not centers:
                        continue
                    
                    center_3d = np.mean(centers, axis=0)
                    all_points = np.array(all_points)
                    
                    # Fläche als detected_face hinzufügen
                    self.detected_faces.append({
                        'type': 'body_face',  # Markierung für Body-Face
                        'body_id': bid,
                        'cell_ids': cell_ids,
                        'normal': tuple(normal_key),
                        'center_3d': tuple(center_3d),
                        'center_2d': (center_3d[0], center_3d[1]),  # Für Kompatibilität
                        'origin': tuple(center_3d),
                        'points_3d': all_points,
                        'mesh': mesh
                    })
                    
            except Exception as e:
                print(f"Body face detection error for body {bid}: {e}")

    def set_sketches(self, sketches):
        self.sketches = list(sketches)
        if not HAS_PYVISTA: return
        for n in self._sketch_actors:
            try: self.plotter.remove_actor(n)
            except: pass
        self._sketch_actors.clear()
        for s, v in self.sketches:
            if v: self._render_sketch(s)
        self.plotter.update()

    def add_body(self, bid, name, verts, faces, color=None):
        if not HAS_PYVISTA: return
        if bid in self._body_actors:
            for n in self._body_actors[bid]: 
                try: self.plotter.remove_actor(n)
                except: pass
        
        v = np.array(verts, dtype=np.float32)
        f = []
        for face in faces: f.extend([len(face)] + list(face))
        
        try:
            mesh = pv.PolyData(v, np.array(f, dtype=np.int32))
            mesh = mesh.clean()
            mesh.compute_normals(cell_normals=True, point_normals=True, split_vertices=True, feature_angle=30, inplace=True)
            
            # Farbe als RGB-Tuple speichern
            if color is None:
                col_rgb = (0.6, 0.6, 0.8)
            elif isinstance(color, str):
                # String zu RGB konvertieren
                color_map = {'lightblue': (0.6, 0.6, 0.8), 'red': (0.8, 0.4, 0.4), 'green': (0.4, 0.8, 0.4)}
                col_rgb = color_map.get(color, (0.6, 0.6, 0.8))
            else:
                col_rgb = tuple(color)
            
            n1 = f"body_{bid}_m"
            self.plotter.add_mesh(mesh, color=col_rgb, name=n1, show_edges=False, 
                                  smooth_shading=True, specular=0.0, diffuse=1.0, ambient=0.15,
                                  pickable=True)
            n2 = f"body_{bid}_e"
            edges = mesh.extract_feature_edges(45)
            self.plotter.add_mesh(edges, color="black", line_width=2, name=n2, pickable=False)
            
            self._body_actors[bid] = (n1, n2)
            self.bodies[bid] = {'mesh': mesh, 'color': col_rgb}  # RGB-Tuple!
            self.plotter.update()
        except: pass

    def set_body_visibility(self, body_id, visible):
        if body_id not in self._body_actors: return
        try:
            m, e = self._body_actors[body_id]
            self.plotter.renderer.actors[m].SetVisibility(visible)
            self.plotter.renderer.actors[e].SetVisibility(visible)
            self.plotter.render()
        except: pass
    
    def set_all_bodies_visible(self, visible):
        """Setzt alle Bodies sichtbar/unsichtbar"""
        for body_id in self._body_actors:
            try:
                m, e = self._body_actors[body_id]
                self.plotter.renderer.actors[m].SetVisibility(visible)
                self.plotter.renderer.actors[e].SetVisibility(visible)
            except: pass
        self.plotter.render()

    def clear_bodies(self):
        for names in self._body_actors.values():
            for n in names: 
                try: self.plotter.remove_actor(n)
                except: pass
        self._body_actors.clear(); self.bodies.clear(); self.plotter.render()

    def get_body_mesh(self, body_id):
        if body_id in self.bodies: return self.bodies[body_id]['mesh']
        return None

    def get_selected_faces(self):
        return [self.detected_faces[i] for i in self.selected_faces if i < len(self.detected_faces)]

    def get_extrusion_data(self, face_idx, height):
        if face_idx < 0 or face_idx >= len(self.detected_faces): 
            return [], []
        
        face = self.detected_faces[face_idx]
        
        # Body-Face oder Sketch-Face?
        if face.get('type') == 'body_face':
            return self._calculate_body_face_extrusion(face, height)
        else:
            return self._calculate_extrusion_geometry(face, height)

    def _calculate_extrusion_geometry(self, face, height):
        if not HAS_SHAPELY: return [], []
        poly = face.get('shapely_poly')
        if not poly: return [], []
        normal = face['normal']; origin = face['origin']
        try:
            tris = triangulate(poly)
            valid_tris = [t for t in tris if poly.contains(t.centroid)]
        except: return [], []
        verts = []; faces = []; v_map = {}
        def add(x,y,z):
            k = (round(x,4), round(y,4), round(z,4))
            if k in v_map: return v_map[k]
            idx = len(verts); verts.append((x,y,z)); v_map[k] = idx; return idx
        def trans(x,y,h): return self._transform_2d_to_3d(x,y,normal,origin)
        def trans_h(x,y,h):
            p = self._transform_2d_to_3d(x,y,normal,origin)
            nx, ny, nz = normal
            return (p[0]+nx*h, p[1]+ny*h, p[2]+nz*h)

        for t in valid_tris:
            xx, yy = t.exterior.coords.xy
            p = list(zip(xx[:3], yy[:3]))
            b = [add(*trans(x,y,0)) for x,y in p]
            t_Idx = [add(*trans_h(x,y,height)) for x,y in p]
            if height > 0:
                faces.append((b[0], b[2], b[1])); faces.append((t_Idx[0], t_Idx[1], t_Idx[2]))
            else:
                faces.append((b[0], b[1], b[2])); faces.append((t_Idx[0], t_Idx[2], t_Idx[1]))

        rings = [poly.exterior] + list(poly.interiors)
        for ring in rings:
            pts = list(ring.coords)
            for i in range(len(pts)-1):
                x1,y1 = pts[i]; x2,y2 = pts[i+1]
                b1 = add(*trans(x1,y1,0)); b2 = add(*trans(x2,y2,0))
                t1 = add(*trans_h(x1,y1,height)); t2 = add(*trans_h(x2,y2,height))
                if height > 0:
                    faces.append((b1, b2, t2)); faces.append((b1, t2, t1))
                else:
                    faces.append((b1, t1, t2)); faces.append((b1, t2, b2))
        return verts, faces

    def _detect_faces(self):
        """
        Erkennt Flächen aus Sketches für 3D-Extrusion.
        FIX: Unterstützt Splines, Slots und erkennt Löcher automatisch durch einheitliche Polygonisierung.
        """
        self.detected_faces = []
        if not HAS_SHAPELY: return
        
        from shapely.ops import polygonize, unary_union
        from shapely.geometry import LineString, Polygon
        
        # Hilfsfunktion zum Runden (WICHTIG für Slots!)
        def rnd(val):
            return round(val, 5)
            
        for s, vis in self.sketches:
            if not vis: continue
            norm = tuple(getattr(s, 'plane_normal', (0,0,1)))
            orig = getattr(s, 'plane_origin', (0,0,0))
            
            # Wir sammeln ALLES als Liniensegmente für maximale Robustheit
            all_segments = []
            
            # 1. LINJEN
            for l in getattr(s, 'lines', []):
                if not getattr(l, 'construction', False):
                    all_segments.append(LineString([
                        (rnd(l.start.x), rnd(l.start.y)), 
                        (rnd(l.end.x), rnd(l.end.y))
                    ]))
            
            # 2. ARCS (Bögen)
            for arc in getattr(s, 'arcs', []):
                if not getattr(arc, 'construction', False):
                    pts = []
                    start = arc.start_angle
                    end = arc.end_angle
                    sweep = end - start
                    if sweep < 0.1: sweep += 360
                    
                    # Feinere Auflösung für 3D
                    steps = max(12, int(sweep / 5))
                    
                    for i in range(steps + 1):
                        t = math.radians(start + sweep * (i / steps))
                        x = arc.center.x + arc.radius * math.cos(t)
                        y = arc.center.y + arc.radius * math.sin(t)
                        pts.append((rnd(x), rnd(y)))
                    
                    if len(pts) >= 2:
                        all_segments.append(LineString(pts))

            # 3. SPLINES (Neu!)
            for spline in getattr(s, 'splines', []):
                if not getattr(spline, 'construction', False):
                    # Punkte aus der Spline holen
                    pts_raw = []
                    # Versuche verschiedene Methoden, um Punkte zu bekommen
                    if hasattr(spline, 'get_curve_points'):
                         pts_raw = spline.get_curve_points(segments_per_span=16)
                    elif hasattr(spline, 'to_lines'):
                         lines = spline.to_lines(segments_per_span=16)
                         if lines:
                             pts_raw.append((lines[0].start.x, lines[0].start.y))
                             for ln in lines:
                                 pts_raw.append((ln.end.x, ln.end.y))
                    
                    # Koordinaten runden
                    pts = [(rnd(p[0]), rnd(p[1])) for p in pts_raw]
                    if len(pts) >= 2:
                        all_segments.append(LineString(pts))

            # 4. KREISE (Auch als Linien behandeln für "Loch-in-Fläche" Logik)
            for c in getattr(s, 'circles', []):
                if not getattr(c, 'construction', False):
                    pts = []
                    for i in range(65): # Geschlossener Loop
                        angle = i * 2 * math.pi / 64
                        x = c.center.x + c.radius * math.cos(angle)
                        y = c.center.y + c.radius * math.sin(angle)
                        pts.append((rnd(x), rnd(y)))
                    all_segments.append(LineString(pts))

            # 5. Polygonize: Findet Flächen und Löcher automatisch
            if all_segments:
                try:
                    merged = unary_union(all_segments)
                    for poly in polygonize(merged):
                        if poly.is_valid and poly.area > 0.01:
                            self.detected_faces.append({
                                'shapely_poly': poly,
                                'coords': list(poly.exterior.coords),
                                'normal': norm,
                                'origin': orig,
                                'sketch': s,
                                'center_2d': (poly.centroid.x, poly.centroid.y)
                            })
                except Exception as e:
                    print(f"3D Face Detection Error: {e}")
        
        # Sortieren: Kleine Flächen zuerst (besser klickbar, falls in großen Flächen liegend)
        self.detected_faces.sort(key=lambda x: x['shapely_poly'].area)
    
    def _compute_atomic_regions_from_polygons(self, polygons):
        """
        Berechnet alle atomaren (nicht weiter teilbaren) Regionen 
        aus einer Liste von überlappenden Polygonen.
        
        Beispiel: 4 überlappende Kreise → 9 atomare Regionen
        """
        if not polygons:
            return []
        if len(polygons) == 1:
            return polygons
        
        from shapely.geometry import Polygon, MultiPolygon
        
        n = len(polygons)
        
        # Für jede mögliche Kombination von Polygonen berechnen wir die Region,
        # die IN allen Polygonen der Kombination liegt, aber NICHT in anderen.
        
        # Effizienterer Ansatz: Iterativ aufbauen
        # Start mit dem ersten Polygon, dann mit jedem weiteren schneiden/subtrahieren
        
        atomic = []
        
        # Methode: Für jedes Polygon berechne die Teile die:
        # - nur in diesem Polygon sind
        # - in diesem UND anderen sind (Schnittmengen)
        
        # Wir verwenden einen "overlay" Ansatz
        all_regions = []
        
        # Initialisiere mit dem ersten Polygon
        current_regions = [{'poly': polygons[0], 'inside': {0}}]
        
        for i in range(1, n):
            new_poly = polygons[i]
            next_regions = []
            
            for region in current_regions:
                poly = region['poly']
                inside_set = region['inside']
                
                if not poly.is_valid or poly.is_empty:
                    continue
                
                try:
                    # Teil der Region der AUCH im neuen Polygon ist
                    intersection = poly.intersection(new_poly)
                    # Teil der Region der NICHT im neuen Polygon ist
                    difference = poly.difference(new_poly)
                    
                    # Intersection hinzufügen (ist in allen bisherigen + dem neuen)
                    if intersection.is_valid and not intersection.is_empty:
                        for p in self._extract_polygons(intersection):
                            if p.area > 0.01:
                                next_regions.append({
                                    'poly': p, 
                                    'inside': inside_set | {i}
                                })
                    
                    # Difference hinzufügen (ist nur in den bisherigen)
                    if difference.is_valid and not difference.is_empty:
                        for p in self._extract_polygons(difference):
                            if p.area > 0.01:
                                next_regions.append({
                                    'poly': p, 
                                    'inside': inside_set.copy()
                                })
                except Exception as e:
                    # Bei Fehler: Region beibehalten
                    next_regions.append(region)
            
            # Auch den Teil des neuen Polygons der in KEINER bisherigen Region war
            try:
                remaining = new_poly
                for region in current_regions:
                    if region['poly'].is_valid:
                        remaining = remaining.difference(region['poly'])
                
                if remaining.is_valid and not remaining.is_empty:
                    for p in self._extract_polygons(remaining):
                        if p.area > 0.01:
                            next_regions.append({
                                'poly': p, 
                                'inside': {i}
                            })
            except:
                pass
            
            current_regions = next_regions
        
        # Extrahiere nur die Polygone
        result = [r['poly'] for r in current_regions if r['poly'].is_valid and r['poly'].area > 0.01]
        
        # Deduplizierung
        final = []
        for poly in result:
            is_dup = False
            for existing in final:
                try:
                    # Wenn fast identisch (symmetrische Differenz sehr klein)
                    sym_diff = poly.symmetric_difference(existing)
                    if sym_diff.area < min(poly.area, existing.area) * 0.05:
                        is_dup = True
                        break
                except:
                    pass
            if not is_dup:
                final.append(poly)
        
        return final
    
    def _extract_polygons(self, geom):
        """Extrahiert alle Polygone aus einer Shapely Geometrie"""
        from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
        
        if geom.is_empty:
            return []
        if geom.geom_type == 'Polygon':
            return [geom]
        elif geom.geom_type == 'MultiPolygon':
            return list(geom.geoms)
        elif geom.geom_type == 'GeometryCollection':
            result = []
            for g in geom.geoms:
                result.extend(self._extract_polygons(g))
            return result
        return []

    def show_extrude_preview(self, height):
        self._clear_preview(); self.extrude_height = height
        
        if not self.selected_faces or abs(height) < 0.1: 
            return
        
        all_verts = []
        all_faces = []
        vert_offset = 0
        
        for idx in self.selected_faces:
            if idx < 0 or idx >= len(self.detected_faces): 
                continue
            
            face = self.detected_faces[idx]
            
            # Body-Face oder Sketch-Face?
            if face.get('type') == 'body_face':
                v, f = self._calculate_body_face_extrusion(face, height)
            else:
                v, f = self.get_extrusion_data(idx, height)
            
            if v and f:
                # Offset für Faces anpassen
                for fv in f:
                    all_faces.append(tuple(vi + vert_offset for vi in fv))
                all_verts.extend(v)
                vert_offset = len(all_verts)
        
        if all_verts:
            vv = np.array(all_verts, dtype=np.float32)
            ff = []
            for face in all_faces: 
                ff.extend([3] + list(face))
            mesh = pv.PolyData(vv, np.array(ff, dtype=np.int32))
            self._preview_actor = 'prev'
            col = '#6699ff' if height > 0 else '#ff6666'  # Blau für Add, Rot für Cut
            self.plotter.add_mesh(mesh, color=col, opacity=0.8, name='prev')
        
        self.plotter.update()
    
    def _calculate_body_face_extrusion(self, face, height):
        """Berechnet vollständige Extrusion für eine Body-Fläche inkl. Seitenwände"""
        try:
            mesh = face.get('mesh')
            cell_ids = face.get('cell_ids', [])
            normal = np.array(face.get('normal', (0,0,1)))
            
            if mesh is None or not cell_ids:
                return [], []
            
            # Normalisiere
            norm_len = np.linalg.norm(normal)
            n = normal / norm_len if norm_len > 0 else np.array([0,0,1])
            
            # Extrahiere Fläche
            face_mesh = mesh.extract_cells(cell_ids)
            
            # Finde Randkanten der Fläche
            edges = face_mesh.extract_feature_edges(
                boundary_edges=True, 
                feature_edges=False, 
                manifold_edges=False
            )
            
            # Wenn keine Kanten gefunden, nutze alle Kanten
            if edges.n_points == 0:
                edges = face_mesh.extract_all_edges()
            
            # Sammle Randpunkte
            edge_points = edges.points if edges.n_points > 0 else face_mesh.points
            
            verts = []
            faces = []
            v_map = {}
            
            def add_vert(p):
                """Fügt Vertex hinzu, vermeidet Duplikate"""
                key = (round(p[0], 4), round(p[1], 4), round(p[2], 4))
                if key in v_map:
                    return v_map[key]
                idx = len(verts)
                verts.append(tuple(p))
                v_map[key] = idx
                return idx
            
            # Bottom-Fläche (Original-Geometrie)
            for i in range(face_mesh.n_cells):
                cell = face_mesh.get_cell(i)
                pts = cell.points
                if len(pts) >= 3:
                    indices = [add_vert(p) for p in pts]
                    if height > 0:
                        faces.append(tuple(reversed(indices)))  # Normale nach unten
                    else:
                        faces.append(tuple(indices))
            
            # Top-Fläche (extrudiert)
            for i in range(face_mesh.n_cells):
                cell = face_mesh.get_cell(i)
                pts = cell.points
                if len(pts) >= 3:
                    top_pts = [p + n * height for p in pts]
                    indices = [add_vert(p) for p in top_pts]
                    if height > 0:
                        faces.append(tuple(indices))  # Normale nach oben
                    else:
                        faces.append(tuple(reversed(indices)))
            
            # Seitenwände aus Randkanten
            if edges.n_lines > 0:
                lines = edges.lines
                i = 0
                while i < len(lines):
                    n_pts_line = lines[i]
                    if n_pts_line >= 2:
                        for j in range(n_pts_line - 1):
                            p1 = edge_points[lines[i + 1 + j]]
                            p2 = edge_points[lines[i + 2 + j]]
                            
                            # 4 Punkte für Quad: bottom1, bottom2, top2, top1
                            b1 = add_vert(p1)
                            b2 = add_vert(p2)
                            t1 = add_vert(p1 + n * height)
                            t2 = add_vert(p2 + n * height)
                            
                            # Zwei Dreiecke für das Quad
                            if height > 0:
                                faces.append((b1, b2, t2))
                                faces.append((b1, t2, t1))
                            else:
                                faces.append((b1, t1, t2))
                                faces.append((b1, t2, b2))
                    i += n_pts_line + 1
            else:
                # Fallback: Nutze alle Kanten des Meshes
                for i in range(face_mesh.n_cells):
                    cell = face_mesh.get_cell(i)
                    pts = cell.points
                    n_cell_pts = len(pts)
                    for j in range(n_cell_pts):
                        p1 = pts[j]
                        p2 = pts[(j + 1) % n_cell_pts]
                        
                        b1 = add_vert(p1)
                        b2 = add_vert(p2)
                        t1 = add_vert(p1 + n * height)
                        t2 = add_vert(p2 + n * height)
                        
                        if height > 0:
                            faces.append((b1, b2, t2))
                            faces.append((b1, t2, t1))
                        else:
                            faces.append((b1, t1, t2))
                            faces.append((b1, t2, b2))
            
            return verts, faces
            
        except Exception as e:
            print(f"Body face extrusion error: {e}")
            import traceback
            traceback.print_exc()
            return [], []

    def confirm_extrusion(self, operation="New Body"):
            """Bestätigt Extrusion und sendet Daten + Operation an Main Window"""
            # Nur senden, wenn Flächen ausgewählt sind und die Höhe nicht 0 ist
            if self.selected_faces and abs(self.extrude_height) >= 0.1:
                # WICHTIG: Hier senden wir die Operation (String) mit!
                # Das Signal muss oben definiert sein als: Signal(list, float, str)
                self.extrude_requested.emit(list(self.selected_faces), self.extrude_height, operation)
                
            # Modus beenden
            self.set_extrude_mode(False)

    def _pick_body_face(self, x, y):
        """Versucht eine planare Fläche auf einem 3D-Körper zu finden"""
        cell_picker = vtk.vtkCellPicker()
        cell_picker.Pick(x, self.plotter.interactor.height()-y, 0, self.plotter.renderer)
        
        if cell_picker.GetCellId() != -1:
            actor = cell_picker.GetActor()
            # Finde Body ID zu Actor
            body_id = None
            for bid, actors in self._body_actors.items():
                if self.plotter.renderer.actors.get(actors[0]) == actor: # Check main mesh
                    body_id = bid
                    break
            
            if body_id is not None:
                # Normalenvektor der geklickten Zelle
                normal = cell_picker.GetPickNormal()
                pos = cell_picker.GetPickPosition()
                
                print(f"Body {body_id} clicked at {pos}, Normal: {normal}")
                # Hier könnte man nun eine neue Skizze auf dieser Fläche starten
                # Signal emitten:
                self.custom_plane_clicked.emit(tuple(pos), tuple(normal))
                return True
        return False
    
    def _hover_body_face(self, x, y):
        """Hebt Body-Flächen beim Hover hervor (Fusion360-Style)"""
        if not self.bodies:
            return
            
        try:
            cell_picker = vtk.vtkCellPicker()
            cell_picker.SetTolerance(0.01)
            height = self.plotter.interactor.height()
            
            picked = cell_picker.Pick(x, height - y, 0, self.plotter.renderer)
            cell_id = cell_picker.GetCellId()
            
            if picked and cell_id != -1:
                actor = cell_picker.GetActor()
                if actor is None:
                    return
                
                # Finde Body ID über Actor-Namen
                body_id = None
                for bid, (mesh_name, edge_name) in self._body_actors.items():
                    # Hole den Actor über den Namen
                    if mesh_name in self.plotter.renderer.actors:
                        body_actor = self.plotter.renderer.actors[mesh_name]
                        if body_actor is actor:
                            body_id = bid
                            break
                
                if body_id is not None:
                    normal = cell_picker.GetPickNormal()
                    pos = cell_picker.GetPickPosition()
                    
                    # Speichere gehovertes Face für Click
                    new_hover = (body_id, cell_id, tuple(normal), tuple(pos))
                    if self.hovered_body_face != new_hover:
                        self.hovered_body_face = new_hover
                        self._draw_body_face_highlight(pos, normal)
                    return
            
            # Nichts getroffen - Highlight entfernen
            if self.hovered_body_face is not None:
                self.hovered_body_face = None
                self._clear_body_face_highlight()
                
        except Exception as e:
            print(f"Hover error: {e}")
    
    def _draw_body_face_highlight(self, pos, normal):
        """Zeichnet Highlight auf gehoverter Body-Fläche"""
        self._clear_body_face_highlight()
        try:
            center = np.array(pos)
            n = np.array(normal)
            norm_len = np.linalg.norm(n)
            if norm_len > 0:
                n = n / norm_len
            else:
                n = np.array([0, 0, 1])
            
            # Erstelle einen Kreis senkrecht zur Normalen
            radius = 5.0
            
            # Finde zwei Vektoren senkrecht zur Normalen
            if abs(n[2]) < 0.9:
                u = np.cross(n, [0, 0, 1])
            else:
                u = np.cross(n, [1, 0, 0])
            u = u / np.linalg.norm(u)
            v = np.cross(n, u)
            
            # Kreis-Punkte
            points = []
            for i in range(33):
                angle = i * 2 * math.pi / 32
                p = center + radius * (math.cos(angle) * u + math.sin(angle) * v)
                points.append(p)
            
            # Als Linie zeichnen
            pts = np.array(points)
            lines = pv.lines_from_points(pts)
            self.plotter.add_mesh(lines, color='cyan', line_width=4, name='body_face_highlight')
            
            # Normale als Pfeil (zeigt Extrude-Richtung)
            arrow_end = center + n * 8
            arrow = pv.Arrow(start=center, direction=n, scale=8)
            self.plotter.add_mesh(arrow, color='cyan', name='body_face_arrow')
            
            self.plotter.update()
        except Exception as e:
            print(f"Draw highlight error: {e}")
    
    def _clear_body_face_highlight(self):
        """Entfernt Body-Face-Highlight"""
        try:
            self.plotter.remove_actor('body_face_highlight')
        except: pass
        try:
            self.plotter.remove_actor('body_face_arrow')
        except: pass
        try:
            self.plotter.update()
        except: pass
    
    def _click_body_face(self):
        """Klick auf Body-Face - bereitet Extrusion vor"""
        if self.hovered_body_face is None:
            return
        
        body_id, cell_id, normal, pos = self.hovered_body_face
        
        # Speichere die Flächen-Daten für Extrusion
        # Wir erstellen ein "detected_face" aus der Body-Fläche
        self.body_face_extrude = {
            'body_id': body_id,
            'cell_id': cell_id,
            'normal': normal,
            'origin': pos,
            'mesh': self.bodies.get(body_id, {}).get('mesh')
        }
        
        # Markiere als ausgewählt
        self.selected_faces.clear()
        self.selected_faces.add(-1)  # Special marker für Body-Face
        
        print(f"Body face selected: body={body_id}, normal={normal}, pos={pos}")
        
        # Zeige Preview
        self._draw_body_face_selection(pos, normal)
    
    def _draw_body_face_selection(self, pos, normal):
        """Zeichnet Auswahl-Highlight auf Body-Face"""
        try:
            center = np.array(pos)
            n = np.array(normal)
            n = n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else np.array([0,0,1])
            
            # Größerer Kreis für Auswahl
            radius = 8.0
            if abs(n[2]) < 0.9:
                u = np.cross(n, [0, 0, 1])
            else:
                u = np.cross(n, [1, 0, 0])
            u = u / np.linalg.norm(u)
            v = np.cross(n, u)
            
            points = []
            for i in range(33):
                angle = i * 2 * math.pi / 32
                p = center + radius * (math.cos(angle) * u + math.sin(angle) * v)
                points.append(p)
            
            pts = np.array(points)
            lines = pv.lines_from_points(pts)
            self.plotter.add_mesh(lines, color='orange', line_width=4, name='body_face_selection')
            self.plotter.update()
        except:
            pass
    
    def _clear_preview(self):
        if self._preview_actor: 
            try: self.plotter.remove_actor(self._preview_actor)
            except: pass; self._preview_actor=None

    def _draw_selectable_faces(self):
        """Zeichnet NUR die gehoverte und ausgewählte Flächen - nicht alle"""
        self._clear_face_actors()
        
        for i, face in enumerate(self.detected_faces):
            is_selected = i in self.selected_faces
            is_hovered = i == self.hovered_face
            
            # Nur zeichnen wenn selected oder hovered!
            if not is_selected and not is_hovered:
                continue
            
            if is_selected: 
                col = 'orange'
                op = 0.7
            else:  # hovered
                col = '#44aaff'
                op = 0.5
            
            # Body-Face oder Sketch-Face?
            if face.get('type') == 'body_face':
                self._draw_body_face_overlay(i, face, col, op)
            else:
                # Sketch-Face
                v, f = self.get_extrusion_data(i, 0)
                if v:
                    vv = np.array(v, dtype=np.float32)
                    ff = []
                    for x in f: ff.extend([3]+list(x))
                    mesh = pv.PolyData(vv, np.array(ff, dtype=np.int32))
                    n = f"face_{i}"
                    self.plotter.add_mesh(mesh, color=col, opacity=op, name=n, pickable=True)
                    self._face_actors.append(n)
        
        self.plotter.update()
    
    def _draw_body_face_overlay(self, face_idx, face, color, opacity):
        """Zeichnet Overlay für eine Body-Fläche"""
        try:
            mesh = face.get('mesh')
            cell_ids = face.get('cell_ids', [])
            
            if mesh is None or not cell_ids:
                return
            
            # Extrahiere nur die Zellen dieser Fläche
            face_mesh = mesh.extract_cells(cell_ids)
            
            n = f"face_{face_idx}"
            self.plotter.add_mesh(face_mesh, color=color, opacity=opacity, 
                                  name=n, pickable=True, show_edges=False)
            self._face_actors.append(n)
        except Exception as e:
            print(f"Body face overlay error: {e}")

    def _clear_face_actors(self):
        for n in self._face_actors:
            try: self.plotter.remove_actor(n)
            except: pass
        self._face_actors.clear()

    def _render_sketch(self, s):
        norm = tuple(getattr(s,'plane_normal',(0,0,1))); orig = getattr(s,'plane_origin',(0,0,0))
        sid = getattr(s,'id',id(s))
        
        def t3d(x,y): return self._transform_2d_to_3d(x,y,norm,orig)
        
        # Linien
        for i,l in enumerate(getattr(s,'lines',[])):
            col = 'gray' if getattr(l,'construction',False) else '#4d94ff'
            self.plotter.add_mesh(pv.Line(t3d(l.start.x,l.start.y), t3d(l.end.x,l.end.y)), color=col, line_width=3, name=f"s_{sid}_l_{i}")
            self._sketch_actors.append(f"s_{sid}_l_{i}")
            
        # Kreise
        for i,c in enumerate(getattr(s,'circles',[])):
            pts = [t3d(c.center.x+c.radius*math.cos(j*6.28/64), c.center.y+c.radius*math.sin(j*6.28/64)) for j in range(65)]
            col = 'gray' if getattr(c,'construction',False) else '#4d94ff'
            self.plotter.add_mesh(pv.lines_from_points(np.array(pts)), color=col, line_width=3, name=f"s_{sid}_c_{i}")
            self._sketch_actors.append(f"s_{sid}_c_{i}")
            
        # Arcs (Neu: Bessere Darstellung)
        for i, arc in enumerate(getattr(s, 'arcs', [])):
            col = 'gray' if getattr(arc,'construction',False) else '#4d94ff'
            pts = []
            start = arc.start_angle
            end = arc.end_angle
            sweep = end - start
            if sweep < 0.1: sweep += 360
            steps = 32
            for j in range(steps+1):
                t = math.radians(start + sweep * (j/steps))
                x = arc.center.x + arc.radius * math.cos(t)
                y = arc.center.y + arc.radius * math.sin(t)
                pts.append(t3d(x, y))
            if len(pts) > 1:
                self.plotter.add_mesh(pv.lines_from_points(np.array(pts)), color=col, line_width=3, name=f"s_{sid}_a_{i}")
                self._sketch_actors.append(f"s_{sid}_a_{i}")
                
        # Splines (Neu!)
        for i, spline in enumerate(getattr(s, 'splines', [])):
            col = 'gray' if getattr(spline,'construction',False) else '#4d94ff'
            pts_2d = []
            # Punkte holen
            if hasattr(spline, 'get_curve_points'):
                 pts_2d = spline.get_curve_points(segments_per_span=10)
            elif hasattr(spline, 'to_lines'):
                 lines = spline.to_lines(segments_per_span=10)
                 if lines:
                     pts_2d.append((lines[0].start.x, lines[0].start.y))
                     for l in lines: pts_2d.append((l.end.x, l.end.y))
            
            if len(pts_2d) > 1:
                pts_3d = [t3d(p[0], p[1]) for p in pts_2d]
                self.plotter.add_mesh(pv.lines_from_points(np.array(pts_3d)), color=col, line_width=3, name=f"s_{sid}_sp_{i}")
                self._sketch_actors.append(f"s_{sid}_sp_{i}")

    def _show_selection_planes(self):
        sz = 150; op = 0.25
        self.plotter.add_mesh(pv.Plane(center=(0,0,0), direction=(0,0,1), i_size=sz, j_size=sz), color='blue', opacity=op, name='xy', pickable=True)
        self.plotter.add_mesh(pv.Plane(center=(0,0,0), direction=(0,1,0), i_size=sz, j_size=sz), color='green', opacity=op, name='xz', pickable=True)
        self.plotter.add_mesh(pv.Plane(center=(0,0,0), direction=(1,0,0), i_size=sz, j_size=sz), color='red', opacity=op, name='yz', pickable=True)
        self._plane_actors = {'xy':'xy','xz':'xz','yz':'yz'}

    def _hide_selection_planes(self):
        for n in ['xy','xz','yz']: 
            try: self.plotter.remove_actor(n)
            except: pass
        self._plane_actors.clear()

    def set_view(self, view_name):
        """Setzt eine Standardansicht"""
        if not HAS_PYVISTA:
            return
        
        views = {
            'iso': 'iso',
            'top': 'xy',
            'front': 'xz', 
            'right': 'yz',
            'back': '-xz',
            'left': '-yz',
            'bottom': '-xy'
        }
        
        if view_name in views:
            try:
                self.plotter.view_vector(views[view_name])
                self.plotter.reset_camera()
                self.view_changed.emit()
            except:
                # Fallback
                if view_name == 'iso':
                    self.plotter.view_isometric()
                elif view_name == 'top':
                    self.plotter.view_xy()
                elif view_name == 'front':
                    self.plotter.view_xz()
                elif view_name == 'right':
                    self.plotter.view_yz()
                self.plotter.reset_camera()
                self.view_changed.emit()
    def update(self): self.plotter.update(); super().update()


def create_viewport(parent=None):
    return PyVistaViewport(parent) if HAS_PYVISTA else QWidget(parent)