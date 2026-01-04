"""
LiteCAD - Unified Main Window
V2.7: Refactored - PyVista required, no OpenGL fallback
"""

import sys
import os
import json
import math

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QToolBar, QStatusBar, QMenuBar, QMenu, QFileDialog,
    QMessageBox, QSplitter, QFrame, QTreeWidget, QTreeWidgetItem,
    QStackedWidget, QApplication, QDialog, QFormLayout,
    QDoubleSpinBox, QDialogButtonBox, QSpinBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QPushButton, QHeaderView, QComboBox,
    QScrollArea
)
from PySide6.QtCore import Qt, Signal, QSize, QTimer, QPointF, QEvent
from PySide6.QtGui import QKeySequence, QAction, QFont, QPainter, QPen, QBrush, QColor, QPolygonF

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from i18n import tr
from sketcher import Sketch
from modeling import Document, Body, ExtrudeFeature, FeatureType
from gui.sketch_editor import SketchEditor, SketchTool
from gui.tool_panel import ToolPanel, PropertiesPanel
from gui.tool_panel_3d import ToolPanel3D, BodyPropertiesPanel
from gui.browser import ProjectBrowser
from gui.input_panels import ExtrudeInputPanel, FilletChamferPanel
from gui.viewport_pyvista import PyVistaViewport, HAS_PYVISTA, HAS_BUILD123D

if not HAS_PYVISTA:
    print("ERROR: PyVista is required! Install with: pip install pyvista pyvistaqt")
    sys.exit(1)


# FeatureTree wird durch ProjectBrowser aus gui/browser.py ersetzt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiteCAD")
        self.setMinimumSize(1400, 900)
        self.document = Document("Projekt1")
        self.mode = "3d"
        self.active_sketch = None
        self._apply_theme()
        self._create_ui()
        self._create_menus()
        self._connect_signals()
        QApplication.instance().installEventFilter(self)
        self._set_mode("3d")
        self.statusBar().showMessage(tr("Ready"))

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #1e1e1e; }
            QMenuBar { background: #1e1e1e; color: #ccc; padding: 2px; border-bottom: 1px solid #333; }
            QMenuBar::item { padding: 4px 8px; }
            QMenuBar::item:selected { background: #333; }
            QMenu { background: #252526; color: #ccc; border: 1px solid #333; }
            QMenu::item { padding: 6px 20px; }
            QMenu::item:selected { background: #0078d4; }
            QToolBar { 
                background: #1e1e1e; 
                border: none;
                border-bottom: 1px solid #333;
                padding: 2px;
                spacing: 2px;
            }
            QToolBar QToolButton {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                color: #ccc;
                padding: 4px 8px;
                font-size: 11px;
            }
            QToolBar QToolButton:hover {
                background: #333;
                border-color: #444;
            }
            QToolBar QToolButton:pressed {
                background: #0078d4;
            }
            QSplitter::handle {
                background: #333;
            }
            QSplitter::handle:horizontal {
                width: 1px;
            }
            QSplitter::handle:vertical {
                height: 1px;
            }
            QSplitter::handle:hover {
                background: #0078d4;
            }
            QStatusBar { 
                background: #1e1e1e; 
                color: #888; 
                border-top: 1px solid #333;
            }
        """)

    def _create_ui(self):
        central = QWidget()
        central.setStyleSheet("background-color: #1e1e1e;")
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # === HAUPTSPLITTER ===
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setStyleSheet("QSplitter { background: #1e1e1e; }")
        
        # === LINKE SEITE: Browser + Tools horizontal ===
        left_widget = QWidget()
        left_widget.setStyleSheet("background-color: #1e1e1e;")
        left_layout = QHBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        
        # Project Browser (collapsible)
        self.browser = ProjectBrowser()
        self.browser.set_document(self.document)
        left_layout.addWidget(self.browser)
        
        # Tool-Panel Stack (3D oder 2D)
        self.tool_stack = QStackedWidget()
        self.tool_stack.setMinimumWidth(140)
        self.tool_stack.setMaximumWidth(200)
        self.tool_stack.setStyleSheet("background-color: #1e1e1e;")
        
        # 3D-ToolPanel (Index 0)
        self.tool_panel_3d = ToolPanel3D()
        self.tool_stack.addWidget(self.tool_panel_3d)
        
        # 2D-ToolPanel (Index 1) 
        self.tool_panel = ToolPanel()
        self.tool_stack.addWidget(self.tool_panel)
        
        left_layout.addWidget(self.tool_stack)
        
        # Left Widget zum Splitter hinzufügen
        self.main_splitter.addWidget(left_widget)
        
        # === MITTE: Viewport / Sketch Editor ===
        self.center_stack = QStackedWidget()
        self.center_stack.setStyleSheet("background-color: #1e1e1e;")
        
        self.viewport_3d = PyVistaViewport()
        self.center_stack.addWidget(self.viewport_3d)
        
        self.sketch_editor = SketchEditor()
        self.center_stack.addWidget(self.sketch_editor)
        
        self.main_splitter.addWidget(self.center_stack)
        self.sketch_editor.viewport = self.viewport_3d
        
        # Splitter-Einstellungen
        self.main_splitter.setStretchFactor(0, 0)  # Links nicht stretchen
        self.main_splitter.setStretchFactor(1, 1)  # Viewport stretchen
        self.main_splitter.setSizes([340, 1000])
        
        layout.addWidget(self.main_splitter)
        
        # === RECHTE SEITE: Properties (nur wenn nötig) ===
        self.right_stack = QStackedWidget()
        self.right_stack.setMinimumWidth(140)
        self.right_stack.setMaximumWidth(200)
        self.right_stack.setStyleSheet("background-color: #1e1e1e;")
        
        # 3D-Properties (Index 0)
        self.body_properties = BodyPropertiesPanel()
        self.right_stack.addWidget(self.body_properties)
        
        # 2D-Properties (Index 1)
        self.properties_panel = PropertiesPanel()
        self.right_stack.addWidget(self.properties_panel)
        
        self.right_stack.setVisible(False)
        layout.addWidget(self.right_stack)
        
        # Extrude Input Panel (immer sichtbar während Extrude-Modus)
        self.extrude_panel = ExtrudeInputPanel(self)
        self.extrude_panel.height_changed.connect(self._on_extrude_panel_height_changed)
        self.extrude_panel.confirmed.connect(self._on_extrude_confirmed)
        self.extrude_panel.cancelled.connect(self._on_extrude_cancelled)
        self.extrude_panel.bodies_visibility_toggled.connect(self._on_toggle_bodies_visibility)
        
        # Fillet/Chamfer Panel
        self.fillet_panel = FilletChamferPanel(self)
        self.fillet_panel.radius_changed.connect(self._on_fillet_radius_changed)
        self.fillet_panel.confirmed.connect(self._on_fillet_confirmed)
        self.fillet_panel.cancelled.connect(self._on_fillet_cancelled)
        
        self._fillet_mode = None  # 'fillet' or 'chamfer'
        self._fillet_target_body = None
        
        self._create_toolbar()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_extrude_panel()
    
    def _position_extrude_panel(self):
        """Positioniert das Extrude-Panel am unteren Rand des Fensters"""
        if hasattr(self, 'extrude_panel') and self.extrude_panel.isVisible():
            panel_width = self.extrude_panel.width() if self.extrude_panel.width() > 100 else 320
            x = (self.width() - panel_width) // 2
            y = self.height() - 100
            self.extrude_panel.move(x, y)

    def _create_toolbar(self):
        """Minimale Toolbar - nur Modus-Umschaltung"""
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)
        
        # Nur Sketch-Fertig Button (erscheint im Sketch-Modus)
        self.btn_finish_sketch = tb.addAction("✓ Fertig (Esc)")
        self.btn_finish_sketch.triggered.connect(self._finish_sketch)
        self.btn_finish_sketch.setToolTip("Sketch beenden und zurück zum 3D-Modus")
        self.btn_finish_sketch.setVisible(False)
        
        # Versteckte Buttons für Tastenkürzel
        self.btn_new_sketch = None  # Wird über Tool Panel gesteuert
        self.btn_extrude = None     # Wird über Tool Panel gesteuert

    def _create_menus(self):
        mb = self.menuBar()
        
        # Datei-Menü
        file_menu = mb.addMenu(tr("&Datei"))
        file_menu.addAction(tr("Neu"), self._new_project, QKeySequence.New)
        file_menu.addAction(tr("Öffnen..."), lambda: None, QKeySequence.Open)
        file_menu.addAction(tr("Speichern"), lambda: None, QKeySequence.Save)
        file_menu.addSeparator()
        file_menu.addAction(tr("STL exportieren..."), self._export_stl)
        file_menu.addSeparator()
        file_menu.addAction(tr("Beenden"), self.close, QKeySequence.Quit)
        
        # Bearbeiten-Menü
        edit_menu = mb.addMenu(tr("&Bearbeiten"))
        edit_menu.addAction(tr("Rückgängig"), lambda: None, QKeySequence.Undo)
        edit_menu.addAction(tr("Wiederholen"), lambda: None, QKeySequence.Redo)
        
        # Ansicht-Menü
        view_menu = mb.addMenu(tr("&Ansicht"))
        view_menu.addAction(tr("Isometrisch"), lambda: self.viewport_3d.set_view('iso') if hasattr(self.viewport_3d, 'set_view') else None)
        view_menu.addAction(tr("Oben (XY)"), lambda: self.viewport_3d.set_view('top') if hasattr(self.viewport_3d, 'set_view') else None)
        view_menu.addAction(tr("Vorne (XZ)"), lambda: self.viewport_3d.set_view('front') if hasattr(self.viewport_3d, 'set_view') else None)
        view_menu.addAction(tr("Rechts (YZ)"), lambda: self.viewport_3d.set_view('right') if hasattr(self.viewport_3d, 'set_view') else None)
        
        # Hilfe-Menü
        help_menu = mb.addMenu(tr("&Hilfe"))
        help_menu.addAction(tr("Über LiteCAD"), self._show_about)

    def _connect_signals(self):
        # 2D Tool Panel
        self.tool_panel.tool_selected.connect(self._on_sketch_tool_selected)
        self.tool_panel.option_changed.connect(self._on_opt_change)
        
        # 3D Tool Panel
        self.tool_panel_3d.action_triggered.connect(self._on_3d_action)
        
        # Browser
        self.browser.feature_double_clicked.connect(self._edit_feature)
        self.browser.feature_selected.connect(self._on_feature_selected)
        self.browser.plane_selected.connect(self._on_browser_plane_selected)
        
        # WICHTIG: Visibility changed muss ALLES neu laden (Sketches + Bodies)
        self.browser.visibility_changed.connect(self._update_viewport_all)
        
        if hasattr(self.viewport_3d, 'set_body_visibility'):
            self.browser.body_vis_changed.connect(self.viewport_3d.set_body_visibility)
        
        # Viewport Signale
        self.viewport_3d.plane_clicked.connect(self._on_plane_selected)
        if hasattr(self.viewport_3d, 'custom_plane_clicked'):
            self.viewport_3d.custom_plane_clicked.connect(self._on_custom_plane_selected)
            
        self.viewport_3d.extrude_requested.connect(self._on_extrusion_finished)
        self.viewport_3d.height_changed.connect(self._on_viewport_height_changed)
    
    def _on_3d_action(self, action: str):
        """Verarbeitet 3D-Tool-Aktionen"""
        actions = {
            'new_sketch': self._new_sketch,
            'extrude': self._extrude_dialog,
            'export_stl': self._export_stl,
            'export_step': lambda: self._show_not_implemented("STEP Export"),
            'export_dxf': lambda: self._show_not_implemented("DXF Export"),
            'primitive_box': lambda: self._show_not_implemented("Box Primitiv"),
            'primitive_cylinder': lambda: self._show_not_implemented("Zylinder Primitiv"),
            'primitive_sphere': lambda: self._show_not_implemented("Kugel Primitiv"),
            'revolve': lambda: self._show_not_implemented("Revolve"),
            'sweep': lambda: self._show_not_implemented("Sweep"),
            'loft': lambda: self._show_not_implemented("Loft"),
            'fillet': self._start_fillet,
            'chamfer': self._start_chamfer,
            'shell': lambda: self._show_not_implemented("Shell"),
            'hole': lambda: self._show_not_implemented("Bohrung"),
            'boolean_union': lambda: self._show_not_implemented("Boolean Vereinen"),
            'boolean_cut': lambda: self._show_not_implemented("Boolean Abziehen"),
            'boolean_intersect': lambda: self._show_not_implemented("Boolean Schneiden"),
            'move_body': lambda: self._show_not_implemented("Körper verschieben"),
            'copy_body': lambda: self._show_not_implemented("Körper kopieren"),
            'rotate_body': lambda: self._show_not_implemented("Körper drehen"),
            'mirror_body': lambda: self._show_not_implemented("Körper spiegeln"),
            'scale_body': lambda: self._show_not_implemented("Körper skalieren"),
            'measure': lambda: self._show_not_implemented("Messen"),
            'mass_props': lambda: self._show_not_implemented("Masseeigenschaften"),
            'check': lambda: self._show_not_implemented("Geometrie prüfen"),
            'section': lambda: self._show_not_implemented("Schnittansicht"),
            'thread': lambda: self._show_not_implemented("Gewinde"),
            'pattern': lambda: self._show_not_implemented("Muster"),
        }
        
        if action in actions:
            actions[action]()
        else:
            print(f"Unbekannte 3D-Aktion: {action}")
    
    def _show_not_implemented(self, feature: str):
        """Zeigt Hinweis für noch nicht implementierte Features"""
        self.statusBar().showMessage(f"⚠ {feature} - {tr('Coming soon!')}", 3000)
    
    def _on_sketch_tool_selected(self, tool_name: str):
        """Verarbeitet Tool-Auswahl aus dem Sketch-ToolPanel"""
        # Spezielle Aktionen (kein Tool-Wechsel)
        special_actions = {
            'import_dxf': self.sketch_editor.import_dxf,
            'export_dxf': self.sketch_editor.export_dxf,
            'import_svg': lambda: self._show_not_implemented("SVG Import"),
            'export_svg': lambda: self._show_not_implemented("SVG Export"),
        }
        
        if tool_name in special_actions:
            special_actions[tool_name]()
        else:
            # Normaler Tool-Wechsel
            tool = getattr(SketchTool, tool_name.upper(), SketchTool.SELECT)
            self.sketch_editor.set_tool(tool)
    
    def _on_feature_selected(self, data):
        """Wird aufgerufen wenn ein Feature im Tree ausgewählt wird"""
        if data and len(data) >= 2:
            if data[0] == 'body':
                self.body_properties.update_body(data[1])
            else:
                self.body_properties.clear()

    def _update_viewport_all(self):
        """Aktualisiert ALLES im Viewport"""
        # Sketches
        self.viewport_3d.set_sketches(self.browser.get_visible_sketches())
        
        # Bodies - komplett neu laden um gelöschte zu entfernen
        self.viewport_3d.clear_bodies()
        colors = [(0.6,0.6,0.8), (0.8,0.6,0.6), (0.6,0.8,0.6)]
        for i, (b, visible) in enumerate(self.browser.get_visible_bodies()):
            if visible and hasattr(b, '_mesh_vertices'):
                self.viewport_3d.add_body(b.id, b.name, b._mesh_vertices, b._mesh_triangles, colors[i%3])

    def _set_mode(self, mode):
        self.mode = mode
        if mode == "3d":
            # 3D-Modus
            self.tool_stack.setCurrentIndex(0)  # 3D-ToolPanel
            self.center_stack.setCurrentIndex(0)  # Viewport
            self.right_stack.setVisible(False)
            self.btn_finish_sketch.setVisible(False)
            self._update_viewport_all()
        else:
            # Sketch-Modus
            self.tool_stack.setCurrentIndex(1)  # 2D-ToolPanel
            self.center_stack.setCurrentIndex(1)  # Sketch Editor
            self.right_stack.setCurrentIndex(1)  # 2D Properties
            self.right_stack.setVisible(True)
            self.btn_finish_sketch.setVisible(True)
            self.sketch_editor.setFocus()

    def _new_sketch(self):
        self.viewport_3d.set_plane_select_mode(True)
        self.statusBar().showMessage(tr("Wähle Ebene: 1=XY, 2=XZ, 3=YZ oder Klick auf Fläche"))
        self.setFocus()

    def _on_browser_plane_selected(self, plane):
        """Wird aufgerufen wenn eine Ebene im Browser angeklickt wird"""
        self._on_plane_selected(plane)
    
    def _on_plane_selected(self, plane):
        self.viewport_3d.set_plane_select_mode(False)
        origins = {'xy':((0,0,0),(0,0,1)), 'xz':((0,0,0),(0,1,0)), 'yz':((0,0,0),(1,0,0))}
        o, n = origins.get(plane, ((0,0,0),(0,0,1)))
        self._create_sketch_at(o, n)

    def _on_custom_plane_selected(self, origin, normal):
        self.viewport_3d.set_plane_select_mode(False)
        self._create_sketch_at(origin, normal)

    def _create_sketch_at(self, origin, normal):
        s = self.document.new_sketch(f"Sketch{len(self.document.sketches)+1}")
        s.plane_origin = origin; s.plane_normal = normal
        self.active_sketch = s; self.sketch_editor.sketch = s
        
        # Bodies als Referenz an SketchEditor übergeben
        self._set_sketch_body_references(origin, normal)
        
        self._set_mode("sketch")
        self.browser.refresh()
    
    def _set_sketch_body_references(self, origin, normal):
        """Sammelt Body-Daten und übergibt sie an den SketchEditor"""
        bodies_data = []
        
        for body in self.document.bodies:
            mesh = self.viewport_3d.get_body_mesh(body.id)
            if mesh is not None:
                # Hole Farbe aus viewport
                body_info = self.viewport_3d.bodies.get(body.id, {})
                color = body_info.get('color', (0.6, 0.6, 0.8))
                bodies_data.append({
                    'mesh': mesh,
                    'color': color
                })
        
        # Übergebe an SketchEditor
        if hasattr(self.sketch_editor, 'set_reference_bodies'):
            self.sketch_editor.set_reference_bodies(bodies_data, normal, origin)

    def _finish_sketch(self):
        # Body-Referenzen im SketchEditor löschen
        if hasattr(self.sketch_editor, 'set_reference_bodies'):
            self.sketch_editor.set_reference_bodies([], (0,0,1), (0,0,0))
        self._set_mode("3d"); self.browser.refresh()

    def _extrude_dialog(self):
        """Startet den Extrude-Modus mit sichtbarem Input-Panel"""
        self.viewport_3d.set_extrude_mode(True)
        self.extrude_panel.reset()
        
        # Panel unten mittig im Fenster positionieren
        panel_width = self.extrude_panel.width() if self.extrude_panel.width() > 100 else 320
        x = (self.width() - panel_width) // 2
        y = self.height() - 100
        self.extrude_panel.move(x, y)
        self.extrude_panel.setVisible(True)
        self.extrude_panel.height_input.setFocus()
        
        self.statusBar().showMessage(tr("Wähle Fläche und ziehe oder gib Höhe ein | Enter=OK | Esc=Abbrechen | F=Flip"))

    def _on_viewport_height_changed(self, h):
        """Wird aufgerufen wenn sich die Höhe durch Maus-Drag ändert"""
        # Update das Input-Panel mit dem aktuellen Wert
        self.extrude_panel.set_height(h)
    
    def _on_extrude_panel_height_changed(self, height):
        """Live-Vorschau wenn Wert im Panel geändert wird"""
        if hasattr(self.viewport_3d, 'show_extrude_preview'):
            self.viewport_3d.show_extrude_preview(height)
    
    def _on_extrude_confirmed(self):
        """Extrude bestätigt"""
        height = self.extrude_panel.get_height()
        
        # WICHTIG: Operation aus dem Panel holen!
        op = "New Body"
        if hasattr(self.extrude_panel, 'get_operation'):
            op = self.extrude_panel.get_operation()
            
        if hasattr(self.viewport_3d, 'selected_faces') and self.viewport_3d.selected_faces:
            self._on_extrusion_finished(list(self.viewport_3d.selected_faces), height, op)
        self.extrude_panel.setVisible(False)
    
    def _on_extrude_cancelled(self):
        """Extrude abgebrochen"""
        self.viewport_3d.set_extrude_mode(False)
        self.extrude_panel.setVisible(False)
        # Bodies wieder einblenden falls versteckt
        self.viewport_3d.set_all_bodies_visible(True)
        self.statusBar().showMessage(tr("Extrude abgebrochen"), 2000)
    
    def _on_toggle_bodies_visibility(self, hide: bool):
        """Toggle alle Bodies sichtbar/unsichtbar im Extrude-Modus"""
        self.viewport_3d.set_all_bodies_visible(not hide)

    def _on_extrusion_finished(self, face_indices, height, operation="New Body"):
        if not face_indices or abs(height) < 0.1:
            self.extrude_panel.setVisible(False)
            self.viewport_3d.set_all_bodies_visible(True)
            return
        
        # Versuche Build123d Extrusion für bessere Qualität
        if HAS_BUILD123D and self.active_sketch:
            success = self._extrude_with_build123d(face_indices, height, operation)
            if success:
                self.extrude_panel.setVisible(False)
                self.viewport_3d.set_extrude_mode(False)
                self.viewport_3d.set_all_bodies_visible(True)
                self.browser.refresh()
                return
        
        # Fallback: PyVista-basierte Extrusion
        if hasattr(self.viewport_3d, 'get_extrusion_data'):
            for idx in face_indices:
                verts, faces = self.viewport_3d.get_extrusion_data(idx, height)
                if verts: 
                    self._create_body_from_data(verts, faces, height, operation)
        
        self.extrude_panel.setVisible(False)
        self.viewport_3d.set_extrude_mode(False)
        self.viewport_3d.set_all_bodies_visible(True)
        self.browser.refresh()
    
    def _extrude_with_build123d(self, face_indices, height, operation):
        """Extrudiert mit Build123d für echte BREP-Geometrie"""
        try:
            from build123d import (
                BuildPart, BuildSketch, Plane, extrude,
                Line, Circle, Arc, Polyline, make_face,
                Location, Vector, fillet, chamfer
            )
            import numpy as np
            
            if not self.active_sketch:
                return False
            
            sketch = self.active_sketch
            plane_origin = getattr(sketch, 'plane_origin', (0, 0, 0))
            plane_normal = getattr(sketch, 'plane_normal', (0, 0, 1))
            
            # Bestimme Build123d Plane
            ox, oy, oz = plane_origin
            nx, ny, nz = plane_normal
            
            if abs(nz - 1) < 0.01:
                b3d_plane = Plane.XY.offset(oz)
            elif abs(ny - 1) < 0.01:
                b3d_plane = Plane.XZ.offset(oy)
            elif abs(nx - 1) < 0.01:
                b3d_plane = Plane.YZ.offset(ox)
            else:
                b3d_plane = Plane(origin=(ox, oy, oz), z_dir=(nx, ny, nz))
            
            # Sammle geschlossene Profile aus dem Sketch
            profiles = self.viewport_3d.closed_profiles if hasattr(self.viewport_3d, 'closed_profiles') else []
            
            if not profiles and hasattr(self, 'sketch_editor'):
                profiles = self.sketch_editor.closed_profiles
            
            if not profiles:
                print("Build123d: Keine geschlossenen Profile gefunden")
                return False
            
            # Erstelle Build123d Part
            with BuildPart() as part:
                with BuildSketch(b3d_plane):
                    # Konvertiere Profile zu Build123d
                    for profile in profiles:
                        if hasattr(profile, 'exterior'):
                            # Shapely Polygon
                            coords = list(profile.exterior.coords)
                            if len(coords) > 2:
                                pts = [(c[0], c[1]) for c in coords[:-1]]  # Ohne letzten Punkt (=erster)
                                Polyline(*pts, close=True)
                        elif isinstance(profile, (list, tuple)):
                            # Liste von Punkten
                            if len(profile) > 2:
                                Polyline(*profile, close=True)
                    
                    make_face()
                
                # Extrudieren
                extrude(amount=height)
            
            solid = part.part
            
            if solid is None or not hasattr(solid, 'vertices'):
                print("Build123d: Extrusion fehlgeschlagen")
                return False
            
            # Boolean Operation wenn nötig
            if operation in ["Join", "Cut", "Intersect"] and self.document.bodies:
                target_body = self.document.bodies[-1]
                if hasattr(target_body, '_build123d_solid') and target_body._build123d_solid:
                    target_solid = target_body._build123d_solid
                    
                    if operation == "Join":
                        solid = target_solid + solid
                    elif operation == "Cut":
                        solid = target_solid - solid
                    elif operation == "Intersect":
                        solid = target_solid & solid
                    
                    # Update bestehenden Body
                    self._update_body_from_build123d(target_body, solid)
                    return True
            
            # Neuer Body
            b = self.document.new_body(f"Body{len(self.document.bodies)+1}")
            feat = ExtrudeFeature(FeatureType.EXTRUDE, "Extrude", None, abs(height))
            b.features.append(feat)
            b._build123d_solid = solid  # BREP speichern für Fillet/Chamfer!
            
            self._update_body_from_build123d(b, solid)
            
            self.statusBar().showMessage(f"Build123d Extrusion: {height}mm", 2000)
            return True
            
        except Exception as e:
            print(f"Build123d Extrusion error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _update_body_from_build123d(self, body, solid):
        """Konvertiert Build123d Solid zu Mesh und aktualisiert Body"""
        try:
            # Tessellieren für Anzeige
            mesh_data = solid.tessellate(tolerance=0.1)
            
            # Build123d gibt (vertices, triangles) zurück
            verts = [(v.X, v.Y, v.Z) for v in mesh_data[0]]
            faces = [tuple(t) for t in mesh_data[1]]
            
            body._mesh_vertices = verts
            body._mesh_triangles = faces
            body._build123d_solid = solid
            
            self.viewport_3d.add_body(body.id, body.name, verts, faces)
            
        except Exception as e:
            print(f"Build123d mesh conversion error: {e}")

    def _show_extrude_input_dialog(self):
        """Legacy Dialog - wird durch Panel ersetzt"""
        # Falls Tab gedrückt wird, fokussiere das Panel
        self.extrude_panel.height_input.setFocus()
        self.extrude_panel.height_input.selectAll()

    def _create_body_from_data(self, verts, faces, height, operation):
        import numpy as np
        import pyvista as pv
        
        # Versuche Build123d für robustere Operationen
        if HAS_BUILD123D and operation in ["Join", "Cut", "Intersect"]:
            result = self._build123d_boolean(verts, faces, height, operation)
            if result is not None:
                return
        
        # PyVista Fallback
        v = np.array(verts, dtype=np.float32)
        f = []
        for face in faces: f.extend([len(face)] + list(face))
        
        new_mesh = pv.PolyData(v, np.array(f, dtype=np.int32))
        new_mesh = self._prepare_mesh_for_boolean(new_mesh)
        
        if operation in ["Join", "Cut", "Intersect"]:
            target_body = None
            target_mesh = None
            
            for body in self.document.bodies:
                mesh = self.viewport_3d.get_body_mesh(body.id)
                if mesh and mesh.n_points > 0:
                    target_body = body
                    target_mesh = self._prepare_mesh_for_boolean(mesh)
                    break
            
            if target_mesh:
                result = self._perform_boolean_operation(target_mesh, new_mesh, operation)
                if result and result.n_points > 0:
                    self._update_body_mesh(target_body, result)
                    return
                else:
                    print(f"Boolean operation '{operation}' failed - creating new body instead")

        # Neuer Body
        b = self.document.new_body(f"Body{len(self.document.bodies)+1}")
        feat = ExtrudeFeature(FeatureType.EXTRUDE, "Extrude", None, abs(height))
        b.features.append(feat)
        b._mesh_vertices = verts
        b._mesh_triangles = faces
        b._build123d_solid = None  # Placeholder für BREP
        self.viewport_3d.add_body(b.id, b.name, verts, faces)
    
    def _build123d_boolean(self, verts, faces, height, operation):
        """Führt Boolean-Operation mit Build123d durch (robuster als PyVista)"""
        try:
            from build123d import Solid, Compound
            import numpy as np
            import pyvista as pv
            
            # Konvertiere neues Mesh zu PyVista
            v = np.array(verts, dtype=np.float32)
            f = []
            for face in faces: f.extend([len(face)] + list(face))
            new_mesh = pv.PolyData(v, np.array(f, dtype=np.int32)).clean()
            
            # Finde Target Body mit Build123d Solid
            target_body = None
            target_solid = None
            
            for body in self.document.bodies:
                if hasattr(body, '_build123d_solid') and body._build123d_solid is not None:
                    target_body = body
                    target_solid = body._build123d_solid
                    break
            
            if target_solid is None:
                # Kein Build123d Solid vorhanden - Fallback
                return None
            
            # Konvertiere neues Mesh zu Build123d (schwierig - braucht Solid)
            # Für jetzt: Fallback zu PyVista
            print("Build123d Boolean: Target hat kein BREP - verwende PyVista")
            return None
            
        except Exception as e:
            print(f"Build123d Boolean error: {e}")
            return None
    
    def _prepare_mesh_for_boolean(self, mesh):
        """Bereitet ein Mesh für Boolean-Operationen vor"""
        import pyvista as pv
        
        try:
            # 1. Triangulieren (boolean braucht Dreiecke)
            mesh = mesh.triangulate()
            
            # 2. Clean (entfernt doppelte Punkte, degenerierte Faces)
            mesh = mesh.clean(tolerance=1e-6)
            
            # 3. Normals berechnen und konsistent machen
            mesh.compute_normals(cell_normals=True, point_normals=True, 
                               split_vertices=False, consistent_normals=True, 
                               inplace=True)
            
            # 4. Fill holes wenn vorhanden
            try:
                mesh = mesh.fill_holes(hole_size=1000)
            except:
                pass
                
            return mesh
        except Exception as e:
            print(f"Mesh preparation error: {e}")
            return mesh
    
    def _perform_boolean_operation(self, target, tool, operation):
        """Führt Boolean-Operation mit Fehlerbehandlung durch"""
        import pyvista as pv
        
        try:
            # Versuche zuerst mit PyVista
            if operation == "Cut":
                result = target.boolean_difference(tool)
            elif operation == "Join":
                result = target.boolean_union(tool)
            elif operation == "Intersect":
                result = target.boolean_intersection(tool)
            else:
                return None
            
            if result and result.n_points > 0:
                # Ergebnis aufräumen
                result = result.clean(tolerance=1e-6)
                result.compute_normals(inplace=True)
                return result
                
        except Exception as e:
            print(f"PyVista boolean failed: {e}")
            
            # Fallback: Versuche mit vtkBooleanOperationPolyDataFilter direkt
            try:
                return self._vtk_boolean_fallback(target, tool, operation)
            except Exception as e2:
                print(f"VTK fallback also failed: {e2}")
        
        return None
    
    def _vtk_boolean_fallback(self, target, tool, operation):
        """VTK Boolean als Fallback"""
        import vtk
        import pyvista as pv
        
        # Konvertiere zu VTK
        target_vtk = target.extract_surface()
        tool_vtk = tool.extract_surface()
        
        # VTK Boolean Filter
        boolean = vtk.vtkBooleanOperationPolyDataFilter()
        boolean.SetInputData(0, target_vtk)
        boolean.SetInputData(1, tool_vtk)
        
        if operation == "Cut":
            boolean.SetOperationToDifference()
        elif operation == "Join":
            boolean.SetOperationToUnion()
        elif operation == "Intersect":
            boolean.SetOperationToIntersection()
        
        boolean.SetTolerance(1e-6)
        boolean.Update()
        
        result = pv.wrap(boolean.GetOutput())
        if result.n_points > 0:
            return result.clean()
        return None

    def _update_body_mesh(self, body, pv_mesh):
        points = pv_mesh.points.tolist()
        faces = []
        i = 0
        while i < len(pv_mesh.faces):
            n = pv_mesh.faces[i]
            faces.append(tuple(pv_mesh.faces[i+1 : i+1+n]))
            i += n + 1
        body._mesh_vertices = points
        body._mesh_triangles = faces
        self.viewport_3d.add_body(body.id, body.name, points, faces)

    def eventFilter(self, obj, event):
        if event.type() == 6:  # KeyPress
            k = event.key()
            
            # Tab - fokussiert Input-Felder
            if k == Qt.Key_Tab:
                if self.mode == "sketch":
                    self.sketch_editor.keyPressEvent(event)
                    return True
                if self.viewport_3d.extrude_mode:
                    # Fokussiere das Extrude-Panel
                    self.extrude_panel.height_input.setFocus()
                    self.extrude_panel.height_input.selectAll()
                    return True
            
            # Enter - bestätigt Extrusion
            if k in (Qt.Key_Return, Qt.Key_Enter):
                if self.viewport_3d.extrude_mode and self.viewport_3d.selected_faces:
                    self._on_extrude_confirmed()
                    return True
            
            # Escape - bricht ab
            if k == Qt.Key_Escape:
                if self.viewport_3d.extrude_mode:
                    self._on_extrude_cancelled()
                    return True
                elif self.viewport_3d.plane_select_mode:
                    self.viewport_3d.set_plane_select_mode(False)
                    return True
                elif self.mode == "sketch":
                    self._finish_sketch()
                    return True
            
            # F - Flip Richtung im Extrude-Modus
            if k == Qt.Key_F and self.viewport_3d.extrude_mode:
                self.extrude_panel._flip_direction()
                return True
            
            # Plane Selection Shortcuts
            if self.viewport_3d.plane_select_mode:
                if k in [Qt.Key_1, Qt.Key_T]:
                    self._on_plane_selected('xy')
                    return True
                if k in [Qt.Key_2, Qt.Key_F]:
                    self._on_plane_selected('xz')
                    return True
                if k in [Qt.Key_3, Qt.Key_R]:
                    self._on_plane_selected('yz')
                    return True
            
            # 3D Mode Shortcuts
            if self.mode == "3d" and not self.viewport_3d.extrude_mode:
                if k == Qt.Key_E:
                    self._extrude_dialog()
                    return True
                if k == Qt.Key_S:
                    self._new_sketch()
                    return True

        return False

    def _on_opt_change(self, o, v): pass
    def _edit_feature(self, d): 
        if d[0]=='sketch': self.active_sketch=d[1]; self.sketch_editor.sketch=d[1]; self._set_mode("sketch")
    def _new_project(self): 
        self.document = Document("Projekt1")
        self.browser.set_document(self.document)
        self._set_mode("3d")
    
    def _export_stl(self): 
        """STL Export"""
        if not self.document.bodies:
            QMessageBox.warning(self, tr("Export"), tr("Keine Körper zum Exportieren"))
            return
        
        path, _ = QFileDialog.getSaveFileName(self, tr("STL exportieren"), "", "STL Files (*.stl)")
        if path:
            # TODO: Implementieren
            self.statusBar().showMessage(f"STL Export nach {path} - Coming soon!", 3000)
    
    def _show_about(self):
        """Über-Dialog"""
        QMessageBox.about(self, tr("Über LiteCAD"),
            f"<h2>LiteCAD</h2>"
            f"<p>Version 2.6</p>"
            f"<p>Schlankes parametrisches CAD für 3D-Druck</p>"
            f"<p>© 2024-2025</p>"
            f"<p><b>Features:</b></p>"
            f"<ul>"
            f"<li>2D Sketch mit Constraints</li>"
            f"<li>3D Extrusion</li>"
            f"<li>PyVista/VTK Rendering</li>"
            f"</ul>"
        )
    
    # ==================== FILLET / CHAMFER ====================
    
    def _start_fillet(self):
        """Startet den Fillet-Modus"""
        if not self.document.bodies:
            self.statusBar().showMessage("Kein Body vorhanden für Fillet!", 3000)
            return
        
        self._fillet_mode = "fillet"
        self._fillet_target_body = self.document.bodies[-1]  # Letzter Body
        
        self.fillet_panel.set_mode("fillet")
        self.fillet_panel.reset()
        self.fillet_panel.show_at(self.viewport_3d)
        
        self.viewport_3d.set_edge_select_mode(True)
        self.statusBar().showMessage("Fillet: Wähle Kanten oder gib Radius ein | Enter=OK | Esc=Abbrechen")
    
    def _start_chamfer(self):
        """Startet den Chamfer-Modus"""
        if not self.document.bodies:
            self.statusBar().showMessage("Kein Body vorhanden für Chamfer!", 3000)
            return
        
        self._fillet_mode = "chamfer"
        self._fillet_target_body = self.document.bodies[-1]
        
        self.fillet_panel.set_mode("chamfer")
        self.fillet_panel.reset()
        self.fillet_panel.show_at(self.viewport_3d)
        
        self.viewport_3d.set_edge_select_mode(True)
        self.statusBar().showMessage("Chamfer: Wähle Kanten oder gib Distanz ein | Enter=OK | Esc=Abbrechen")
    
    def _on_fillet_radius_changed(self, radius):
        """Preview für Fillet/Chamfer"""
        # Könnte Preview implementieren
        pass
    
    def _on_fillet_confirmed(self):
        """Fillet/Chamfer anwenden"""
        if not self._fillet_target_body:
            self._on_fillet_cancelled()
            return
        
        radius = self.fillet_panel.get_radius()
        
        try:
            mesh = self.viewport_3d.get_body_mesh(self._fillet_target_body.id)
            if mesh:
                if self._fillet_mode == "fillet":
                    result = self._apply_fillet_to_mesh(mesh, radius)
                else:
                    result = self._apply_chamfer_to_mesh(mesh, radius)
                
                if result and result.n_points > 0:
                    self._update_body_mesh(self._fillet_target_body, result)
                    self.statusBar().showMessage(f"{self._fillet_mode.capitalize()} angewendet: {radius}mm", 3000)
                else:
                    self.statusBar().showMessage(f"{self._fillet_mode.capitalize()} fehlgeschlagen!", 3000)
        except Exception as e:
            print(f"Fillet/Chamfer error: {e}")
            self.statusBar().showMessage(f"Fehler: {e}", 3000)
        
        self._on_fillet_cancelled()
    
    def _on_fillet_cancelled(self):
        """Fillet/Chamfer abbrechen"""
        self.fillet_panel.setVisible(False)
        self.viewport_3d.set_edge_select_mode(False)
        self._fillet_mode = None
        self._fillet_target_body = None
    
    def _apply_fillet_to_mesh(self, mesh, radius):
        """Wendet Fillet auf alle Feature-Kanten eines Meshes an"""
        import pyvista as pv
        import numpy as np
        
        # Versuche Build123d Fillet
        if HAS_BUILD123D and self._fillet_target_body:
            result = self._build123d_fillet(radius)
            if result:
                return result
        
        # PyVista Fallback
        try:
            smoothed = mesh.subdivide(nsub=1, subfilter='loop')
            smoothed = smoothed.smooth(n_iter=int(radius * 10), relaxation_factor=0.1)
            return smoothed.clean()
        except Exception as e:
            print(f"Fillet subdivision failed: {e}")
        
        try:
            decimated = mesh.decimate(0.9)
            smoothed = decimated.smooth(n_iter=50, relaxation_factor=0.1)
            return smoothed.clean()
        except Exception as e:
            print(f"Fillet smooth failed: {e}")
        
        return None
    
    def _build123d_fillet(self, radius):
        """Echtes Fillet mit Build123d"""
        try:
            from build123d import fillet
            
            body = self._fillet_target_body
            if not hasattr(body, '_build123d_solid') or body._build123d_solid is None:
                print("Build123d Fillet: Kein BREP Solid vorhanden")
                return None
            
            solid = body._build123d_solid
            edges = solid.edges()
            
            if not edges:
                print("Build123d Fillet: Keine Kanten gefunden")
                return None
            
            filleted = fillet(edges, radius=radius)
            body._build123d_solid = filleted
            
            mesh_data = filleted.tessellate(tolerance=0.1)
            verts = [(v.X, v.Y, v.Z) for v in mesh_data[0]]
            faces = [tuple(t) for t in mesh_data[1]]
            
            body._mesh_vertices = verts
            body._mesh_triangles = faces
            
            v = np.array(verts, dtype=np.float32)
            f = []
            for face in faces: f.extend([3] + list(face))
            
            result = pv.PolyData(v, np.array(f, dtype=np.int32))
            self.statusBar().showMessage(f"Build123d Fillet: {radius}mm", 2000)
            return result
            
        except Exception as e:
            print(f"Build123d Fillet error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _apply_chamfer_to_mesh(self, mesh, distance):
        """Wendet Chamfer auf Feature-Kanten an"""
        import pyvista as pv
        
        # Versuche Build123d Chamfer
        if HAS_BUILD123D and self._fillet_target_body:
            result = self._build123d_chamfer(distance)
            if result:
                return result
        
        # PyVista Fallback
        try:
            smoothed = mesh.smooth(n_iter=20, relaxation_factor=0.05)
            return smoothed.clean()
        except Exception as e:
            print(f"Chamfer failed: {e}")
        
        return None
    
    def _build123d_chamfer(self, distance):
        """Echtes Chamfer mit Build123d"""
        try:
            from build123d import chamfer
            import numpy as np
            import pyvista as pv
            
            body = self._fillet_target_body
            if not hasattr(body, '_build123d_solid') or body._build123d_solid is None:
                print("Build123d Chamfer: Kein BREP Solid vorhanden")
                return None
            
            solid = body._build123d_solid
            edges = solid.edges()
            
            if not edges:
                return None
            
            chamfered = chamfer(edges, length=distance)
            body._build123d_solid = chamfered
            
            mesh_data = chamfered.tessellate(tolerance=0.1)
            verts = [(v.X, v.Y, v.Z) for v in mesh_data[0]]
            faces = [tuple(t) for t in mesh_data[1]]
            
            body._mesh_vertices = verts
            body._mesh_triangles = faces
            
            v = np.array(verts, dtype=np.float32)
            f = []
            for face in faces: f.extend([3] + list(face))
            
            result = pv.PolyData(v, np.array(f, dtype=np.int32))
            self.statusBar().showMessage(f"Build123d Chamfer: {distance}mm", 2000)
            return result
            
        except Exception as e:
            print(f"Build123d Chamfer error: {e}")
            return None