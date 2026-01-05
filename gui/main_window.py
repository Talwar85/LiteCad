"""
LiteCAD - Unified Main Window
V2.7: Refactored - PyVista required, no OpenGL fallback
"""

import sys
import os
import json
import math
import numpy as np
from loguru import logger


from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QToolBar, QStatusBar, QMenuBar, QMenu, QFileDialog,
    QMessageBox, QSplitter, QFrame, QTreeWidget, QTreeWidgetItem,
    QStackedWidget, QApplication, QDialog, QFormLayout,
    QDoubleSpinBox, QDialogButtonBox, QSpinBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QPushButton, QHeaderView, QComboBox,
    QScrollArea, QGraphicsOpacityEffect
)

from PySide6.QtCore import Qt, Signal, QSize, QTimer, QPointF, QEvent, QPropertyAnimation, QEasingCurve, QObject, QPoint
from PySide6.QtGui import QKeySequence, QAction, QFont, QPainter, QPen, QBrush, QColor, QPolygonF


_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from i18n import tr
from sketcher import Sketch
from modeling import Document, Body, ExtrudeFeature, FilletFeature, ChamferFeature, FeatureType
from modeling.brep_utils import pick_face_by_ray
from gui.sketch_editor import SketchEditor, SketchTool
from gui.tool_panel import ToolPanel, PropertiesPanel
from gui.tool_panel_3d import ToolPanel3D, BodyPropertiesPanel
from gui.browser import ProjectBrowser
from gui.input_panels import ExtrudeInputPanel, FilletChamferPanel, TransformPanel
from gui.viewport_pyvista import PyVistaViewport, HAS_PYVISTA, HAS_BUILD123D

try:
    from ocp_tessellate.tessellator import tessellate
    HAS_OCP_TESSELLATE = True
except ImportError:
    HAS_OCP_TESSELLATE = False
    print("! ocp-tessellate nicht gefunden. Nutze Standard-Tessellierung.")


if not HAS_PYVISTA:
    print("ERROR: PyVista is required! Install with: pip install pyvista pyvistaqt")
    sys.exit(1)
    
    
class NotificationWidget(QFrame):
    """Modernes Overlay für Nachrichten (Robust über PyVista)"""
    def __init__(self, text, level="info", parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)
        
        # --- WICHTIGE FLAGS FÜR SICHTBARKEIT ---
        # 1. Tool & Frameless: Macht es zu einem unabhängigen Overlay ohne Fensterrahmen
        # 2. WindowStaysOnTopHint: Zwingt es über das PyVista/OpenGL Fenster
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        
        # 3. ShowWithoutActivating: Klaut dem Hauptfenster nicht den Fokus (Tastatur geht weiter)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        
        # 4. StyledBackground: Zwingt Qt, den Hintergrund aus dem Stylesheet zu malen! (DER FIX)
        self.setAttribute(Qt.WA_StyledBackground, True)
        
        # Styles basierend auf Level
        colors = {
            "info": "#0078d4",      # Blau
            "success": "#107c10",   # Grün
            "warning": "#ffb900",   # Gelb/Orange
            "error": "#d13438"      # Rot
        }
        accent = colors.get(level, "#0078d4")
        bg_color = "#252526" # Etwas heller als der Hintergrund damit es sich abhebt
        
        # Explizites CSS mit ID-Selektor Simulation
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 1px solid #454545;
                border-left: 5px solid {accent};
                border-radius: 4px;
            }}
            QLabel {{ 
                border: none; 
                background: transparent; 
                color: #f0f0f0; 
                font-family: Segoe UI, sans-serif;
                font-size: 13px;
            }}
            QPushButton {{
                color: #aaa;
                background: transparent;
                border: none;
                font-weight: bold;
            }}
            QPushButton:hover {{ color: white; }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(10)
        
        # Icon
        icons = {"info": "ℹ", "success": "✓", "warning": "⚠", "error": "✕"}
        lbl_icon = QLabel(icons.get(level, "ℹ"))
        lbl_icon.setStyleSheet(f"font-size: 18px; color: {accent}; font-weight: bold; border: none;")
        layout.addWidget(lbl_icon)
        
        # Text
        lbl_text = QLabel(text)
        lbl_text.setWordWrap(True)
        # Wichtig: Background transparent setzen, damit er nicht grau wird
        lbl_text.setStyleSheet("background-color: transparent; border: none;") 
        layout.addWidget(lbl_text, 1)
        
        # Close Button
        btn_close = QPushButton("✕")
        btn_close.setFlat(True)
        btn_close.setFixedSize(24, 24)
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.clicked.connect(self.close_anim)
        layout.addWidget(btn_close)
        
        # Animation Setup
        self.anim = QPropertyAnimation(self, b"pos")
        self.target_pos = None 
        
        # Auto-Close Timer
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.close_anim)
        
        duration = 5000 if level == "error" else 3000
        self.timer.start(duration)
        
    def show_anim(self, target_pos):
        """Startet Slide-In Animation"""
        self.target_pos = target_pos
        start_pos = QPoint(target_pos.x(), target_pos.y() - 20)
        
        self.move(start_pos)
        self.show()
        self.raise_()
        
        self.anim.setDuration(250)
        self.anim.setStartValue(start_pos)
        self.anim.setEndValue(target_pos)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.anim.start()
        
    def close_anim(self):
        """Slide-Out Animation"""
        if not self.target_pos:
            self.close()
            return
            
        end_pos = QPoint(self.target_pos.x(), self.target_pos.y() - 20)
        
        self.anim.setDuration(200)
        self.anim.setStartValue(self.pos())
        self.anim.setEndValue(end_pos)
        self.anim.finished.connect(self.close)
        self.anim.start()

class QtLogHandler(QObject):
    """Verbindet Loguru mit Qt Signals"""
    new_message = Signal(str, str) # Level, Text

    def write(self, message):
        record = message.record
        level = record["level"].name.lower()
        text = record["message"]
        self.new_message.emit(level, text)
        
        
class VectorInputDialog(QDialog):
    def __init__(self, title="Eingabe", labels=("X:", "Y:", "Z:"), defaults=(0.0, 0.0, 0.0), parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        self.inputs = []
        
        for label, default in zip(labels, defaults):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(-99999.0, 99999.0)
            spin.setDecimals(2)
            spin.setValue(default)
            row.addWidget(spin)
            layout.addLayout(row)
            self.inputs.append(spin)
            
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_values(self):
        return [spin.value() for spin in self.inputs]

class BooleanDialog(QDialog):
    """Dialog für Boolesche Operationen: Wähle Target und Tool"""
    def __init__(self, bodies, operation="Cut", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Boolean: {operation}")
        self.bodies = bodies
        layout = QFormLayout(self)
        
        self.cb_target = QComboBox()
        self.cb_tool = QComboBox()
        
        for b in bodies:
            self.cb_target.addItem(b.name, b.id)
            self.cb_tool.addItem(b.name, b.id)
            
        # Standard: Letzter ist Tool, Vorletzter ist Target (typischer Workflow)
        if len(bodies) >= 2:
            self.cb_target.setCurrentIndex(len(bodies)-2)
            self.cb_tool.setCurrentIndex(len(bodies)-1)
            
        layout.addRow("Ziel-Körper (bleibt):", self.cb_target)
        layout.addRow("Werkzeug-Körper (wird benutzt):", self.cb_tool)
        
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)
        
    def get_ids(self):
        return self.cb_target.currentData(), self.cb_tool.currentData()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MashCAD")
        self.setMinimumSize(1400, 900)
        self._setup_logging()
        self.document = Document("Projekt1")
        self.mode = "3d"
        self.active_sketch = None
        self.selected_edges = [] # Liste der Indizes für Fillet
        # Debounce Timer für Viewport Updates (Verhindert Mehrfachaufrufe)
        self.notifications = []
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(50) # 50ms warten
        self._update_timer.timeout.connect(self._update_viewport_all_impl)                                                                   
        self._apply_theme()
        self._create_ui()
        self._create_menus()
        self._connect_signals()
        QApplication.instance().installEventFilter(self)
        self._set_mode("3d")
        self.statusBar().showMessage(tr("Ready"))
        logger.info("Ready. PyVista & Build123d active.")
        
    def _setup_logging(self):
        # Entferne Standard-Logger und füge Qt-Handler hinzu
        logger.remove()
        self.qt_log_handler = QtLogHandler()
        self.qt_log_handler.new_message.connect(self._show_notification)
        
        # Format: Zeit | Level | Nachricht
        logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
        logger.add(self.qt_log_handler.write, format="{message}", level="INFO")

    def _show_notification(self, level, message):
        """Erstellt ein Toast-Popup"""
        # Mapping von Loguru levels
        if level in ["critical", "error"]: style = "error"
        elif level == "warning": style = "warning"
        elif level == "success": style = "success"
        else: style = "info"
        
        # Widget erstellen (Parent ist self, damit es im Fenster bleibt)
        notif = NotificationWidget(message, style, self)
        
        self.notifications.append(notif)
        
        # Position berechnen
        self._reposition_notifications()

    def _cleanup_notification(self, notif):
        if notif in self.notifications:
            self.notifications.remove(notif)
        notif.deleteLater()
        # Nach dem Löschen die anderen aufrücken lassen
        # (Optional, hier vereinfacht lassen wir sie stehen bis sie verschwinden)

    def _reposition_notifications(self):
        """Berechnet Positionen und startet Animationen"""
        top_margin = 90
        spacing = 10
        y_pos = top_margin
        
        # Iteriere über alle aktiven Notifications
        for notif in self.notifications:
            if not notif.isVisible() and not notif.target_pos:
                # Neue Notification (noch nicht animiert)
                # Zentrieren
                x = (self.width() - notif.width()) // 2
                
                # Cleanup Signal verbinden
                notif.anim.finished.connect(
                    lambda n=notif: self._cleanup_notification(n) if n.anim.direction() == QPropertyAnimation.Backward else None
                )
                
                # Animation starten
                notif.show_anim(QPoint(x, y_pos))
            
            elif notif.isVisible():
                # Bereits sichtbare Notifications verschieben wir nicht (einfacher Stack)
                # Oder wir könnten sie hier updaten, wenn sich Fenstergröße ändert
                pass
            
            # Platz für die nächste berechnen
            y_pos += notif.height() + spacing
            
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
        
    def _calculate_plane_axes(self, normal_vec):
        """
        Berechnet stabile X- und Y-Achsen für eine Ebene basierend auf der Normalen.
        Muss IDENTISCH zu viewport_pyvista.py sein!
        """
        n = np.array(normal_vec)
        norm = np.linalg.norm(n)
        if norm == 0: return (1,0,0), (0,1,0)
        n = n / norm
        
        # Globale Up-Vektor Strategie (Z-Up)
        if abs(n[2]) > 0.999:
            # Normale ist (0,0,1) oder (0,0,-1)
            x_dir = np.array([1.0, 0.0, 0.0])
            y_dir = np.cross(n, x_dir)
            y_dir = y_dir / np.linalg.norm(y_dir)
            x_dir = np.cross(y_dir, n)
        else:
            global_up = np.array([0.0, 0.0, 1.0])
            x_dir = np.cross(global_up, n)
            x_dir = x_dir / np.linalg.norm(x_dir)
            y_dir = np.cross(n, x_dir)
            y_dir = y_dir / np.linalg.norm(y_dir)
            
        return tuple(x_dir), tuple(y_dir)
        
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
        
        self.transform_panel = TransformPanel(self)
        self.transform_panel.values_changed.connect(self._on_transform_val_change)
        self.transform_panel.confirmed.connect(self._on_transform_confirmed)
        self.transform_panel.cancelled.connect(self._on_transform_cancelled)
        self._active_transform_body = None
        self._transform_mode = None
        
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
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
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
        self._reposition_notifications()
    
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
        self.browser.visibility_changed.connect(self._trigger_viewport_update)
        
        if hasattr(self.viewport_3d, 'set_body_visibility'):
            self.browser.body_vis_changed.connect(self.viewport_3d.set_body_visibility)
        
        # Viewport Signale
        self.viewport_3d.plane_clicked.connect(self._on_plane_selected)
        if hasattr(self.viewport_3d, 'custom_plane_clicked'):
            self.viewport_3d.custom_plane_clicked.connect(self._on_custom_plane_selected)
            
        self.viewport_3d.extrude_requested.connect(self._on_extrusion_finished)
        self.viewport_3d.height_changed.connect(self._on_viewport_height_changed)
    
    # --- DEBOUNCED UPDATE LOGIC ---
    def _trigger_viewport_update(self):
        """Startet den Timer für das Update (Debounce)"""
        self._update_timer.start() # Reset timer if called again
        
    def _update_viewport_all_impl(self):
        """Das eigentliche Update, wird vom Timer aufgerufen"""
        # Sketches
        self.viewport_3d.set_sketches(self.browser.get_visible_sketches())
        
        # Bodies
        self.viewport_3d.clear_bodies()
        colors = [(0.6,0.6,0.8), (0.8,0.6,0.6), (0.6,0.8,0.6)]
        
        for i, (b, visible) in enumerate(self.browser.get_visible_bodies()):
            if visible and hasattr(b, '_mesh_vertices'):
                # Hier übergeben wir jetzt AUCH Normals und Edges, falls vorhanden!
                self.viewport_3d.add_body(
                    b.id, 
                    b.name, 
                    b._mesh_vertices, 
                    b._mesh_triangles, 
                    color=colors[i%3],
                    normals=getattr(b, '_mesh_normals', None),
                    edges=getattr(b, '_mesh_edges', None)
                )

    def _update_viewport_all(self):
        """Legacy Wrapper: Ruft sofortiges Update auf (wenn nötig) oder Trigger"""
        self._trigger_viewport_update()                                        
    def _on_3d_action(self, action: str):
        """Verarbeitet 3D-Tool-Aktionen"""
        actions = {
            'new_sketch': self._new_sketch,
            'extrude': self._extrude_dialog,
            'export_stl': self._export_stl,
            'export_step': self._export_step,
            'export_dxf': lambda: self._show_not_implemented("DXF Export"),
            'primitive_box': lambda: self._show_not_implemented("Box Primitiv"),
            'primitive_cylinder': lambda: self._show_not_implemented("Zylinder Primitiv"),
            'primitive_sphere': lambda: self._show_not_implemented("Kugel Primitiv"),
            'revolve': lambda: self._show_not_implemented("Revolve"),
            'sweep': lambda: self._show_not_implemented("Sweep"),
            'loft': lambda: self._show_not_implemented("Loft"),
            
            # --- Implementierte Transformationen ---
            'move_body': lambda: self._start_transform_mode("move"),
            'copy_body': self._copy_body,
            'rotate_body': lambda: self._start_transform_mode("rotate"),
            'mirror_body': self._mirror_body,
            'scale_body': lambda: self._start_transform_mode("scale"),
            
            # --- Implementierte Booleans ---
            'boolean_union': lambda: self._boolean_operation_dialog("Union"),
            'boolean_cut': lambda: self._boolean_operation_dialog("Cut"),
            'boolean_intersect': lambda: self._boolean_operation_dialog("Intersect"),
            
            'fillet': self._start_fillet,
            'chamfer': self._start_chamfer,
            
            'export_dxf': lambda: self._show_not_implemented("DXF Export"),
            'primitive_box': lambda: self._show_not_implemented("Box Primitiv"),
            
            'shell': lambda: self._show_not_implemented("Shell"),
            'hole': lambda: self._show_not_implemented("Bohrung"),
            
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
    
    def _get_export_candidates(self):
        """
        Hilfsfunktion: Gibt eine Liste von Bodies zurück.
        Entweder der aktuell selektierte, oder ALLE sichtbaren, wenn nichts selektiert ist.
        """
        selected = self._get_active_body()
        if selected:
            return [selected]
        
        # Wenn nichts selektiert ist, nehmen wir alle sichtbaren
        candidates = []
        for body in self.document.bodies:
            # Check visibility via viewport
            if self.viewport_3d.is_body_visible(body.id):
                candidates.append(body)
        return candidates
        
    def _on_feature_selected(self, data):
        """Wird aufgerufen wenn ein Feature im Tree ausgewählt wird"""
        if data and len(data) >= 2:
            if data[0] == 'body':
                self.body_properties.update_body(data[1])
            else:
                self.body_properties.clear()
    
    def _start_transform_mode(self, mode):
        """Startet den Modus. Wenn kein Body gewählt ist, wartet er auf Klick."""
        body = self._get_active_body()
        
        # Fall 1: Kein Körper gewählt -> Warte auf Klick im Viewport
        if not body:
            self.statusBar().showMessage(f"{mode.capitalize()}: Klicke jetzt auf einen Körper im 3D-Fenster...")
            self._pending_transform_mode = mode 
            self.viewport_3d.setCursor(Qt.CrossCursor)
            return
            
        # Fall 2: Körper ist da -> Los geht's
        self._transform_mode = mode
        self._active_transform_body = body
        self._pending_transform_mode = None
        self.viewport_3d.setCursor(Qt.ArrowCursor)
        
        # Panel anzeigen
        self.transform_panel.set_mode(mode)
        self.transform_panel.show_at(self.viewport_3d)
        
        # Gizmo aktivieren
        if hasattr(self.viewport_3d, 'start_transform'):
            self.viewport_3d.start_transform(body.id, mode)
            
            # --- FIX FÜR DEN FEHLER ---
            # Wir trennen explizit NUR unsere Callback-Methode
            try:
                self.viewport_3d.transform_changed.disconnect(self._on_viewport_transform_update)
            except Exception:
                pass # War noch nicht verbunden, das ist okay.
            
            # Neu verbinden
            self.viewport_3d.transform_changed.connect(self._on_viewport_transform_update)
            
        self.statusBar().showMessage(f"{mode.capitalize()}: Ziehe am Kasten oder gib Werte ein | Enter=OK")

    def _on_transform_val_change(self, x, y, z):
        """Live Update vom Panel -> Viewport Actor"""
        if self._transform_mode:
            self.viewport_3d.apply_transform_values(x, y, z, self._transform_mode)
            
    def _on_viewport_transform_update(self, x, y, z):
        """Live Update vom Viewport Gizmo -> Panel Input Felder"""
        # Werte an das Panel senden, damit die Zahlen sich drehen
        if hasattr(self, 'transform_panel'):
            self.transform_panel.update_values(x, y, z)
        
    def _on_transform_confirmed(self):
        """Finalisieren der Transformation"""
        if not self._active_transform_body: return
        
        # Werte aus dem Panel sind die "Wahrheit"
        vals = self.transform_panel.get_values()
        dx, dy, dz = vals
        
        body = self._active_transform_body
        
        if HAS_BUILD123D and getattr(body, '_build123d_solid', None):
            from build123d import Location, Axis
            
            try:
                if self._transform_mode == "move":
                    # Translation
                    body._build123d_solid = body._build123d_solid.move(Location((dx, dy, dz)))
                    
                elif self._transform_mode == "scale":
                    # Uniform Scale (Panel hat x=y=z bei scale, meistens)
                    # Wir nehmen X als Faktor
                    factor = dx
                    if factor > 0:
                        body._build123d_solid = body._build123d_solid.scale(factor)
                        
                elif self._transform_mode == "rotate":
                    # Euler Rotation (sequentiell)
                    # Build123d rotiert um Achsen
                    solid = body._build123d_solid
                    if dx != 0: solid = solid.rotate(Axis.X, dx)
                    if dy != 0: solid = solid.rotate(Axis.Y, dy)
                    if dz != 0: solid = solid.rotate(Axis.Z, dz)
                    body._build123d_solid = solid
                    
                # Mesh aktualisieren
                self._update_body_from_build123d(body, body._build123d_solid)
                print(f"Transform {self._transform_mode} applied via Build123d")
                
            except Exception as e:
                print(f"Transform Error: {e}")
                
        else:
            # Fallback Mesh-Only (wenn kein Build123d Objekt da ist)
            mesh = self.viewport_3d.get_body_mesh(body.id)
            if mesh:
                if self._transform_mode == "move":
                    mesh.translate((dx, dy, dz), inplace=True)
                elif self._transform_mode == "scale":
                    mesh.scale(dx, inplace=True) # Uniform scale logic needed usually
                elif self._transform_mode == "rotate":
                    mesh.rotate_x(dx, inplace=True)
                    mesh.rotate_y(dy, inplace=True)
                    mesh.rotate_z(dz, inplace=True)
                self._update_body_mesh(body, mesh)

        self._on_transform_cancelled() # Cleanup UI
        self.browser.refresh()

    def _on_transform_cancelled(self):
        self.transform_panel.hide()
        self.viewport_3d.end_transform()
        # Viewport Refresh um visuelle Gizmo-Vorschau zurückzusetzen falls Cancel
        self._update_viewport_all()
        self._active_transform_body = None
        self._transform_mode = None
        
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
        
        # Berechne die lokalen Achsen, damit Viewport und Build123d synchron sind
        x_dir, y_dir = self._calculate_plane_axes(normal)
        
        # Speichere ALLE Orientierungsdaten im Sketch Objekt
        s.plane_origin = origin
        s.plane_normal = normal
        s.plane_x_dir = x_dir # WICHTIG für Build123d
        s.plane_y_dir = y_dir
        
        self.active_sketch = s
        self.sketch_editor.sketch = s
        
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
            
        # WICHTIG: Aktiven Sketch zurücksetzen, damit im 3D-Modus 
        # keine Verwechslung bei der Extrusion passiert!
        self.active_sketch = None
        
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
        """Wird aufgerufen, wenn im Panel OK oder Enter gedrückt wurde"""
        height = self.extrude_panel.get_height()
        operation = self.extrude_panel.get_operation()
        
        # Hole die selektierten Faces vom Viewport
        if hasattr(self.viewport_3d, 'selected_faces') and self.viewport_3d.selected_faces:
            faces = list(self.viewport_3d.selected_faces)
            
            # Starte die eigentliche Extrusion
            self._on_extrusion_finished(faces, height, operation)
            
        else:
            self.statusBar().showMessage("Keine Fläche ausgewählt!")
    
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
        """Erstellt parametrische Extrusionen"""
        if not face_indices or abs(height) < 0.001:
            self._finish_extrusion_ui(success=False)
            return

        sketch_faces_indices = []
        body_faces_data = []
        target_sketch = None

        # Sortiere Selektion: Sketch-Faces oder Body-Faces?
        if hasattr(self.viewport_3d, 'detected_faces'):
            for idx in face_indices:
                if 0 <= idx < len(self.viewport_3d.detected_faces):
                    face = self.viewport_3d.detected_faces[idx]
                    if face.get('type') == 'body_face':
                        body_faces_data.append(face)
                    else:
                        sketch_faces_indices.append(idx)
                        if target_sketch is None: target_sketch = face.get('sketch')

        if HAS_BUILD123D:
            success = False
            
            # FALL A: Sketch Extrusion (Parametrisch)
            if sketch_faces_indices and target_sketch:
                try:
                    # Sammle Zentrumspunkte der gewählten Profile als Selektor
                    selector_points = []
                    for idx in sketch_faces_indices:
                        f = self.viewport_3d.detected_faces[idx]
                        selector_points.append(f.get('center_2d'))

                    # Erstelle FEATURE Objekt
                    feature = ExtrudeFeature(
                        sketch=target_sketch,
                        distance=height,
                        operation=operation,
                        selector=selector_points 
                    )
                    
                    target_body = self._get_active_body()
                    if operation == "New Body" or not target_body:
                        target_body = self.document.new_body()
                        feature.operation = "New Body"

                    # Hinzufügen triggert Rebuild
                    target_body.add_feature(feature)
                    
                    # Update Visuals
                    self._update_body_mesh(target_body, None)
                    success = True
                        
                except Exception as e:
                    logger.error(f"BREP Extrusion Error: {e}")

            # FALL B: Body Face (Push/Pull)
            elif body_faces_data:
                 for face_data in body_faces_data:
                      if self._extrude_body_face_build123d(face_data, height, operation):
                           success = True

            if success:
                self._finish_extrusion_ui(success=True, msg=f"Extrusion ({operation}) erstellt.")
                return

        # Fallback
        self._finish_extrusion_ui(success=False, msg="Extrusion fehlgeschlagen.")

    def _finish_extrusion_ui(self, success=True, msg=""):
        """Hilfsfunktion zum Aufräumen der UI"""
        self.extrude_panel.setVisible(False)
        self.viewport_3d.set_extrude_mode(False)
        self.viewport_3d.set_all_bodies_visible(True)
        if success:
            self.browser.refresh()
            if msg: self.statusBar().showMessage(msg)
        
        
    def _extrude_body_face_build123d(self, face_data, height, operation):
        """
        Robuste Extrusion einer Body-Fläche (Push/Pull) mit Scoring-System.
        Findet die passende CAD-Fläche auch bei Ungenauigkeiten.
        """
        try:
            body_id = face_data.get('body_id')
            target_body = next((b for b in self.document.bodies if b.id == body_id), None)
            
            if not target_body:
                print("Fehler: Body ID nicht gefunden.")
                return False
                
            if not hasattr(target_body, '_build123d_solid') or target_body._build123d_solid is None:
                print(f"WARNUNG: Körper '{target_body.name}' hat keine BREP-Daten.")
                return False

            from build123d import Vector, extrude, Face
            
            # Daten aus dem Viewport
            mesh_center = Vector(face_data['center_3d'])
            mesh_normal = Vector(face_data['normal'])
            
            print(f"Suche CAD-Face für Mesh-Center {mesh_center}...")

            # --- SCORING SYSTEM ---
            best_face = None
            best_score = -float('inf')
            
            for face in target_body._build123d_solid.faces():
                try:
                    cad_center = face.center()
                    cad_normal = face.normal_at(cad_center)
                    
                    align = cad_normal.dot(mesh_normal)
                    if align < 0.8: continue 
                    
                    dist = (cad_center - mesh_center).length
                    score = (align * 1000) - dist
                    
                    if face.is_inside(mesh_center):
                        score += 5000 
                    
                    if score > best_score:
                        best_score = score
                        best_face = face
                except Exception:
                    continue
            
            if best_face is None:
                print("Keine passende CAD-Fläche gefunden.")
                return False
                
            # --- EXTRUSION ---
            new_geo = extrude(best_face, amount=height)
            
            # --- BOOLEAN / NEW BODY LOGIC ---
            final_solid = None
            
            if operation == "New Body":
                # FIX 1: Nutze Standard new_body() ohne Namen, damit er "Body X" heißt
                new_body = self.document.new_body() 
                
                # FIX 2: Füge ein Feature hinzu, damit der Browser es gleich darstellt
                # Wir haben keinen Sketch, daher sketch=None, aber wir setzen die Werte
                from modeling import ExtrudeFeature
                feat = ExtrudeFeature(
                    sketch=None, 
                    distance=height, 
                    operation="New Body", 
                    name="Extrude (Face)"
                )
                new_body.features.append(feat)
                
                new_body._build123d_solid = new_geo
                self._update_body_from_build123d(new_body, new_geo)
                return True
                
            elif operation == "Join":
                final_solid = target_body._build123d_solid + new_geo
                # Optional: Auch hier ein Feature hinzufügen für History? 
                # Das ist komplizierter bei bestehenden Bodies, lassen wir erstmal.
            elif operation == "Cut":
                final_solid = target_body._build123d_solid - new_geo
            elif operation == "Intersect":
                final_solid = target_body._build123d_solid & new_geo
            
            if final_solid is not None:
                target_body._build123d_solid = final_solid
                self._update_body_from_build123d(target_body, final_solid)
                print("BREP Update erfolgreich.")
                return True
            
            return False
            
        except Exception as e:
            print(f"CRITICAL ERROR in _extrude_body_face_build123d: {e}")
            import traceback
            traceback.print_exc()
            return False
            
            
    def _extrude_with_build123d(self, face_indices, height, operation="New Body"):
        try:
            # 1. Solid erstellen
            solid, verts, faces = self.sketch_editor.get_build123d_part(height, operation)
            
            if solid is None or not verts:
                print("Build123d: Keine Geometrie erzeugt.")
                return False

            # 2. Neuen Body im Dokument anlegen
            new_body = self.document.new_body(f"Body {len(self.document.bodies)+1}")
            
            # 3. Daten zuweisen
            new_body._build123d_solid = solid 
            new_body._mesh_vertices = verts
            new_body._mesh_triangles = faces
            
            # 4. Im Viewport anzeigen
            self.viewport_3d.add_body(new_body.id, new_body.name, verts, faces)
            
            # 5. WICHTIG: Browser aktualisieren, damit der Body in der Liste erscheint!
            if hasattr(self, 'browser'):
                self.browser.refresh()  # <--- DIESE ZEILE HAT GEFEHLT
            
            print(f"Extrusion erfolgreich. Solid gespeichert.")
            return True
            
        except Exception as e:
            print(f"Build123d Extrude Error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    
    
    def _update_body_from_build123d(self, body, solid):
        """Generiert Mesh mit korrekter Kanten-Extraktion für OCP"""
        # print(f"DEBUG: _update_body_from_build123d START für '{body.name}'", flush=True)
        
        try:
            success = False
            
            # Speicher initialisieren
            body._mesh_vertices = []
            body._mesh_triangles = []
            body._mesh_normals = []
            body._mesh_edges = []      # Für Indizes (Standard)
            body._mesh_edge_lines = [] # NEU: Für Koordinaten (OCP)
            
            body._build123d_solid = solid
            
            shape = solid
            if hasattr(solid, 'wrapped'): shape = solid.wrapped

            # --- OPTION A: OCP Tessellate ---
            if HAS_OCP_TESSELLATE:
                try:
                    from ocp_tessellate.tessellator import tessellate
                    
                    result = tessellate(
                        shape, f"{id(shape)}", 0.1, quality=0.1,
                        angular_tolerance=0.2, compute_faces=True, compute_edges=True, debug=False
                    )
                    
                    # Entpacken
                    verts_flat = None; tris_flat = None; norms_flat = None; edges_flat = None
                    if isinstance(result, dict):
                        verts_flat = result.get("vertices")
                        tris_flat = result.get("triangles")
                        norms_flat = result.get("normals")
                        edges_flat = result.get("edges")
                    elif isinstance(result, tuple):
                        if len(result) >= 3: verts_flat, tris_flat, norms_flat = result[0], result[1], result[2]
                        if len(result) >= 4: edges_flat = result[3]
                    
                    import numpy as np

                    # 1. Vertices
                    if verts_flat is not None:
                        if isinstance(verts_flat, np.ndarray): body._mesh_vertices = verts_flat.reshape(-1, 3).tolist()
                        else: body._mesh_vertices = [tuple(verts_flat[i:i+3]) for i in range(0, len(verts_flat), 3)]
                    
                    # 2. Normals
                    if norms_flat is not None:
                        if isinstance(norms_flat, np.ndarray): body._mesh_normals = norms_flat.reshape(-1, 3).tolist()
                        else: body._mesh_normals = [tuple(norms_flat[i:i+3]) for i in range(0, len(norms_flat), 3)]

                    # 3. Triangles
                    if tris_flat is not None:
                        if isinstance(tris_flat, np.ndarray): body._mesh_triangles = tris_flat.reshape(-1, 3).tolist()
                        else: body._mesh_triangles = [tuple(tris_flat[i:i+3]) for i in range(0, len(tris_flat), 3)]
                        
                    # 4. Edges (FIX: Als Koordinaten behandeln!)
                    if edges_flat is not None and len(edges_flat) > 0:
                        # OCP liefert Segmente: [P1x, P1y, P1z, P2x, P2y, P2z, ...]
                        if isinstance(edges_flat, np.ndarray):
                            body._mesh_edge_lines = edges_flat.reshape(-1, 3).tolist()
                        else:
                            body._mesh_edge_lines = [tuple(edges_flat[i:i+3]) for i in range(0, len(edges_flat), 3)]
                    
                    success = True

                except Exception as e:
                    print(f"DEBUG: OCP Tessellation Exception: {e}", flush=True)
            
            # --- OPTION B: Fallback ---
            if not success:
                mesh = solid.tessellate(tolerance=0.1)
                body._mesh_vertices = [(v.X, v.Y, v.Z) for v in mesh[0]]
                body._mesh_triangles = [tuple(t) for t in mesh[1]]
            
            # --- VIEWPORT UPDATE ---
            if not body._mesh_vertices or not body._mesh_triangles:
                return

            self.viewport_3d.add_body(
                body.id, 
                body.name, 
                body._mesh_vertices, 
                body._mesh_triangles, 
                color=getattr(body, 'color', None),
                normals=getattr(body, '_mesh_normals', None),
                edges=getattr(body, '_mesh_edges', None),          # Indizes
                edge_lines=getattr(body, '_mesh_edge_lines', None) # NEU: Koordinaten
            )
            
        except Exception as e:
            print(f"DEBUG: MESHING ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()

    def _show_extrude_input_dialog(self):
        """Legacy Dialog - wird durch Panel ersetzt"""
        # Falls Tab gedrückt wird, fokussiere das Panel
        self.extrude_panel.height_input.setFocus()
        self.extrude_panel.height_input.selectAll()
    
    def _on_3d_click(self, event):
        """Kernel-Level Selektion: Klickt auf das mathematische Modell"""
        # 1. Klick-Position
        pos = event.position() if hasattr(event, 'position') else event.pos()
        x, y = int(pos.x()), int(pos.y())
        
        # 2. Strahl holen (aus Viewport Helper)
        ray_origin, ray_dir = self.viewport_3d.get_ray_from_click(x, y)
        
        best_face = None
        best_body = None
        min_dist = float('inf')
        
        # 3. Durchlaufe alle CAD-Bodies im Dokument
        for body in self.document.bodies:
            if hasattr(body, '_build123d_solid') and body._build123d_solid:
                # Kernel fragen!
                face, dist = pick_face_by_ray(body._build123d_solid, ray_origin, ray_dir)
                
                if face and dist < min_dist:
                    min_dist = dist
                    best_face = face
                    best_body = body

        # 4. Ergebnis verarbeiten
        if best_face:
            self.statusBar().showMessage(f"BREP: {best_face.geom_type} auf {best_body.name} selektiert")
            
            # Highlighten (hier nutzen wir einen Trick: Wir erstellen ein temporäres Mesh NUR für die Face)
            # Das ist viel schneller als den ganzen Body neu zu meshen.
            self.selected_brep_face = best_face
            self.selected_body = best_body
            
            # Optional: Sende das an den Viewport zur Anzeige
            # self.viewport_3d.highlight_brep_face(best_face) # Müsste im Viewport implementiert werden
            
        else:
            self.statusBar().showMessage("Nichts getroffen")
            self.selected_brep_face = None
        
        
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
        pass
    
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

    def _update_body_mesh(self, body, mesh_override=None):
        """Lädt die Mesh-Daten aus dem Body-Objekt in den Viewport"""
        
        # Wenn wir manuelles Mesh übergeben (Legacy Fallback)
        if mesh_override:
             points = mesh_override.points.tolist()
             faces = []
             i = 0
             while i < len(mesh_override.faces):
                n = mesh_override.faces[i]
                faces.append(tuple(mesh_override.faces[i+1 : i+1+n]))
                i += n + 1
             body._mesh_vertices = points
             body._mesh_triangles = faces
        
        # Normale Route: Daten aus dem Body nehmen (wurden von _rebuild berechnet)
        if body._mesh_vertices and body._mesh_triangles:
             # Farbe bestimmen
             col_idx = self.document.bodies.index(body) % 3
             colors = [(0.6,0.6,0.8), (0.8,0.6,0.6), (0.6,0.8,0.6)]
             
             self.viewport_3d.add_body(
                 body.id, 
                 body.name, 
                 body._mesh_vertices, 
                 body._mesh_triangles,
                 color=colors[col_idx]
             )

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

        # Click handling for direct Body Selection in 3D
        if obj == self.viewport_3d and event.type() == QEvent.MouseButtonPress:
             if self.mode == "3d" and not self.viewport_3d.extrude_mode:
                 # Nur wenn wir nicht in einem anderen Modus sind
                 if not hasattr(self, '_fillet_mode') or not self._fillet_mode:
                     self._on_3d_click(event)
                     pos = event.position() if hasattr(event, 'position') else event.pos()
                     
                     # Prüfen ob ein Body geklickt wurde
                     if hasattr(self.viewport_3d, 'select_body_at'):
                         bid = self.viewport_3d.select_body_at(pos.x(), pos.y())
                         
                         if bid:
                             body = next((b for b in self.document.bodies if b.id == bid), None)
                             if body:
                                 # Selektiere den Body im Browser (optisch)
                                 # self.browser.select_body(body) # TODO
                                 
                                 # WICHTIG: Wenn wir auf Move geklickt haben (pending), starten wir jetzt!
                                 if hasattr(self, '_pending_transform_mode') and self._pending_transform_mode:
                                     # Aktiven Body setzen, damit _start_transform_mode ihn findet
                                     # Da _get_active_body auf Browser schaut, müssen wir tricksen oder Browser updaten
                                     # Wir setzen ihn hier direkt temporär als aktiv
                                     self._active_transform_body = body 
                                     
                                     # Modus starten
                                     self._start_transform_mode(self._pending_transform_mode)
                                 else:
                                     # Nur selektieren (Normaler Klick)
                                     self._active_transform_body = body
                                     self.statusBar().showMessage(f"Körper '{body.name}' selektiert")
        return False

    def _on_opt_change(self, o, v): pass
    def _edit_feature(self, d): 
        if d[0]=='sketch': self.active_sketch=d[1]; self.sketch_editor.sketch=d[1]; self._set_mode("sketch")
    def _new_project(self): 
        self.document = Document("Projekt1")
        self.browser.set_document(self.document)
        self._set_mode("3d")
    
    def _export_stl(self): 
        """STL Export mit Loguru statt QMessageBox"""
        bodies = self._get_export_candidates()
        if not bodies:
            logger.warning("Keine sichtbaren Körper zum Exportieren.")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, tr("STL exportieren"), "", "STL Files (*.stl)")
        if not path: return

        try:
            import pyvista as pv
            merged_polydata = None
            
            for body in bodies:
                mesh_to_add = None
                if HAS_BUILD123D and hasattr(body, '_build123d_solid') and body._build123d_solid:
                    try:
                        b3d_mesh = body._build123d_solid.tessellate(tolerance=0.01)
                        verts = [(v.X, v.Y, v.Z) for v in b3d_mesh[0]]
                        faces = []
                        for t in b3d_mesh[1]: faces.extend([3] + list(t))
                        import numpy as np
                        mesh_to_add = pv.PolyData(np.array(verts), np.array(faces))
                    except Exception as e:
                        logger.warning(f"Build123d Tessellierung fehlgeschlagen, nutze Fallback: {e}")
                
                if mesh_to_add is None:
                    mesh_to_add = self.viewport_3d.get_body_mesh(body.id)
                
                if mesh_to_add:
                    if merged_polydata is None: merged_polydata = mesh_to_add
                    else: merged_polydata = merged_polydata.merge(mesh_to_add)

            if merged_polydata:
                merged_polydata.save(path)
                logger.success(f"STL gespeichert: {path}")
            else:
                logger.error("Konnte keine Mesh-Daten generieren.")

        except Exception as e:
            logger.error(f"STL Export Fehler: {e}")

    def _export_step(self):
        """STEP Export mit FIX für 'Part object has no attribute export_step'"""
        if not HAS_BUILD123D:
            logger.error("Build123d fehlt. STEP Export nicht möglich.")
            return

        bodies = self._get_export_candidates()
        if not bodies:
            logger.warning("Keine sichtbaren Körper zum Exportieren.")
            return
            
        valid_solids = []
        for b in bodies:
            if hasattr(b, '_build123d_solid') and b._build123d_solid:
                valid_solids.append(b._build123d_solid)
                
        if not valid_solids:
            logger.error("Keine CAD-Daten (BREP) vorhanden. Nur Mesh-Objekte können nicht als STEP exportiert werden.")
            return

        path, _ = QFileDialog.getSaveFileName(self, tr("STEP exportieren"), "", "STEP Files (*.step *.stp)")
        if not path: return
        
        try:
            from build123d import Compound, export_step
            
            # WICHTIG: Wir nutzen die Funktion export_step(obj, path), nicht die Methode!
            if len(valid_solids) == 1:
                export_shape = valid_solids[0]
            else:
                export_shape = Compound(children=valid_solids)
            
            # DER FIX: Globale Funktion nutzen
            export_step(export_shape, path)
            
            logger.success(f"STEP gespeichert: {path}")
                
        except Exception as e:
            logger.error(f"STEP Export Fehler: {e}")
            import traceback
            traceback.print_exc()
    
    def _show_not_implemented(self, feature: str):
        logger.info(f"{feature} - Coming soon!")
        
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
        body = self._get_active_body()
        if not body:
            self.statusBar().showMessage("Bitte Körper auswählen!")
            return
            
        if not hasattr(body, '_build123d_solid'):
             self.statusBar().showMessage("Warnung: Nur Mesh-Daten.")
             
        self.selected_edges = []
        self._fillet_mode = "fillet"  # <--- WICHTIG: Wir merken uns den Modus hier
        
        self.viewport_3d.clear_highlight()
        
        # Signal sauber trennen (verhindert Mehrfach-Verbindungen)
        try: self.viewport_3d.clicked_3d_point.disconnect()
        except: pass
        
        self.viewport_3d.clicked_3d_point.connect(self._on_fillet_click)
        
        self.statusBar().showMessage(f"Fillet: Klicke auf Kanten von '{body.name}'...")
        self.fillet_panel.set_target_body(body)
        self.fillet_panel.set_mode("fillet")
        self.fillet_panel.show_at(self.viewport_3d)
    
    def _start_chamfer(self):
        body = self._get_active_body()
        if not body: return
        
        self.selected_edges = []
        self._fillet_mode = "chamfer" # <--- WICHTIG: Wir merken uns den Modus hier
        
        self.viewport_3d.clear_highlight()
        
        try: self.viewport_3d.clicked_3d_point.disconnect()
        except: pass
        
        self.viewport_3d.clicked_3d_point.connect(self._on_fillet_click)
        
        self.fillet_panel.set_target_body(body)
        self.fillet_panel.set_mode("chamfer")
        self.fillet_panel.show_at(self.viewport_3d)
        
    def _on_fillet_click(self, body_id, pos):
        """Findet die Kante in der Nähe des Klicks und markiert sie"""
        body = self.fillet_panel.get_target_body()
        if not body or body.id != body_id: return
        
        # Wir brauchen Vector für Distanzberechnung
        from build123d import Vector
        click_pt = Vector(pos)
        
        if not hasattr(body, '_build123d_solid') or not body._build123d_solid:
            return

        all_edges = body._build123d_solid.edges()
        
        best_dist = float('inf')
        best_edge_idx = -1
        
        # Finde Kante mit geringstem Abstand zum Klick
        for i, edge in enumerate(all_edges):
            try:
                # center() ist der Mittelpunkt der Kante
                dist = (edge.center() - click_pt).length
                if dist < best_dist: 
                    best_dist = dist
                    best_edge_idx = i
            except: pass
            
        # Toleranz (15mm Kugel um Klick)
        if best_dist < 15.0 and best_edge_idx != -1:
            if best_edge_idx in self.selected_edges:
                self.selected_edges.remove(best_edge_idx)
            else:
                self.selected_edges.append(best_edge_idx)
                
            # Visualisierung aktualisieren
            self.viewport_3d.clear_highlight()
            for idx in self.selected_edges:
                if idx < len(all_edges):
                    edge = all_edges[idx]
                    # Linie zeichnen zur Markierung
                    try:
                        p1 = edge.position_at(0)
                        p2 = edge.position_at(1)
                        self.viewport_3d.highlight_edge((p1.X, p1.Y, p1.Z), (p2.X, p2.Y, p2.Z))
                    except: pass

    def _on_fillet_confirmed(self):
        """Erstellt das parametrische Feature"""
        radius = self.fillet_panel.get_radius()
        body = self.fillet_panel.get_target_body()
        
        try:
            selectors = []
            # Prüfen ob wir CAD Daten haben
            if hasattr(body, '_build123d_solid') and body._build123d_solid:
                all_edges = body._build123d_solid.edges()
                
                if not self.selected_edges:
                    # Wenn nichts gewählt: Warnung
                     res = QMessageBox.question(self, "Alles?", "Keine Kanten gewählt. Ganzen Körper bearbeiten?")
                     if res != QMessageBox.Yes: return
                     selectors = None # None bedeutet "Alle Kanten"
                else:
                     # Wir speichern Punkte im Raum (Mittelpunkte der Kanten)
                     # Das ist robuster als Indizes, wenn sich das Modell ändert
                     for idx in self.selected_edges:
                         if idx < len(all_edges):
                             c = all_edges[idx].center()
                             selectors.append((c.X, c.Y, c.Z))
            
            # HIER WAR DER FEHLER: Wir nutzen jetzt die Variable aus dem MainWindow
            mode = getattr(self, '_fillet_mode', 'fillet') 
            
            if mode == "chamfer":
                feat = ChamferFeature(distance=radius, edge_selectors=selectors)
            else:
                feat = FilletFeature(radius=radius, edge_selectors=selectors)
            
            # Feature zum Body hinzufügen -> Das triggert body._rebuild()
            # Der Body kümmert sich jetzt um Error-Handling und Smart-Retries
            body.add_feature(feat)
            
            # Mesh visualisieren
            self._update_body_mesh(body, None)
            
            self.fillet_panel.hide()
            self.viewport_3d.clear_highlight()
            self.browser.refresh()
            logger.success(f"{mode.capitalize()} Feature erstellt.")
            
        except Exception as e:
            logger.error(f"Feature Creation Error: {e}")
            import traceback
            traceback.print_exc()

            
    def _on_fillet_radius_changed(self, radius):
        """Callback wenn der Slider bewegt wird (optional für Preview)"""
        # Aktuell leer, verhindert aber den Crash
        pass
        
        
        
            
    def _on_fillet_cancelled(self):
        self.fillet_panel.setVisible(False)
        self.viewport_3d.clear_highlight()
        try: self.viewport_3d.clicked_3d_point.disconnect()
        except: pass
        
    
    
    
            
    # ==================== 3D OPERATIONEN ====================

    def _get_active_body(self):
        """Hilfsfunktion: Gibt den aktuell im Browser ausgewählten Body zurück"""
        items = self.browser.tree.selectedItems()
        if not items: return None
        # Wir suchen die ID aus dem Tree-Item
        # (Dies hängt von Ihrer Browser-Implementierung ab, hier eine generische Lösung)
        # Wenn Browser selection ein Body-Objekt liefert:
        if hasattr(self.browser, 'get_selected_body'):
            return self.browser.get_selected_body()
        
        # Fallback: Letzten Body nehmen
        if self.document.bodies:
            return self.document.bodies[-1]
        return None

    def _move_body(self):
        body = self._get_active_body()
        if not body: return self.statusBar().showMessage("Kein Körper ausgewählt!")
        
        dlg = VectorInputDialog("Verschieben", ("X (mm):", "Y (mm):", "Z (mm):"), (0,0,0), self)
        if dlg.exec():
            dx, dy, dz = dlg.get_values()
            if dx==0 and dy==0 and dz==0: return
            
            # 1. Build123d Transformation (für saubere Historie)
            if HAS_BUILD123D and hasattr(body, '_build123d_solid') and body._build123d_solid:
                from build123d import Location
                body._build123d_solid = body._build123d_solid.move(Location((dx, dy, dz)))
                # Mesh neu generieren
                self._update_body_from_build123d(body, body._build123d_solid)
            else:
                # 2. PyVista Mesh Transformation (Fallback)
                mesh = self.viewport_3d.get_body_mesh(body.id)
                if mesh:
                    mesh.translate((dx, dy, dz), inplace=True)
                    self._update_body_mesh(body, mesh)
            
            self.browser.refresh()
            self.statusBar().showMessage(f"Körper verschoben: {dx}, {dy}, {dz}")

    def _scale_body(self):
        body = self._get_active_body()
        if not body: return self.statusBar().showMessage("Kein Körper ausgewählt!")
        
        dlg = VectorInputDialog("Skalieren", ("Faktor:",), (1.0,), self)
        if dlg.exec():
            s = dlg.get_values()[0]
            if s == 1.0 or s <= 0: return

            if HAS_BUILD123D and hasattr(body, '_build123d_solid') and body._build123d_solid:
                body._build123d_solid = body._build123d_solid.scale(s)
                self._update_body_from_build123d(body, body._build123d_solid)
            else:
                mesh = self.viewport_3d.get_body_mesh(body.id)
                if mesh:
                    mesh.scale(s, inplace=True)
                    self._update_body_mesh(body, mesh)
            self.browser.refresh()

    def _rotate_body(self):
        body = self._get_active_body()
        if not body: return self.statusBar().showMessage("Kein Körper ausgewählt!")
        
        # Einfacher Dialog: Achse + Winkel
        dlg = VectorInputDialog("Rotieren", ("X-Achse (0/1):", "Y-Achse (0/1):", "Z-Achse (0/1):", "Winkel (°):"), (0,0,1,90), self)
        # Hack: Label 4 ist Winkel. Wir nutzen VectorInputDialog generisch.
        if dlg.exec():
            ax, ay, az, angle = dlg.get_values()
            if angle == 0: return
            
            axis = (ax, ay, az)
            
            if HAS_BUILD123D and hasattr(body, '_build123d_solid') and body._build123d_solid:
                from build123d import Axis, Location
                # Rotation um Zentrum oder Ursprung? Hier Ursprung (einfacher)
                # Für Rotation um Objektzentrum müsste man BoundingBox Center berechnen
                body._build123d_solid = body._build123d_solid.rotate(Axis((0,0,0), axis), angle)
                self._update_body_from_build123d(body, body._build123d_solid)
            else:
                mesh = self.viewport_3d.get_body_mesh(body.id)
                if mesh:
                    mesh.rotate_vector(vector=axis, angle=angle, inplace=True)
                    self._update_body_mesh(body, mesh)
            self.browser.refresh()

    def _copy_body(self):
        body = self._get_active_body()
        if not body: return
        
        import copy
        # Neuen Body erstellen
        new_b = self.document.new_body(f"{body.name}_Kopie")
        
        # Daten kopieren
        if hasattr(body, '_mesh_vertices'):
            new_b._mesh_vertices = copy.deepcopy(body._mesh_vertices)
            new_b._mesh_triangles = copy.deepcopy(body._mesh_triangles)
            
        if hasattr(body, '_build123d_solid') and body._build123d_solid:
             new_b._build123d_solid = copy.deepcopy(body._build123d_solid)
        
        # Anzeigen
        self.viewport_3d.add_body(new_b.id, new_b.name, new_b._mesh_vertices, new_b._mesh_triangles)
        self.browser.refresh()
        self.statusBar().showMessage(f"Kopie erstellt: {new_b.name}")

    def _mirror_body(self):
        body = self._get_active_body()
        if not body: return

        # Einfachheitshalber Mirror an XZ Ebene (Y spiegeln)
        # In Zukunft könnte man Plane Selection machen
        if HAS_BUILD123D and hasattr(body, '_build123d_solid') and body._build123d_solid:
            from build123d import Plane
            # Mirror an XZ Plane (Normal Y)
            body._build123d_solid = body._build123d_solid.mirror(Plane.XZ)
            self._update_body_from_build123d(body, body._build123d_solid)
        else:
            mesh = self.viewport_3d.get_body_mesh(body.id)
            if mesh:
                mesh.reflect((0,1,0), point=(0,0,0), inplace=True)
                self._update_body_mesh(body, mesh)
        self.browser.refresh()

    def _boolean_operation_dialog(self, op_type="Cut"):
        """Führt Union, Cut oder Intersect aus"""
        if len(self.document.bodies) < 2:
            QMessageBox.warning(self, "Fehler", "Mindestens 2 Körper benötigt!")
            return

        dlg = BooleanDialog(self.document.bodies, op_type, self)
        if dlg.exec():
            tid, tool_id = dlg.get_ids()
            if tid == tool_id: return
            
            target = next(b for b in self.document.bodies if b.id == tid)
            tool = next(b for b in self.document.bodies if b.id == tool_id)
            
            success = False
            
            # 1. Versuch: Build123d (Exakt)
            if HAS_BUILD123D:
                try:
                    s1 = getattr(target, '_build123d_solid', None)
                    s2 = getattr(tool, '_build123d_solid', None)
                    
                    if s1 and s2:
                        new_solid = None
                        if op_type == "Union": new_solid = s1 + s2
                        elif op_type == "Cut": new_solid = s1 - s2
                        elif op_type == "Intersect": new_solid = s1 & s2
                        
                        if new_solid:
                            target._build123d_solid = new_solid
                            self._update_body_from_build123d(target, new_solid)
                            
                            # Tool löschen oder verstecken? Meistens löschen bei Boolean
                            # Wir verstecken es erstmal
                            self.viewport_3d.set_body_visibility(tool.id, False)
                            success = True
                except Exception as e:
                    print(f"Build123d Boolean Error: {e}")

            # 2. Versuch: PyVista Mesh Boolean (Fallback)
            if not success:
                m1 = self.viewport_3d.get_body_mesh(target.id)
                m2 = self.viewport_3d.get_body_mesh(tool.id)
                
                if m1 and m2:
                    res = self._perform_boolean_operation(m1, m2, op_type)
                    if res:
                        self._update_body_mesh(target, res)
                        self.viewport_3d.set_body_visibility(tool.id, False)
                        success = True
            
            if success:
                self.statusBar().showMessage(f"Boolean {op_type} erfolgreich.")
                self.browser.refresh()
            else:
                QMessageBox.warning(self, "Fehler", "Operation fehlgeschlagen (Geometrie Fehler).")