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
from gui.log_panel import LogPanel
from modeling.brep_utils import find_closest_face
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
        self.selection_mode = "all" 
        self.statusBar().showMessage(tr("Ready"))
        logger.info("Ready. PyVista & Build123d active.")
        
    def _setup_logging(self):
        logger.remove()
        self.qt_log_handler = QtLogHandler()
        # Verbinde Signal mit neuer zentraler Log-Methode
        self.qt_log_handler.new_message.connect(self._handle_log_message)
        
        # Konsole (stderr) für Debugging
        logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
        # Qt Handler für GUI
        logger.add(self.qt_log_handler.write, format="{message}", level="INFO")

    def _handle_log_message(self, level, message):
        """
        Zentrale Stelle für alle Nachrichten.
        1. Fügt IMMER einen Eintrag ins Log-Panel hinzu.
        2. Zeigt NUR bei Success/Error/Warning ein Overlay an.
        """
        # 1. Ins persistente Log schreiben
        self.log_panel.add_message(level, message)
        
        # 2. Overlay Entscheidung
        show_overlay = False
        if level in ["success", "error", "critical"]:
            show_overlay = True
        elif level == "warning":
            # Warnings auch zeigen, aber vielleicht kürzer (optional)
            show_overlay = True
            
        if show_overlay:
            self._show_toast_overlay(level, message)
            
    def _show_toast_overlay(self, level, message):
        """Erstellt das Toast-Popup (ehemals _show_notification)"""
        # Mapping von Loguru levels für Style
        if level in ["critical", "error"]: style = "error"
        elif level == "warning": style = "warning"
        elif level == "success": style = "success"
        else: style = "info"
        
        # Widget erstellen
        notif = NotificationWidget(message, style, self)
        self.notifications.append(notif)
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
        left_container = QWidget()  # <--- Hier heißt die Variable "left_container"
        left_layout = QHBoxLayout(left_container)
        left_layout.setContentsMargins(0,0,0,0)
        left_layout.setSpacing(0)

        # 1. Spalte: Browser (oben) + Log (unten)
        browser_log_splitter = QSplitter(Qt.Vertical)
        browser_log_splitter.setHandleWidth(1)
        
        self.browser = ProjectBrowser()
        self.browser.set_document(self.document)
        browser_log_splitter.addWidget(self.browser)
        
        self.log_panel = LogPanel() # Log Panel Instanz
        browser_log_splitter.addWidget(self.log_panel)
        
        # Verhältnisse setzen (Browser groß, Log klein)
        browser_log_splitter.setStretchFactor(0, 3)
        browser_log_splitter.setStretchFactor(1, 1)

        left_layout.addWidget(browser_log_splitter)
        
        # Tool-Panel Stack (3D oder 2D)
        self.tool_stack = QStackedWidget()
        self.tool_stack.setMinimumWidth(220)
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
        
        # === FIX: left_container statt left_widget ===
        self.main_splitter.addWidget(left_container) 
        
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
        """Positioniert das Extrude-Panel am unteren Rand des Fensters, zentriert."""
        if hasattr(self, 'extrude_panel') and self.extrude_panel.isVisible():
            # Panel Größe holen (oder Standard annehmen)
            pw = self.extrude_panel.width() if self.extrude_panel.width() > 10 else 320
            ph = self.extrude_panel.height() if self.extrude_panel.height() > 10 else 150
            
            # Koordinaten berechnen (Relativ zum MainWindow)
            # x = Mitte - halbe Panelbreite
            x = (self.width() - pw) // 2
            # y = Unten - Panelhöhe - etwas Abstand (z.B. 40px)
            y = self.height() - ph - 40
            
            self.extrude_panel.move(x, y)
            self.extrude_panel.raise_() # Sicherstellen, dass es vorne ist

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
        file_menu = mb.addMenu(tr("File"))
        file_menu.addAction(tr("New Project"), self._new_project, QKeySequence.New)
        file_menu.addAction(tr("Open..."), lambda: None, QKeySequence.Open)
        file_menu.addAction(tr("Save..."), lambda: None, QKeySequence.Save)
        file_menu.addSeparator()
        file_menu.addAction(tr("Export STL..."), self._export_stl)
        file_menu.addAction("Export STEP...", self._export_step)
        file_menu.addSeparator()
        file_menu.addAction(tr("Quit"), self.close, QKeySequence.Quit)
        
        # Bearbeiten-Menü
        edit_menu = mb.addMenu(tr("Edit"))
        edit_menu.addAction(tr("Undo"), lambda: None, QKeySequence.Undo)
        edit_menu.addAction(tr("Redo"), lambda: None, QKeySequence.Redo)
        
        # Ansicht-Menü
        view_menu = mb.addMenu(tr("View"))
        view_menu.addAction("Isometric", lambda: self.viewport_3d.set_view('iso'))
        view_menu.addAction(tr("Top"), lambda: self.viewport_3d.set_view('top'))
        view_menu.addAction(tr("Front"), lambda: self.viewport_3d.set_view('front'))
        view_menu.addAction(tr("Right"), lambda: self.viewport_3d.set_view('right'))
        
        # Hilfe-Menü
        help_menu = mb.addMenu(tr("Help"))
        help_menu.addAction(tr("About LiteCAD"), self._show_about)

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
        
    # In Klasse MainWindow:
    
    # In gui/main_window.py

    def _import_mesh_dialog(self):
        """Importiert STL/OBJ Dateien als neuen Body"""
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Mesh importieren", 
            "", 
            "Mesh Files (*.stl *.obj *.ply);;All Files (*.*)"
        )
        
        if not path:
            return

        try:
            import pyvista as pv
            import os
            
            # 1. Datei laden
            mesh = pv.read(path)
            
            # FIX: n_cells statt n_faces nutzen (PyVista Update)
            if not mesh or mesh.n_cells == 0:
                self.statusBar().showMessage("Fehler: Leeres Mesh oder ungültiges Format.")
                return

            # 2. Neuen Body im Dokument anlegen
            filename = os.path.basename(path)
            new_body = self.document.new_body(name=filename)
            
            # 3. WICHTIG: Das Mesh direkt zuweisen (als VTK Objekt)
            new_body.vtk_mesh = mesh
            
            # 4. Viewport aktualisieren
            self.viewport_3d.add_body(
                bid=new_body.id,
                name=new_body.name,
                mesh_obj=mesh,
                color=(0.7, 0.7, 0.7) 
            )
            
            # 5. Browser aktualisieren
            self.browser.refresh()
            
            # FIX: n_cells statt n_faces für die Anzeige
            self.statusBar().showMessage(f"Importiert: {filename} ({mesh.n_cells} Faces)")
            
            # Optional: Alles zeigen
            # self.viewport_3d.plotter.reset_camera()

        except Exception as e:
            from loguru import logger
            logger.error(f"Import Fehler: {e}")
            self.statusBar().showMessage(f"Import fehlgeschlagen: {str(e)}")
    
    def _update_viewport_all_impl(self):
        """Das eigentliche Update, wird vom Timer aufgerufen"""
        if not HAS_PYVISTA: return

        # 1. Sketches updaten
        visible_sketches = self.browser.get_visible_sketches()
        self.viewport_3d.set_sketches(visible_sketches)
        
        # 2. Bodies updaten
        self.viewport_3d.clear_bodies()
        
        # FIX: Variablen definieren, die vorher fehlten!
        default_color = (0.7, 0.7, 0.7)
        colors = [(0.6, 0.6, 0.8), (0.8, 0.6, 0.6), (0.6, 0.8, 0.6)]
        
        visible_bodies = self.browser.get_visible_bodies()
        
        for i, (b, visible) in enumerate(visible_bodies):
            if not visible:
                continue

            try:
                # Fall A: Vtk Cache vorhanden (Schnell & Modern)
                col = getattr(b, 'color', default_color)
                # Falls color None ist (manchmal bei Init), Default nehmen
                if col is None: col = default_color
                
                if hasattr(b, 'vtk_mesh') and b.vtk_mesh is not None:
                    self.viewport_3d.add_body(
                        bid=b.id, 
                        name=b.name, 
                        mesh_obj=b.vtk_mesh, 
                        edge_mesh_obj=b.vtk_edges, 
                        color=col
                    )
                    
                # Fall B: Fallback auf alte Listen (Legacy)
                elif hasattr(b, '_mesh_vertices') and b._mesh_vertices:
                    # FIX: Benannte Argumente (keywords) nutzen!
                    self.viewport_3d.add_body(
                        bid=b.id, 
                        name=b.name, 
                        verts=b._mesh_vertices,     # <-- Keyword wichtig!
                        faces=b._mesh_triangles,    # <-- Keyword wichtig!
                        color=colors[i % 3]
                    )
            except Exception as e:
                print(f"Fehler beim Laden von Body {b.name}: {e}")

        # Finales Rendering erzwingen
        if hasattr(self.viewport_3d, 'plotter'):
            self.viewport_3d.plotter.render()
        self.viewport_3d.update()

    def _update_viewport_all(self):
        """Aktualisiert ALLES im Viewport (Legacy Wrapper)"""
        # Sketches
        self.viewport_3d.set_sketches(self.browser.get_visible_sketches())
        
        # Bodies - komplett neu laden
        self.viewport_3d.clear_bodies()
        default_col = (0.7, 0.1, 0.1)
        
        for i, (b, visible) in enumerate(self.browser.get_visible_bodies()):
            if visible and hasattr(b, '_mesh_vertices') and b._mesh_vertices:
                # FIX: Benannte Argumente verwenden!
                self.viewport_3d.add_body(
                    bid=b.id, 
                    name=b.name, 
                    verts=b._mesh_vertices,     # Explizit benennen
                    faces=b._mesh_triangles,    # Explizit benennen
                    color=default_col
                ) 
        
    def _on_3d_action(self, action: str):
        """Verarbeitet 3D-Tool-Aktionen"""
        actions = {
            'new_sketch': self._new_sketch,
            'extrude': self._extrude_dialog,
            'import_mesh': self._import_mesh_dialog,
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
            'convert_to_brep': self._convert_selected_body_to_brep,
        }
        
        if action in actions:
            actions[action]()
        else:
            print(f"Unbekannte 3D-Aktion: {action}")
            
    def _convert_selected_body_to_brep(self):
        from PySide6.QtWidgets import QApplication, QMessageBox, QInputDialog
        from PySide6.QtCore import Qt

        body = self._get_active_body()
        if not body: 
            self.statusBar().showMessage("Kein Körper ausgewählt.")
            return

        # --- NEU: Parameter-Abfrage ---
        options = [
            "V1: Standard (Schnell, Sewing - für saubere Meshes)",
            "V5: Reverse Engineering (Smart - für 3D Scans, Rundungen & Löcher)"
        ]
        
        # Dialog anzeigen
        item, ok = QInputDialog.getItem(
            self, 
            "Konvertierungsmethode wählen", 
            "Strategie:", 
            options, 
            0, # Standard-Auswahl (Index 0 = V1)
            False # Nicht editierbar
        )

        if not ok or not item:
            return # Nutzer hat Abbrechen geklickt

        # Auswahl in technischen Parameter übersetzen
        mode_param = "v1"
        if "V5" in item:
            mode_param = "v5"
        # ------------------------------

        self.statusBar().showMessage(f"Konvertiere '{body.name}' mit [{mode_param.upper()}] (bitte warten)...")
        
        # UI updaten und Cursor auf "Warten" setzen (da V5 dauern kann)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents() 

        try:
            # 2. Konvertierung starten (mit Parameter!)
            success = body.convert_to_brep(mode=mode_param)
            
            if success:
                self.statusBar().showMessage(f"Erfolg! '{body.name}' ist jetzt ein CAD-Solid.")
                
                # 3. Browser aktualisieren
                self.browser.refresh()
                
                # 4. VIEWPORT FIX: Alte Darstellung entfernen
                if body.id in self.viewport_3d._body_actors:
                    for actor_name in self.viewport_3d._body_actors[body.id]:
                        try:
                            self.viewport_3d.plotter.remove_actor(actor_name)
                        except: pass
                
                # 5. Neu laden
                self._update_viewport_all_impl()
                
                # 6. Rendern erzwingen
                if hasattr(self.viewport_3d, 'plotter'):
                    self.viewport_3d.plotter.render()
                    self.viewport_3d.update()
                
            else:
                QApplication.restoreOverrideCursor() # Cursor zurücksetzen vor der Box
                error_msg = "Unbekannter Fehler."
                if mode_param == "v5":
                    error_msg = "V5 Reverse Engineering fehlgeschlagen.\nSind Open3D/RANSAC installiert und das Mesh sauber?"
                else:
                    error_msg = "Standard-Konvertierung fehlgeschlagen.\nIst das Mesh geschlossen?"
                
                QMessageBox.warning(self, "Fehler", error_msg)
                self.statusBar().showMessage("Konvertierung fehlgeschlagen.")

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Absturz", f"Kritischer Fehler bei Konvertierung: {e}")
            traceback.print_exc()

        finally:
            # WICHTIG: Cursor immer zurücksetzen, auch bei Fehlern
            QApplication.restoreOverrideCursor()
            
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
            #self._update_viewport_all()
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
        
        # DEFINITION: (Origin, Normal, X_Direction)
        # Damit legen wir fest, wo "Rechts" und "Oben" auf dem Bildschirm ist.
        plane_defs = {
            'xy': ((0,0,0), (0,0,1), (1,0,0)), # Boden: Z hoch, X rechts
            'xz': ((0,0,0), (0,1,0), (1,0,0)), # Vorne: Y hinten, X rechts (Standard CAD "Front")
            'yz': ((0,0,0), (1,0,0), (0,1,0))  # Rechts: X rechts, Y hoch (Standard CAD "Right")
        }
        
        # Standardwerte falls was schiefgeht
        default = ((0,0,0), (0,0,1), (1,0,0))
        
        origin, normal, x_dir = plane_defs.get(plane, default)
        
        # Wir nutzen die neue Logik mit x_dir_override
        self._create_sketch_at(origin, normal, x_dir_override=x_dir)

    def _on_custom_plane_selected(self, origin, normal):
        self.viewport_3d.set_plane_select_mode(False)
        
        # NEU: Versuchen, die stabile X-Achse vom Viewport zu holen
        x_dir = getattr(self.viewport_3d, '_last_picked_x_dir', None)
        
        self._create_sketch_at(origin, normal, x_dir)

    def _create_sketch_at(self, origin, normal, x_dir_override=None):
        s = self.document.new_sketch(f"Sketch{len(self.document.sketches)+1}")
        
        # Berechne Achsen
        if x_dir_override:
            # PERFEKT: Wir haben eine stabile Achse vom Detector
            x_dir = x_dir_override
            # Y berechnen (Kreuzprodukt)
            import numpy as np
            n_vec = np.array(normal)
            x_vec = np.array(x_dir)
            y_vec = np.cross(n_vec, x_vec)
            y_dir = tuple(y_vec)
        else:
            # Fallback: Raten (das was bisher Probleme machte)
            x_dir, y_dir = self._calculate_plane_axes(normal)
        
        # Speichere ALLES im Sketch
        s.plane_origin = origin
        s.plane_normal = normal
        s.plane_x_dir = x_dir  # <--- Das ist der Schlüssel zum Erfolg
        s.plane_y_dir = y_dir
        
        self.active_sketch = s
        self.sketch_editor.sketch = s
        
        # Bodies als Referenz übergeben (für Snapping auf Kanten)
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
        """Beendet den Sketch-Modus und räumt auf."""
        # Body-Referenzen im SketchEditor löschen (Ghost Bodies entfernen)
        if hasattr(self.sketch_editor, 'set_reference_bodies'):
            self.sketch_editor.set_reference_bodies([], (0,0,1), (0,0,0))
            
        self.active_sketch = None
        self._set_mode("3d")
        
        # Browser Refresh triggert Visibility-Check
        self.browser.refresh()
        
        # WICHTIG: Explizit Update anstoßen für sauberen Statuswechsel
        self._trigger_viewport_update()

    def _extrude_dialog(self):
        """Startet den Extrude-Modus."""
        # 1. Detector leeren und füllen
        self._update_detector() 
        
        if not self.viewport_3d.detector.selection_faces:
            self.statusBar().showMessage("Keine geschlossenen Flächen gefunden!", 3000)
            return

        # 2. Modus aktivieren
        self.viewport_3d.set_extrude_mode(True)
        self.viewport_3d.set_selection_mode("face")
        
        # 3. Panel anzeigen
        self.extrude_panel.reset()
        self.extrude_panel.setVisible(True)
        
        # FIX PROBLEM 1: Panel Positionierung verzögern!
        # Qt braucht ein paar Millisekunden, um die Breite des Panels zu berechnen.
        # Ohne Timer ist width() oft 0 oder falsch, daher landet es oben.
        QTimer.singleShot(10, self._position_extrude_panel)
        
        self.statusBar().showMessage("Fläche wählen und ziehen. Bestätigen mit Enter oder Rechtsklick.")

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
        
        # Wir übergeben None für die Indizes, da _on_extrusion_finished 
        # jetzt direkt den Detector im Viewport abfragt.
        self._on_extrusion_finished(None, height, operation)
    
    def _on_extrude_cancelled(self):
        """Extrude abgebrochen"""
        self.viewport_3d.set_extrude_mode(False)
        self.extrude_panel.setVisible(False)
        # Bodies wieder einblenden falls versteckt
        self.viewport_3d.set_all_bodies_visible(True)
        if hasattr(self.viewport_3d, 'detector'):
            self.viewport_3d.detector.clear()
        self.viewport_3d.selected_face_ids.clear()
        self.viewport_3d.hover_face_id = -1
        self.viewport_3d._draw_selectable_faces_from_detector()
        self.statusBar().showMessage(tr("Extrude abgebrochen"), 2000)
    
    def _on_toggle_bodies_visibility(self, hide: bool):
        """Toggle alle Bodies sichtbar/unsichtbar im Extrude-Modus"""
        self.viewport_3d.set_all_bodies_visible(not hide)
        
    
    def _update_detector(self):
        """
        Lädt ALLE Geometrien in den Detector und gibt Debug-Infos aus.
        """
        if not hasattr(self.viewport_3d, 'detector'): return
        
        self.viewport_3d.detector.clear()
        print("DEBUG: Detector Clear. Starte Sketch-Processing...")
        
        # A) Sketches verarbeiten
        visible_sketches = self.browser.get_visible_sketches()
        if not visible_sketches:
            print("DEBUG: Keine sichtbaren Sketches gefunden.")

        for sketch, visible in visible_sketches:
            if visible:
                x_dir = getattr(sketch, 'plane_x_dir', None)
                y_dir = getattr(sketch, 'plane_y_dir', None)
                
                # Fallback Berechnung falls Achsen fehlen (bei alten Projekten)
                if x_dir is None:
                     x_dir, y_dir = self.viewport_3d._calculate_plane_axes(sketch.plane_normal)
                
                print(f"DEBUG: Process Sketch '{sketch.name}' mit Normal {sketch.plane_normal}")
                
                self.viewport_3d.detector.process_sketch(
                    sketch, 
                    sketch.plane_origin, 
                    sketch.plane_normal, 
                    x_dir,
                    y_dir 
                )
        
        # B) Body-Flächen verarbeiten
        for body in self.document.bodies:
            if self.viewport_3d.is_body_visible(body.id):
                mesh = self.viewport_3d.get_body_mesh(body.id)
                if mesh:
                    self.viewport_3d.detector.process_body_mesh(body.id, mesh)

        count = len(self.viewport_3d.detector.selection_faces)
        print(f"DEBUG: Detector Update fertig. {count} selektierbare Flächen gefunden.")
        if count == 0:
            self.statusBar().showMessage("Warnung: Keine geschlossenen Flächen erkannt!")
    
    def _get_plane_from_sketch(self, sketch):
        """Erstellt ein build123d Plane Objekt aus den Sketch-Metadaten"""
        from build123d import Plane, Vector
        return Plane(
            origin=Vector(sketch.plane_origin),
            z_dir=Vector(sketch.plane_normal),
            x_dir=Vector(sketch.plane_x_dir)
        )
    
    def _on_extrusion_finished(self, face_indices, height, operation="New Body"):
        """Erstellt die finale Geometrie basierend auf der Auswahl im Detector"""
        
        # FIX 1: Verhindere doppelte Ausführung
        if getattr(self, '_is_processing_extrusion', False):
            return
        self._is_processing_extrusion = True

        try:
            # 1. Daten sammeln
            selection_data = []
            for fid in self.viewport_3d.selected_face_ids:
                face = next((f for f in self.viewport_3d.detector.selection_faces if f.id == fid), None)
                if face: selection_data.append(face)
            
            if not selection_data: 
                self.statusBar().showMessage("Nichts selektiert.")
                return

            first_face = selection_data[0]

            # Fall A: Sketch-Extrusion
            if first_face.domain_type.startswith('sketch'):
                try:
                    source_id = first_face.owner_id
                    target_sketch = next((s for s in self.document.sketches if s.id == source_id), None)
                    
                    polys = [f.shapely_poly for f in selection_data]

                    from modeling import ExtrudeFeature
                    feature = ExtrudeFeature(
                        sketch=target_sketch,
                        distance=height,
                        operation=operation,
                        precalculated_polys=polys
                    )
                    
                    target_body = self._get_active_body()
                    if operation == "New Body" or not target_body:
                        target_body = self.document.new_body()
                    
                    # WICHTIG: add_feature ruft intern bereits _rebuild() auf!
                    target_body.add_feature(feature)
                    
                    # --- GELÖSCHT: Doppelter Rebuild entfernt ---
                    # if hasattr(target_body, '_rebuild'):
                    #    target_body._rebuild()
                    
                    # FIX 2: Visuelles Update erzwingen
                    self._update_body_mesh(target_body)
                    
                    self._finish_extrusion_ui(msg=f"Extrusion erstellt: {target_body.name}")
                except Exception as e:
                    from loguru import logger
                    logger.error(f"Sketch Extrude Error: {e}")
                    import traceback
                    traceback.print_exc()

            # Fall B: Body-Face Extrusion (Push/Pull)
            elif first_face.domain_type == 'body_face':
                success = self._extrude_body_face_build123d({
                    'body_id': first_face.owner_id,       
                    'center_3d': first_face.plane_origin, 
                    'normal': first_face.plane_normal
                }, height, operation)
                
                if success:
                    self._finish_extrusion_ui(msg="Push/Pull erfolgreich.")
                else:
                    self.statusBar().showMessage("Push/Pull fehlgeschlagen.")
        
        finally:
            self._is_processing_extrusion = False

    def _finish_extrusion_ui(self, success=True, msg=""):
        """Hilfsfunktion zum Aufräumen der UI"""
        self.extrude_panel.setVisible(False)
        self.viewport_3d.set_extrude_mode(False)
        self.viewport_3d.set_all_bodies_visible(True)
        
        if hasattr(self.viewport_3d, 'detector'):
            self.viewport_3d.detector.clear()
        self.viewport_3d.selected_face_ids.clear()
        self.viewport_3d.hover_face_id = -1
        self.viewport_3d._draw_selectable_faces_from_detector()    
            
        if success:
            self.browser.refresh()
            if msg: self.statusBar().showMessage(msg)
        
        
    def _extrude_body_face_build123d(self, face_data, height, operation):
        """
        Extrem robuste Version 3.0:
        - "Entpackt" verschachtelte Compounds/Shells (löst das 0-Faces Problem)
        - Nutzt BRepExtrema für exakte Distanz
        """
        try:
            body_id = face_data.get('body_id')
            target_body = next((b for b in self.document.bodies if b.id == body_id), None)
            
            if not target_body or not hasattr(target_body, '_build123d_solid') or target_body._build123d_solid is None:
                print(f"Fehler: Body oder BREP-Daten fehlen.")
                return False

            from build123d import Vector, extrude, Shape, Compound
            from OCP.BRepExtrema import BRepExtrema_DistShapeShape
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
            from OCP.gp import gp_Pnt
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE
            from OCP.TopoDS import TopoDS

            # --- SCHRITT 1: Faces robust extrahieren (Der Fix für "0 Faces") ---
            b3d_obj = target_body._build123d_solid
            
            # Versuch 1: Standard Build123d Zugriff
            candidate_faces = b3d_obj.faces()
            
            # Versuch 2: Wenn leer (passiert oft bei importierten Compounds), OCP Explorer nutzen
            if not candidate_faces:
                print(f"Warnung: Standard .faces() ist leer (Typ: {type(b3d_obj)}). Versuche Deep-Scan...")
                explorer = TopExp_Explorer(b3d_obj.wrapped, TopAbs_FACE)
                candidate_faces = []
                while explorer.More():
                    # Face in Build123d Wrapper packen
                    from build123d import Face
                    candidate_faces.append(Face(TopoDS.Face_s(explorer.Current())))
                    explorer.Next()

            print(f"--- Debug: Suche Fläche nahe {face_data['center_3d']} ---")
            print(f"Anzahl Faces gefunden: {len(candidate_faces)}")
            
            if not candidate_faces:
                print("ABBRUCH: Body enthält topologisch keine Faces (Mesh Import fehlgeschlagen?).")
                return False

            # --- SCHRITT 2: Distanz-Messung ---
            mesh_center = Vector(face_data['center_3d'])
            ocp_pt_vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(mesh_center.X, mesh_center.Y, mesh_center.Z)).Vertex()
            
            best_face = None
            best_dist = float('inf')
            
            for i, face in enumerate(candidate_faces):
                try:
                    extrema = BRepExtrema_DistShapeShape(ocp_pt_vertex, face.wrapped)
                    if extrema.IsDone():
                        dist = extrema.Value()
                        if dist < best_dist:
                            best_dist = dist
                            best_face = face
                except: pass
            
            print(f"Suche beendet. Bester Abstand: {best_dist}")

            # Toleranz: Bei konvertierten Meshes darf der Klick auch mal 2mm daneben liegen
            if best_face is None or best_dist > 2.0:
                print(f"FEHLER: Keine Fläche in Reichweite gefunden.")
                return False

            print(f"Treffer! Extrudiere Face...")
            
            # --- AB HIER: Extrusion wie gehabt ---
            new_geo = extrude(best_face, amount=height)
            
            # --- BOOLEAN / NEW BODY LOGIC ---
            final_solid = None
            
            if operation == "New Body":
                new_body = self.document.new_body() 
                from modeling import ExtrudeFeature
                feat = ExtrudeFeature(sketch=None, distance=height, operation="New Body", name="Extrude (Face)")
                new_body.features.append(feat)
                new_body._build123d_solid = new_geo
                self._update_body_from_build123d(new_body, new_geo)
                return True
                
            elif operation == "Join":
                final_solid = target_body._build123d_solid + new_geo
            elif operation == "Cut":
                final_solid = target_body._build123d_solid - new_geo
            elif operation == "Intersect":
                final_solid = target_body._build123d_solid & new_geo
            
            if final_solid is not None:
                target_body._build123d_solid = final_solid
                self._update_body_from_build123d(target_body, final_solid)
                return True
            
            return False
            
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
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
        """
        Delegiert die Berechnung an die Body-Klasse (modeling.py),
        welche den CADTessellator nutzt.
        """
        # 1. Solid im Body speichern
        body._build123d_solid = solid
        
        # 2. Mesh zentral generieren lassen (nutzt Cache & OCP)
        # Dies füllt body.vtk_mesh und body.vtk_edges
        if hasattr(body, '_update_mesh_from_solid'):
            body._update_mesh_from_solid(solid)
        
        # 3. Viewport aktualisieren
        self._update_body_mesh(body)

    def _show_extrude_input_dialog(self):
        """Legacy Dialog - wird durch Panel ersetzt"""
        # Falls Tab gedrückt wird, fokussiere das Panel
        self.extrude_panel.height_input.setFocus()
        self.extrude_panel.height_input.selectAll()
    
    def _on_3d_click(self, event):
        """CAD-Selection über GeometryDetector (Sketch + Body Faces)"""

        # 1. Klick-Position
        pos = event.position() if hasattr(event, 'position') else event.pos()
        x, y = int(pos.x()), int(pos.y())

        # 2. Ray aus Viewport
        ray_origin, ray_dir = self.viewport_3d.get_ray_from_click(x, y)

        # 3. Selection-Mode an Viewport übergeben (wird dort in Filter übersetzt)
        self.viewport_3d.set_selection_mode(self.selection_mode)

        # 4. ZENTRALER CAD-Pick
        face_id = self.viewport_3d.detector.pick(
            ray_origin,
            ray_dir,
            selection_filter=self.viewport_3d.active_selection_filter
        )

        # 5. Ergebnis verarbeiten
        if face_id != -1:
            face = next(
                (f for f in self.viewport_3d.detector.selection_faces if f.id == face_id),
                None
            )

            if face:
                self.statusBar().showMessage(
                    f"Selection: {face.domain_type} ({face.owner_id})"
                )

                # Einheitliche Selection States
                self.selected_face_id = face.id
                self.selected_body = (
                    self.document.get_body(face.owner_id)
                    if face.domain_type == "body_face"
                    else None
                )

                # Optional: Highlight im Viewport
                self.viewport_3d.highlight_selection(face.id)

        else:
            self.statusBar().showMessage("Nichts getroffen")
            self.selected_face_id = None
            self.selected_body = None

        
        
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
        self.viewport_3d.add_body(
            bid=b.id, 
            name=b.name, 
            verts=verts, 
            faces=faces
        )
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
        
        # 1. Fallback für manuelles Mesh (z.B. aus Boolean-Preview)
        if mesh_override:
             import numpy as np
             if hasattr(mesh_override, 'points'): # PyVista Mesh
                 self.viewport_3d.add_body(
                     bid=body.id,
                     name=body.name,
                     mesh_obj=mesh_override,
                     color=getattr(body, 'color', None)
                 )
             return

        # 2. NEUER PFAD: Prüfen auf VTK/PyVista Cache (aus cad_tessellator)
        if hasattr(body, 'vtk_mesh') and body.vtk_mesh is not None:
             # Farbe bestimmen (Round Robin)
             try:
                 col_idx = self.document.bodies.index(body) % 3
             except: col_idx = 0
             default_col = (0.7, 0.7, 0.7)
             
             self.viewport_3d.add_body(
                 bid=body.id, 
                 name=body.name, 
                 mesh_obj=body.vtk_mesh, 
                 edge_mesh_obj=body.vtk_edges,
                 color=default_col
             )
             return

        # 3. LEGACY PFAD: Alte Listen (nur Fallback)
        if hasattr(body, '_mesh_vertices') and body._mesh_vertices:
             try:
                 col_idx = self.document.bodies.index(body) % 3
             except: col_idx = 0
             colors = [(0.6,0.6,0.8), (0.8,0.6,0.6), (0.6,0.8,0.6)]
             
             self.viewport_3d.add_body(
                 bid=body.id, 
                 name=body.name, 
                 verts=body._mesh_vertices, 
                 faces=body._mesh_triangles,
                 color=colors[col_idx]
             )

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent, Qt

        if event.type() == QEvent.KeyPress:
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
            
            if k == Qt.Key_E:
                if not self.viewport_3d.extrude_mode:
                    self._extrude_dialog()
                return True
                
            # Selektions-Modi umschalten (Pipeline an Viewport senden)
            if self.mode == "3d":
                if k == Qt.Key_1:
                    self.viewport_3d.set_selection_mode("face")
                    self.statusBar().showMessage("Modus: Flächen")
                    return True # Event konsumiert
                elif k == Qt.Key_2:
                    self.viewport_3d.set_selection_mode("hole")
                    self.statusBar().showMessage("Modus: Löcher")
                    return True
                elif k == Qt.Key_3:
                    self.viewport_3d.set_selection_mode("sketch")
                    self.statusBar().showMessage("Modus: Skizze")
                    return True

            # Bestätigung für Extrude
            if k in (Qt.Key_Return, Qt.Key_Enter):
                if self.viewport_3d.extrude_mode:
                    self._on_extrude_confirmed()
                    return True
            
            if k == Qt.Key_Escape:
                if self.viewport_3d.extrude_mode:
                    self._on_extrude_cancelled()
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
        #return super().eventFilter(obj, event)

    def _on_opt_change(self, o, v): pass
    def _edit_feature(self, d): 
        if d[0]=='sketch': self.active_sketch=d[1]; self.sketch_editor.sketch=d[1]; self._set_mode("sketch")
    def _new_project(self): 
        self.document = Document("Projekt1")
        self.browser.set_document(self.document)
        self._set_mode("3d")
    
    def set_selection_mode(self, mode):
        from gui.geometry_detector import SelectionFilter

        if mode == "face":
            self.active_selection_filter = SelectionFilter.FACE
        elif mode == "hole":
            self.active_selection_filter = SelectionFilter.HOLE
        elif mode == "sketch":
            self.active_selection_filter = SelectionFilter.SKETCH
        else:
            self.active_selection_filter = SelectionFilter.ALL
        
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