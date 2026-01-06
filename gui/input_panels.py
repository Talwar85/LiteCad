"""
LiteCAD - Input Panels
Fixed: TransformPanel Signal Blocking for circular updates.
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QFrame, QDoubleSpinBox, QCheckBox, QComboBox,
    QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QEvent, QPoint
from PySide6.QtGui import QFont, QKeyEvent

# --- Hilfsklasse für Enter-Taste ---
class ActionSpinBox(QDoubleSpinBox):
    """Eine SpinBox, die Enter-Tasten verarbeitet"""
    enterPressed = Signal()
    escapePressed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.interpretText() 
            self.enterPressed.emit()
            event.accept()
            return
        if event.key() == Qt.Key_Escape:
            self.escapePressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

class ExtrudeInputPanel(QFrame):
    """Input panel for extrude operation"""
    
    height_changed = Signal(float)
    confirmed = Signal()
    cancelled = Signal()
    bodies_visibility_toggled = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._height = 0.0
        self._bodies_hidden = False
        
        self.setMinimumWidth(520)
        self.setFixedHeight(60)
        
        self.setStyleSheet("""
            QFrame {
                background-color: #2d2d30;
                border: 2px solid #0078d4;
                border-radius: 8px;
            }
            QLabel { color: #ffffff; font-weight: bold; border: none; font-size: 12px; }
            QComboBox {
                background: #1e1e1e; border: 1px solid #555;
                border-radius: 4px; color: #fff; padding: 4px; min-width: 90px;
            }
            QDoubleSpinBox {
                background: #1e1e1e; color: #fff; border: 1px solid #555;
                border-radius: 4px; padding: 4px; font-weight: bold; min-width: 90px; font-size: 13px;
            }
            QPushButton {
                background: #444; color: #fff; border: 1px solid #555;
                border-radius: 4px; padding: 5px 12px; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover { background: #555; border-color: #777; }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 5, 15, 5)
        layout.setSpacing(12)
        
        self.label = QLabel("Extrude:")
        layout.addWidget(self.label)
        
        self.height_input = ActionSpinBox()
        self.height_input.setRange(-99999.0, 99999.0)
        self.height_input.setDecimals(2)
        self.height_input.setSuffix(" mm")
        self.height_input.setValue(0.0)
        
        self.height_input.valueChanged.connect(self._on_height_changed)
        self.height_input.enterPressed.connect(self._confirm)
        self.height_input.escapePressed.connect(self.cancelled.emit)
        
        layout.addWidget(self.height_input)
        
        self.op_combo = QComboBox()
        self.op_combo.addItems(["New Body", "Join", "Cut", "Intersect"])
        layout.addWidget(self.op_combo)

        self.btn_vis = QPushButton("👁")
        self.btn_vis.setCheckable(True)
        self.btn_vis.setFixedWidth(35)
        self.btn_vis.clicked.connect(self._toggle_vis)
        layout.addWidget(self.btn_vis)
        
        self.btn_ok = QPushButton("OK")
        self.btn_ok.setStyleSheet("background: #0078d4; color: white; border: none;")
        self.btn_ok.clicked.connect(self._confirm)
        layout.addWidget(self.btn_ok)
        
        self.btn_cancel = QPushButton("X")
        self.btn_cancel.setFixedWidth(35)
        self.btn_cancel.setStyleSheet("background: #d83b01; color: white; border: none;")
        self.btn_cancel.clicked.connect(self.cancelled.emit)
        layout.addWidget(self.btn_cancel)
        
        self.hide()

    def set_height(self, h):
        self.height_input.blockSignals(True)
        self.height_input.setValue(h)
        self.height_input.blockSignals(False)
        self._height = h

    def get_height(self):
        return self.height_input.value()

    def get_operation(self) -> str:
        return self.op_combo.currentText()

    def reset(self):
        self._height = 0.0
        self.height_input.blockSignals(True)
        self.height_input.setValue(0.0)
        self.height_input.blockSignals(False)
        self.op_combo.setCurrentIndex(0)
        self.btn_vis.setChecked(False)

    def _on_height_changed(self, val):
        self._height = val
        self.height_changed.emit(val)
        
    def _confirm(self):
        # Wir holen den aktuellen Wert direkt aus der Spinbox
        self._height = self.height_input.value()
        # Nur bestätigen, wenn wirklich Geometrie entstehen kann (> 0.001mm)
        if abs(self._height) > 0.001:
            self.confirmed.emit()
        else:
            self.cancelled.emit()

    def _toggle_vis(self):
        self._bodies_hidden = self.btn_vis.isChecked()
        self.bodies_visibility_toggled.emit(self._bodies_hidden)

    def show_at(self, pos_widget):
        self.show()
        self.raise_()
        if pos_widget:
            parent = pos_widget.parent() if pos_widget.parent() else pos_widget
            x = (parent.width() - self.width()) // 2
            y = parent.height() - self.height() - 50 
            self.move(x, y)
            self.height_input.setFocus()
            self.height_input.selectAll()


class FilletChamferPanel(QFrame):
    """Input panel for Fillet/Chamfer operations"""
    
    radius_changed = Signal(float)
    confirmed = Signal()
    cancelled = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._radius = 2.0
        self._mode = "fillet"
        self._target_body = None 
        
        self.setMinimumWidth(350)
        self.setFixedHeight(60)

        self.setStyleSheet("""
            QFrame {
                background: #2d2d30;
                border: 2px solid #0078d4;
                border-radius: 8px;
            }
            QLabel { color: #fff; font-weight: bold; border: none; font-size: 12px; }
            QDoubleSpinBox {
                background: #1e1e1e; color: #fff; border: 1px solid #555;
                border-radius: 4px; padding: 4px; font-weight: bold; font-size: 13px;
            }
            QPushButton {
                background: #444; color: #fff; border: 1px solid #555;
                border-radius: 4px; padding: 5px; font-weight: bold;
            }
            QPushButton:hover { background: #555; }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 5, 15, 5)
        layout.setSpacing(10)
        
        self.label = QLabel("Fillet:")
        layout.addWidget(self.label)
        
        self.radius_input = ActionSpinBox()
        self.radius_input.setRange(0.1, 1000.0)
        self.radius_input.setDecimals(2)
        self.radius_input.setSuffix(" mm")
        self.radius_input.setValue(2.0)
        
        self.radius_input.valueChanged.connect(self._on_value_changed)
        self.radius_input.enterPressed.connect(self._confirm)
        self.radius_input.escapePressed.connect(self._cancel)
            
        layout.addWidget(self.radius_input)
        
        self.ok_btn = QPushButton("OK")
        self.ok_btn.setStyleSheet("background: #0078d4; border: none;")
        self.ok_btn.clicked.connect(self._confirm)
        layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("X")
        self.cancel_btn.setFixedWidth(30)
        self.cancel_btn.setStyleSheet("background: #d83b01; border: none;")
        self.cancel_btn.clicked.connect(self._cancel)
        layout.addWidget(self.cancel_btn)
        
        self.hide()
    
    def set_target_body(self, body):
        self._target_body = body

    def get_target_body(self):
        return self._target_body

    def set_mode(self, mode: str):
        self._mode = mode
        self.label.setText("Fillet:" if mode == "fillet" else "Chamfer:")
    
    def _on_value_changed(self, value):
        self._radius = value
        self.radius_changed.emit(value)
    
    def _confirm(self):
        self._radius = self.radius_input.value()
        self.confirmed.emit()
    
    def _cancel(self):
        self.cancelled.emit()
    
    def get_radius(self) -> float:
        return self.radius_input.value()
    
    def reset(self):
        self._radius = 2.0
        self.radius_input.blockSignals(True)
        self.radius_input.setValue(2.0)
        self.radius_input.blockSignals(False)
    
    def show_at(self, pos_widget):
        self.show()
        self.raise_()
        
        if pos_widget:
            # Position relativ zum Parent (dem Hauptfenster) berechnen
            parent = pos_widget.parent() if pos_widget.parent() else pos_widget
            
            # X mittig zentrieren
            x = (parent.width() - self.width()) // 2
            
            # Y unten positionieren (mit 50px Abstand vom Rand)
            y = parent.height() - self.height() - 50 
            
            # Sicherheitscheck, damit es nicht oben raus rutscht
            if y < 0: y = 50
            
            self.move(x, y)
            
            # Fokus auf das Eingabefeld setzen (falls vorhanden)
            if hasattr(self, 'height_input'):
                self.height_input.setFocus()
                self.height_input.selectAll()
            elif hasattr(self, 'radius_input'):
                self.radius_input.setFocus()
                self.radius_input.selectAll()
            
            
class TransformPanel(QFrame):
    """Floating Panel für Move, Rotate, Scale"""
    
    values_changed = Signal(float, float, float) # x, y, z
    confirmed = Signal()
    cancelled = Signal()
    copy_toggled = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "move" # move, rotate, scale
        self.ignore_signals = False
        
        self.setMinimumWidth(280)
        self.setFixedHeight(90)
        
        self.setStyleSheet("""
            QFrame {
                background: #2d2d30;
                border: 2px solid #0078d4;
                border-radius: 8px;
            }
            QLabel { color: #ccc; font-weight: bold; font-size: 11px; border: none; }
            QDoubleSpinBox {
                background: #1e1e1e; color: #fff; border: 1px solid #555;
                border-radius: 4px; padding: 2px; font-weight: bold;
            }
            QPushButton {
                background: #444; color: #fff; border: 1px solid #555;
                border-radius: 4px; padding: 4px; font-weight: bold;
            }
            QPushButton:hover { background: #0078d4; border-color: #0078d4; }
            QCheckBox { color: #fff; spacing: 5px; }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)
        
        # Zeile 1: Titel und Checkbox
        row1 = QHBoxLayout()
        self.lbl_title = QLabel("Move:")
        row1.addWidget(self.lbl_title)
        
        self.chk_copy = QCheckBox("Kopie")
        self.chk_copy.toggled.connect(self.copy_toggled.emit)
        row1.addWidget(self.chk_copy)
        
        row1.addStretch()
        
        # Buttons oben rechts
        self.btn_ok = QPushButton("✓")
        self.btn_ok.setFixedWidth(30)
        self.btn_ok.clicked.connect(self.confirmed.emit)
        row1.addWidget(self.btn_ok)
        
        self.btn_cancel = QPushButton("✕")
        self.btn_cancel.setFixedWidth(30)
        self.btn_cancel.clicked.connect(self.cancelled.emit)
        row1.addWidget(self.btn_cancel)
        
        layout.addLayout(row1)
        
        # Zeile 2: X, Y, Z Inputs
        row2 = QHBoxLayout()
        self.inputs = []
        labels = ["X", "Y", "Z"]
        
        for l in labels:
            lbl = QLabel(l)
            lbl.setStyleSheet(f"color: {'#ff5555' if l=='X' else '#55ff55' if l=='Y' else '#5555ff'};")
            row2.addWidget(lbl)
            
            spin = ActionSpinBox()
            spin.setRange(-99999, 99999)
            spin.setDecimals(2)
            spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
            spin.valueChanged.connect(self._on_val_changed)
            spin.enterPressed.connect(self.confirmed.emit)
            spin.escapePressed.connect(self.cancelled.emit)
            
            row2.addWidget(spin)
            self.inputs.append(spin)
            
        layout.addLayout(row2)
        self.hide()

    def set_mode(self, mode):
        self._mode = mode
        titles = {"move": "Verschieben", "rotate": "Rotieren (°)", "scale": "Skalieren"}
        self.lbl_title.setText(titles.get(mode, mode))
        
        # Reset values
        self.ignore_signals = True
        defaults = [1.0, 1.0, 1.0] if mode == "scale" else [0.0, 0.0, 0.0]
        for spin, val in zip(self.inputs, defaults):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
            
            if mode == "scale": spin.setSingleStep(0.1)
            else: spin.setSingleStep(1.0)
        self.ignore_signals = False

    def update_values(self, x, y, z):
        """Update vom Viewport (wenn man mit der Maus zieht)"""
        self.ignore_signals = True
        vals = [x, y, z]
        for i, val in enumerate(vals):
            self.inputs[i].blockSignals(True)
            self.inputs[i].setValue(val)
            self.inputs[i].blockSignals(False)
        self.ignore_signals = False

    def get_values(self):
        return [s.value() for s in self.inputs]

    def _on_val_changed(self):
        if not self.ignore_signals:
            vals = self.get_values()
            self.values_changed.emit(vals[0], vals[1], vals[2])

    def show_at(self, pos_widget):
        self.show()
        self.raise_()
        
        if pos_widget:
            # Position relativ zum Parent (dem Hauptfenster) berechnen
            parent = pos_widget.parent() if pos_widget.parent() else pos_widget
            
            # X mittig zentrieren
            x = (parent.width() - self.width()) // 2
            
            # Y unten positionieren (mit 50px Abstand vom Rand)
            y = parent.height() - self.height() - 50 
            
            # Sicherheitscheck, damit es nicht oben raus rutscht
            if y < 0: y = 50
            
            self.move(x, y)
            
            # Fokus auf das Eingabefeld setzen (falls vorhanden)
            if hasattr(self, 'height_input'):
                self.height_input.setFocus()
                self.height_input.selectAll()
            elif hasattr(self, 'radius_input'):
                self.radius_input.setFocus()
                self.radius_input.selectAll()