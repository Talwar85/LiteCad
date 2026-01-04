"""
LiteCAD - Extrude Input Panel
Fusion360-style input panel that's always visible during extrude operations
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QFrame, QDoubleSpinBox, QCheckBox, QComboBox  # <--- Hier hinzugefügt
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtGui import QFont, QKeyEvent


class ExtrudeInputPanel(QFrame):
    """Input panel for extrude operation - always visible during extrude mode"""
    
    height_changed = Signal(float)  # Emitted when height value changes
    direction_flipped = Signal()     # Emitted when direction is flipped
    confirmed = Signal()             # Emitted when Enter pressed or OK clicked
    cancelled = Signal()             # Emitted when Escape pressed or Cancel clicked
    bodies_visibility_toggled = Signal(bool)  # Emitted when bodies should be hidden/shown
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._height = 0.0
        self._direction = 1  # 1 or -1
        self._bodies_hidden = False
        
        self.setStyleSheet("""
            QFrame {
                background: #2d2d30;
                border: 1px solid #0078d4;
                border-radius: 6px;
            }
            QComboBox {
                background: #1e1e1e;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                color: #fff;
                padding: 4px;
                min-width: 80px;
            }
            QComboBox:focus { border-color: #0078d4; }
            QComboBox::drop-down { border: none; }
        """)
        self.setFixedHeight(50)
        self.setMinimumWidth(380) # Etwas breiter für das Dropdown
        
        self._setup_ui()
        self.setVisible(False)
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        
        # Label
        label = QLabel("Extrude:")
        label.setStyleSheet("color: #ccc; font-size: 12px; font-weight: bold; border: none;")
        layout.addWidget(label)
        
        # Height Input
        self.height_input = QDoubleSpinBox()
        self.height_input.setRange(-1000, 1000)
        self.height_input.setDecimals(2)
        self.height_input.setSuffix(" mm")
        self.height_input.setValue(0)
        self.height_input.setFixedWidth(90)
        self.height_input.setStyleSheet("""
            QDoubleSpinBox {
                background: #1e1e1e;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                color: #fff;
                font-size: 14px;
                font-weight: bold;
                padding: 4px 8px;
            }
            QDoubleSpinBox:focus {
                border-color: #0078d4;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background: #3a3a3a;
                border: none;
                width: 16px;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background: #4a4a4a;
            }
        """)
        self.height_input.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.height_input)
        
        # NEU: Operation Dropdown
        self.op_combo = QComboBox()
        self.op_combo.addItems(["New Body", "Join", "Cut", "Intersect"])
        layout.addWidget(self.op_combo)

        # Hide Bodies Toggle Button
        self.hide_bodies_btn = QPushButton("👁")
        self.hide_bodies_btn.setToolTip("Bodies ein/ausblenden (H)")
        self.hide_bodies_btn.setCheckable(True)
        self.hide_bodies_btn.setFixedSize(32, 32)
        self.hide_bodies_btn.setStyleSheet("""
            QPushButton {
                background: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                color: #ccc;
                font-size: 16px;
            }
            QPushButton:hover {
                background: #4a4a4a;
                color: #fff;
            }
            QPushButton:checked {
                background: #c42b1c;
                color: #fff;
            }
        """)
        self.hide_bodies_btn.clicked.connect(self._toggle_bodies)
        layout.addWidget(self.hide_bodies_btn)

        # Flip Direction Button
        self.flip_btn = QPushButton("⇅")
        self.flip_btn.setToolTip("Richtung umkehren (F)")
        self.flip_btn.setFixedSize(32, 32)
        self.flip_btn.setStyleSheet("""
            QPushButton {
                background: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                color: #ccc;
                font-size: 16px;
            }
            QPushButton:hover {
                background: #4a4a4a;
                color: #fff;
            }
            QPushButton:pressed {
                background: #0078d4;
            }
        """)
        self.flip_btn.clicked.connect(self._flip_direction)
        layout.addWidget(self.flip_btn)
        
        layout.addSpacing(8)
        
        # OK Button
        self.ok_btn = QPushButton("✓ OK")
        self.ok_btn.setToolTip("Bestätigen (Enter)")
        self.ok_btn.setFixedWidth(60)
        self.ok_btn.setStyleSheet("""
            QPushButton {
                background: #0e639c;
                border: none;
                border-radius: 3px;
                color: white;
                font-size: 11px;
                font-weight: bold;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background: #1177bb;
            }
            QPushButton:pressed {
                background: #094771;
            }
        """)
        self.ok_btn.clicked.connect(self._confirm)
        layout.addWidget(self.ok_btn)
        
        # Cancel Button
        self.cancel_btn = QPushButton("✕")
        self.cancel_btn.setToolTip("Abbrechen (Escape)")
        self.cancel_btn.setFixedSize(32, 32)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                color: #ccc;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #c42b1c;
                border-color: #c42b1c;
                color: white;
            }
        """)
        self.cancel_btn.clicked.connect(self._cancel)
        layout.addWidget(self.cancel_btn)
    
    def _on_value_changed(self, value):
        self._height = value * self._direction
        self.height_changed.emit(self._height)
    
    def _flip_direction(self):
        self._direction *= -1
        self._height = self.height_input.value() * self._direction
        self.height_changed.emit(self._height)
        self.direction_flipped.emit()
    
    def _toggle_bodies(self):
        """Toggle Bodies visibility"""
        self._bodies_hidden = self.hide_bodies_btn.isChecked()
        self.bodies_visibility_toggled.emit(self._bodies_hidden)
    
    def _confirm(self):
        if abs(self._height) >= 0.1:
            self.confirmed.emit()
    
    def _cancel(self):
        self.cancelled.emit()
    
    def set_height(self, height: float):
        """Set height from external source (e.g., mouse drag)"""
        self._height = height
        self._direction = 1 if height >= 0 else -1
        self.height_input.blockSignals(True)
        self.height_input.setValue(abs(height))
        self.height_input.blockSignals(False)
    
    def get_height(self) -> float:
        return self._height
    
    def get_operation(self) -> str:
        """Gibt die gewählte Operation zurück"""
        return self.op_combo.currentText()

    def reset(self):
        """Reset to default state"""
        self._height = 0.0
        self._direction = 1
        self._bodies_hidden = False
        self.height_input.blockSignals(True)
        self.height_input.setValue(0)
        self.height_input.blockSignals(False)
        self.op_combo.setCurrentIndex(0) # Reset auf New Body
        self.hide_bodies_btn.setChecked(False)
    
    def show_at(self, viewport_widget):
        """Show panel at bottom center of viewport widget"""
        self.setVisible(True)
        self.adjustSize()
        
        if viewport_widget and self.parent():
            # Position relativ zum Parent berechnen
            vp_rect = viewport_widget.geometry()
            x = vp_rect.x() + (vp_rect.width() - self.width()) // 2
            y = vp_rect.y() + vp_rect.height() - self.height() - 30
            self.move(max(10, x), max(10, y))
        
        self.height_input.setFocus()
        self.height_input.selectAll()
    
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self._confirm()
        elif event.key() == Qt.Key_Escape:
            self._cancel()
        elif event.key() == Qt.Key_F:
            self._flip_direction()
        else:
            super().keyPressEvent(event)


class DimensionInputPanel(QFrame):
    """Generic dimension input panel for 2D sketch operations"""
    
    value_changed = Signal(str, float)  # (dimension_name, value)
    confirmed = Signal()
    cancelled = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._dimensions = {}  # name -> QDoubleSpinBox
        
        self.setStyleSheet("""
            QFrame {
                background: #2d2d30;
                border: 1px solid #0078d4;
                border-radius: 6px;
            }
        """)
        self.setMinimumWidth(200)
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 8, 12, 8)
        self.main_layout.setSpacing(6)
        
        # Title
        self.title_label = QLabel("Eingabe")
        self.title_label.setStyleSheet("color: #0078d4; font-size: 11px; font-weight: bold; border: none;")
        self.main_layout.addWidget(self.title_label)
        
        # Dimensions container
        self.dims_layout = QVBoxLayout()
        self.dims_layout.setSpacing(4)
        self.main_layout.addLayout(self.dims_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self.ok_btn = QPushButton("✓ OK")
        self.ok_btn.setStyleSheet("""
            QPushButton {
                background: #0e639c;
                border: none;
                border-radius: 3px;
                color: white;
                font-size: 10px;
                padding: 4px 12px;
            }
            QPushButton:hover { background: #1177bb; }
        """)
        self.ok_btn.clicked.connect(lambda: self.confirmed.emit())
        btn_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("✕")
        self.cancel_btn.setFixedWidth(28)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                color: #ccc;
                font-size: 12px;
            }
            QPushButton:hover { background: #c42b1c; color: white; }
        """)
        self.cancel_btn.clicked.connect(lambda: self.cancelled.emit())
        btn_layout.addWidget(self.cancel_btn)
        
        self.main_layout.addLayout(btn_layout)
        self.setVisible(False)
    
    def setup_for(self, title: str, dimensions: list):
        """
        Setup panel for specific operation
        dimensions: list of (name, label, default_value, min, max, suffix)
        """
        self.title_label.setText(title)
        
        # Clear old dimensions
        for name, spinbox in self._dimensions.items():
            spinbox.deleteLater()
        self._dimensions.clear()
        
        # Clear layout
        while self.dims_layout.count():
            item = self.dims_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add new dimensions
        for dim in dimensions:
            name, label_text, default, min_val, max_val, suffix = dim
            
            row = QHBoxLayout()
            row.setSpacing(8)
            
            label = QLabel(label_text)
            label.setStyleSheet("color: #aaa; font-size: 10px; border: none;")
            label.setFixedWidth(60)
            row.addWidget(label)
            
            spinbox = QDoubleSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setValue(default)
            spinbox.setSuffix(suffix)
            spinbox.setDecimals(2)
            spinbox.setStyleSheet("""
                QDoubleSpinBox {
                    background: #1e1e1e;
                    border: 1px solid #3a3a3a;
                    border-radius: 2px;
                    color: #fff;
                    font-size: 11px;
                    padding: 2px 4px;
                }
                QDoubleSpinBox:focus { border-color: #0078d4; }
            """)
            spinbox.valueChanged.connect(lambda v, n=name: self.value_changed.emit(n, v))
            row.addWidget(spinbox)
            
            self._dimensions[name] = spinbox
            self.dims_layout.addLayout(row)
        
        self.adjustSize()
    
    def get_value(self, name: str) -> float:
        if name in self._dimensions:
            return self._dimensions[name].value()
        return 0.0
    
    def set_value(self, name: str, value: float):
        if name in self._dimensions:
            self._dimensions[name].blockSignals(True)
            self._dimensions[name].setValue(value)
            self._dimensions[name].blockSignals(False)
    
    def show_at(self, x: int, y: int):
        self.move(x, y)
        self.setVisible(True)
        # Focus first input
        if self._dimensions:
            first = list(self._dimensions.values())[0]
            first.setFocus()
            first.selectAll()
    
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.confirmed.emit()
        elif event.key() == Qt.Key_Escape:
            self.cancelled.emit()
        elif event.key() == Qt.Key_Tab:
            # Cycle through inputs
            pass
        else:
            super().keyPressEvent(event)


class FilletChamferPanel(QFrame):
    """Input panel for Fillet/Chamfer operations"""
    
    radius_changed = Signal(float)
    confirmed = Signal()
    cancelled = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._radius = 2.0
        self._mode = "fillet"  # "fillet" or "chamfer"
        
        self.setStyleSheet("""
            QFrame {
                background: #2d2d30;
                border: 1px solid #0078d4;
                border-radius: 6px;
            }
        """)
        self.setFixedHeight(50)
        self.setMinimumWidth(300)
        
        self._setup_ui()
        self.setVisible(False)
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        
        # Label
        self.label = QLabel("Fillet:")
        self.label.setStyleSheet("color: #ccc; font-size: 12px; font-weight: bold; border: none;")
        layout.addWidget(self.label)
        
        # Radius Input
        self.radius_input = QDoubleSpinBox()
        self.radius_input.setRange(0.1, 100)
        self.radius_input.setDecimals(2)
        self.radius_input.setSuffix(" mm")
        self.radius_input.setValue(2.0)
        self.radius_input.setFixedWidth(90)
        self.radius_input.setStyleSheet("""
            QDoubleSpinBox {
                background: #1e1e1e;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                color: #fff;
                font-size: 14px;
                font-weight: bold;
                padding: 4px 8px;
            }
            QDoubleSpinBox:focus { border-color: #0078d4; }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background: #3a3a3a;
                border: none;
                width: 16px;
            }
        """)
        self.radius_input.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.radius_input)
        
        layout.addSpacing(8)
        
        # OK Button
        self.ok_btn = QPushButton("✓ OK")
        self.ok_btn.setFixedWidth(60)
        self.ok_btn.setStyleSheet("""
            QPushButton {
                background: #0e639c;
                border: none;
                border-radius: 3px;
                color: white;
                font-size: 11px;
                font-weight: bold;
                padding: 6px 12px;
            }
            QPushButton:hover { background: #1177bb; }
        """)
        self.ok_btn.clicked.connect(self._confirm)
        layout.addWidget(self.ok_btn)
        
        # Cancel Button
        self.cancel_btn = QPushButton("✕")
        self.cancel_btn.setFixedSize(32, 32)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                color: #ccc;
                font-size: 14px;
            }
            QPushButton:hover { background: #c42b1c; color: #fff; }
        """)
        self.cancel_btn.clicked.connect(self._cancel)
        layout.addWidget(self.cancel_btn)
    
    def set_mode(self, mode: str):
        """'fillet' or 'chamfer'"""
        self._mode = mode
        self.label.setText("Fillet:" if mode == "fillet" else "Chamfer:")
    
    def _on_value_changed(self, value):
        self._radius = value
        self.radius_changed.emit(value)
    
    def _confirm(self):
        self.confirmed.emit()
    
    def _cancel(self):
        self.cancelled.emit()
    
    def get_radius(self) -> float:
        return self._radius
    
    def reset(self):
        self._radius = 2.0
        self.radius_input.blockSignals(True)
        self.radius_input.setValue(2.0)
        self.radius_input.blockSignals(False)
    
    def show_at(self, viewport_widget):
        """Show panel at bottom center"""
        self.setVisible(True)
        self.adjustSize()
        
        parent = viewport_widget.parent() if viewport_widget.parent() else viewport_widget
        x = (parent.width() - self.width()) // 2
        y = parent.height() - 100
        self.move(x, y)
        self.radius_input.setFocus()
        self.radius_input.selectAll()
    
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._confirm()
        elif event.key() == Qt.Key_Escape:
            self._cancel()
        else:
            super().keyPressEvent(event)
