# MashCAD

A lightweight CAD application focused on 2D sketching for 3D printing workflows.

## Features

### 2D Sketch Editor
- **Drawing Tools**: Line, Rectangle, Circle (3 modes), Arc, Polygon, Slot, Spline, Point
- **Transform Tools**: Move, Copy, Rotate, Mirror, Scale, Linear/Circular Pattern
- **Modify Tools**: Trim, Extend, Offset, Fillet, Chamfer
- **Constraints**: Horizontal, Vertical, Parallel, Perpendicular, Equal, Concentric, Tangent
- **Generators**: Involute Gear, Star, Hex Nut (M2-M14)
- **Precision Input**: Tab-key for numeric input (Fusion360-style)
- **Smart Snapping**: Endpoint, Midpoint, Center, Intersection, Grid

### 3D Viewport (PyVista)
- Hardware-accelerated rendering
- Extrude with live preview
- Face detection for overlapping shapes
- ViewCube navigation
- Boolean operations (New Body, Join, Cut, Intersect)

### Export
- DXF (2D sketches)
- STL (3D models)


###Missing:
Step, SVG


## Installation

```bash
pip install -r requirements.txt
python main.py
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| L | Line | R | Rectangle | C | Circle | P | Polygon |
| M | Move | X | Construction mode | G | Grid snap |
| Tab | Precision input | Enter | Confirm | Escape | Cancel |
| Delete | Delete selected | Ctrl+Z/Y | Undo/Redo |
| F | Fit view | E | Extrude (3D) |

## Project Structure

```
litecad/
├── main.py                 # Entry point
├── gui/
│   ├── sketch_editor.py    # 2D editor core
│   ├── sketch_handlers.py  # Tool handlers
│   ├── sketch_renderer.py  # Rendering
│   ├── viewport_pyvista.py # 3D viewport
│   └── ...
├── sketcher/               # 2D geometry & constraints
└── core/                   # 3D bodies
```

## License
MIT
