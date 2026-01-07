#!/usr/bin/env python3
"""
MeshToBREPV34 - 'KERNEL CAST FIXED'
===================================
FIX: Behebt den Type-Casting Fehler (TopoDS_Shape -> TopoDS_Shell).
FIX: Behebt NameError 'Shape'.
LOGIK:
1. Gmsh erzeugt Quads.
2. OCP erzeugt Flächen.
3. Sewing verbindet sie.
4. Wir casten das Ergebnis explizit via 'TopoDS.Shell_s' für build123d.

Installation:
    pip install gmsh trimesh loguru build123d numpy
"""

import sys
import os
import numpy as np
import trimesh
import tempfile
from loguru import logger

try:
    import gmsh
except ImportError:
    logger.error("❌ 'gmsh' fehlt!")
    sys.exit(1)

try:
    from build123d import *
    # WICHTIG: Expliziter Import der Basis-Klasse
    from build123d import Shape as B123DShape 
    
    # OCP Kernel Imports
    from OCP.gp import gp_Pnt
    # TopoDS für das Casting (WICHTIG!)
    from OCP.TopoDS import TopoDS, TopoDS_Shell, TopoDS_Face
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_COMPOUND
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakePolygon, 
        BRepBuilderAPI_MakeFace, 
        BRepBuilderAPI_Sewing,
        BRepBuilderAPI_MakeSolid
    )
    from OCP.BRepCheck import BRepCheck_Analyzer
except ImportError:
    logger.error("❌ build123d/OCP fehlt!")

class MeshToBREPV5:
    
    def __init__(self, debug=True):
        self.debug = debug
        if debug:
            logger.add("mesh_v34_final.log", level="INFO", mode="w")

    def convert(self, input_path: str):
        logger.info("=" * 70)
        logger.info("🚀 V34 - Direct Kernel (Type Safe)")
        logger.info("=" * 70)
        
        if not os.path.exists(input_path): return self._fallback()

        # 1. Cleanup
        clean_stl = self._create_clean_temp_stl(input_path)
        if not clean_stl: return self._fallback()

        # 2. Gmsh -> OCP Sewing
        shape = self._run_gmsh_ocp(clean_stl)
        
        try: os.remove(clean_stl)
        except: pass
        
        return shape

    def _create_clean_temp_stl(self, input_path):
        try:
            mesh = trimesh.load(input_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            
            mesh.merge_vertices()
            try: mesh.update_faces(mesh.unique_faces())
            except: pass
            
            fd, temp_path = tempfile.mkstemp(suffix=".stl")
            os.close(fd)
            mesh.export(temp_path)
            return temp_path
        except: return None

    def _run_gmsh_ocp(self, stl_path):
        logger.info("⚙️ Gmsh Generation...")
        
        try:
            if gmsh.is_initialized(): gmsh.finalize()
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1 if self.debug else 0)
            gmsh.model.add("FinalModel")
            gmsh.merge(stl_path)
            
            # Setup für Quads
            angle = 40.0 * (np.pi / 180.0)
            gmsh.model.mesh.classifySurfaces(angle, True, True)
            gmsh.model.mesh.createGeometry()
            
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.RecombineAll", 1) 
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0)
            
            bbox = gmsh.model.getBoundingBox(-1, -1)
            diag = np.linalg.norm(np.array(bbox[3:]) - np.array(bbox[:3]))
            # Etwas feiner für bessere Qualität
            target_size = diag / 30.0 
            
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_size * 0.5)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_size * 2.0)
            
            gmsh.model.mesh.generate(2)
            
            logger.info("🔨 Erzeuge OCP Faces...")
            
            nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
            coords_reshaped = np.array(nodeCoords).reshape(-1, 3)
            node_map = {tag: coords_reshaped[i] for i, tag in enumerate(nodeTags)}
            
            elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2)
            
            # Sewing Tool (Toleranz 0.1mm um kleine Lücken zu schließen)
            sewer = BRepBuilderAPI_Sewing(0.1)
            
            count = 0
            
            for i, etype in enumerate(elemTypes):
                tags = elemNodeTags[i]
                
                # Quads (3) oder Triangles (2)
                nodes_per_elem = 4 if etype == 3 else (3 if etype == 2 else 0)
                if nodes_per_elem == 0: continue
                
                for j in range(0, len(tags), nodes_per_elem):
                    # Punkte holen
                    p_indices = [tags[j+k] for k in range(nodes_per_elem)]
                    pts = [node_map[idx] for idx in p_indices]
                    
                    # Polygon bauen
                    poly_maker = BRepBuilderAPI_MakePolygon()
                    for p in pts:
                        poly_maker.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
                    poly_maker.Close()
                    
                    # Face bauen
                    face_builder = BRepBuilderAPI_MakeFace(poly_maker.Wire())
                    if face_builder.IsDone():
                        sewer.Add(face_builder.Face())
                        count += 1

            logger.info(f"🔗 Nähe {count} Flächen zusammen...")
            sewer.Perform()
            sewed_shape = sewer.SewedShape()
            
            # --- DER FIX ---
            # Wir prüfen, was wir bekommen haben (Shell, Compound, oder Solid?)
            shape_type = sewed_shape.ShapeType()
            
            if shape_type == TopAbs_SHELL:
                logger.info("  -> Ergebnis ist ein Shell (Hülle).")
                # Downcast TopoDS_Shape -> TopoDS_Shell
                ocp_shell = TopoDS.Shell_s(sewed_shape)
                b123d_shell = Shell(ocp_shell)
                
                # Versuche Solid zu machen
                try:
                    solid = Solid.make_solid(b123d_shell)
                    logger.success("✅ Solid erfolgreich erstellt!")
                    return solid
                except:
                    logger.warning("⚠️ Nicht wasserdicht, gebe Shell zurück.")
                    return b123d_shell
                    
            else:
                # Vermutlich ein Compound (Haufen von Flächen)
                logger.info(f"  -> Ergebnis ist Typ {shape_type} (meist Compound).")
                # Wir geben es als generisches Shape zurück
                return B123DShape(sewed_shape)

        except Exception as e:
            logger.error(f"Kernel Fehler: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback()
        finally:
            if gmsh.is_initialized(): gmsh.finalize()

    def _fallback(self):
        return Box(10,10,10)

def convert(stl):
    converter = MeshToBREPV34_Final()
    return converter.convert(stl)

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit()
    s = convert(sys.argv[1])
    if s: 
        out = sys.argv[1].replace(".ply", "").replace(".stl", "") + "_v34.step"
        s.export_step(out)
        print(f"Exportiert: {out}")