import numpy as np
import pyvista as pv
from loguru import logger
import traceback

# Optional dependencies
try:
    import pymeshlab as ml
    HAS_MESHLAB = True
except ImportError:
    HAS_MESHLAB = False
    logger.warning("pymeshlab nicht gefunden. Mesh-zu-BREP Konvertierung wird eingeschränkt sein.")

from OCP.gp import gp_Pnt
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from build123d import Solid, Shape

class MeshToBREPConverter:
    """
    Smart Converter: 
    - Low Poly -> Direct BREP -> Planar Simplify (Coplanar Faces mergen)
    - High Poly -> MeshLab (Decimation + Smooth) -> BREP
    """
    
    def __init__(self):
        pass

    def _simplify_brep(self, ocp_shape):
        """
        Verschmilzt coplanare Flächen (macht aus 100 Dreiecken auf einer Wand 1 Rechteck).
        Macht aus 'Triangle Soup' ein sauberes CAD-Modell.
        """
        try:
            # ShapeUpgrade_UnifySameDomain ist das OCP Tool zum Aufräumen
            # Argumente: Shape, UnifyFaces, UnifyEdges, ConcatBSplines
            upgrader = ShapeUpgrade_UnifySameDomain(ocp_shape, True, True, True)
            
            # Toleranz etwas lockern, damit auch ungenaue STLs gemerged werden
            upgrader.SetLinearTolerance(0.1) 
            upgrader.SetAngularTolerance(0.1) # ca 5 Grad
            
            upgrader.Build()
            return upgrader.Shape()
        except Exception as e:
            logger.warning(f"Simplifizierung fehlgeschlagen: {e}")
            return ocp_shape

    def convert(self, mesh: pv.PolyData, target_faces=5000, smooth=True) -> Shape:
        """
        Hauptmethode zur Konvertierung.
        """
        logger.info(f"Analysiere Mesh für Konvertierung ({mesh.n_cells} Faces)...")

        # Sicherstellen, dass wir Dreiecke haben
        if not mesh.is_all_triangles:
            mesh = mesh.triangulate()

        # Strategie-Entscheidung
        use_meshlab = False
        
        # Nur MeshLab nutzen wenn VIEL zu viele Polygone da sind
        if mesh.n_cells > target_faces:
            use_meshlab = True
            
        # Wenn Mesh sehr klein ist -> KEIN Smoothing, da es Ecken rund lutscht
        if mesh.n_cells < 2000:
            smooth = False

        if use_meshlab and not HAS_MESHLAB:
            use_meshlab = False

        try:
            out_verts = []
            out_faces = []

            # --- PFAD A: MeshLab Pipeline (für Scans/Organisch) ---
            if use_meshlab:
                logger.info("High-Poly erkannt: Starte MeshLab Pipeline...")
                ms = ml.MeshSet()
                verts = mesh.points.astype(np.float64)
                faces = mesh.faces.reshape(-1, 4)[:, 1:4].astype(np.int32)
                
                m = ml.Mesh(verts, faces)
                ms.add_mesh(m)

                if smooth:
                    try:
                        ms.generate_surface_reconstruction_screened_poisson(depth=8, scale=1.1)
                    except: pass

                # Decimation
                current_faces = ms.current_mesh().face_number()
                if current_faces > target_faces:
                    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)

                out_mesh = ms.current_mesh()
                out_verts = out_mesh.vertex_matrix()
                out_faces = out_mesh.face_matrix()

            # --- PFAD B: Direkte Konvertierung (für technische STLs) ---
            else:
                logger.info("Low-Poly erkannt: Direkte Konvertierung + Simplifizierung.")
                out_verts = mesh.points
                out_faces = mesh.faces.reshape(-1, 4)[:, 1:4]

            # --- BREP Erstellung (Sewing) ---
            logger.info(f"Erstelle BREP aus {len(out_faces)} Facetten...")
            
            # Sewing Toleranz
            sewing = BRepBuilderAPI_Sewing(tolerance=1e-3)

            for face_idx in out_faces:
                p0 = out_verts[face_idx[0]]
                p1 = out_verts[face_idx[1]]
                p2 = out_verts[face_idx[2]]

                poly = BRepBuilderAPI_MakePolygon()
                poly.Add(gp_Pnt(float(p0[0]), float(p0[1]), float(p0[2])))
                poly.Add(gp_Pnt(float(p1[0]), float(p1[1]), float(p1[2])))
                poly.Add(gp_Pnt(float(p2[0]), float(p2[1]), float(p2[2])))
                poly.Close()

                if poly.IsDone():
                    face_builder = BRepBuilderAPI_MakeFace(poly.Wire())
                    if face_builder.IsDone():
                        sewing.Add(face_builder.Face())

            sewing.Perform()
            sewed_shape = sewing.SewedShape()

            if sewed_shape.IsNull():
                raise ValueError("Sewing erzeugte leere Shape.")

            # --- WICHTIG: POST-PROCESSING (Der Fix für dein Problem) ---
            logger.info("Optimiere Flächen (Planar Simplification)...")
            simplified_shape = self._simplify_brep(sewed_shape)
            
            # --- Validierung und Wrap ---
            b3d_shape = Shape(simplified_shape)
            
            # Versuch: Solid reparieren falls offen
            if b3d_shape.shape_type == "Shell":
                try:
                    # Manchmal hilft simplify() schon, das Loch zu schließen
                    # Wenn nicht, versuchen wir MakeSolid
                    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
                    maker = BRepBuilderAPI_MakeSolid(simplified_shape)
                    if maker.IsDone():
                        b3d_shape = Solid(maker.Solid())
                    else:
                        # Fixversuch über build123d fix()
                        b3d_shape = b3d_shape.fix()
                except: pass

            if b3d_shape.is_valid():
                return b3d_shape
            else:
                return Shape(simplified_shape)

        except Exception as e:
            logger.error(f"BREP Konvertierung fehlgeschlagen: {e}")
            traceback.print_exc()
            return None