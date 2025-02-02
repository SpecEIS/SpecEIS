import kml2geojson
import numpy as np
import pygmsh
import pyproj
import os

class Mesher:
    def __init__(self):
        pass

    def kml_to_coordinates(self,path,in_epsg=4326,out_epsg=3338):
        data = kml2geojson.main.convert(f'{path}')
        self.coordinates = np.array(data[0]['features'][0]['geometry']['coordinates']).squeeze()
        self.transformer = pyproj.Transformer.from_crs(in_epsg, out_epsg,always_xy=True)
        self.coordinates = self.transformer.transform(*self.coordinates[:,:2].T)

    def build_mesh(self,mesh_size,output_path,center=False,normalize_by_length_scale=False):
        with pygmsh.geo.Geometry() as geom:
            points = []
            for point in np.array(self.coordinates).T[:-1]:
                p = geom.add_point(point,mesh_size=mesh_size)
                points.append(p)

            lines = []
            for i in range(-1,len(points)-1):
                l = geom.add_line(points[i],points[i+1])
                lines.append(l)

            loop = geom.add_curve_loop(lines)
            surface = geom.add_plane_surface(loop)
            geom.add_physical(loop.curves,'s')
            #geom.save_geometry(f'{output_path}/geo_file.geo_unrolled')

            self.mesh = geom.generate_mesh(dim=2)#,algorithm=5)

            if center:
                self.center = (self.mesh.points.max(axis=0) + self.mesh.points.min(axis=0))/2.
                self.mesh.points -= self.center

            if normalize_by_length_scale:
                self.mesh.points /= normalize_by_length_scale

            os.makedirs(f'{output_path}', exist_ok=True)
            self.mesh.write(f'{output_path}/mesh.msh',file_format='gmsh22')

mesher = Mesher()
mesher.kml_to_coordinates('../data/outline/kennicott_outline.kml')
mesher.build_mesh(1000,'../meshes/mesh_1000',center=True,normalize_by_length_scale=10000)
print(mesher.center)


