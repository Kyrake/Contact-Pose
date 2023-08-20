import time
from utilities.import_open3d import *


class HandPoseAnimator:

    def __init__(self, geoms_list, delay=0.1):
        self.geoms_list = geoms_list
        self.current_geoms_index = 0
        self.delay = delay

    def clear_geoms(self, vis):
        time.sleep(self.delay)
        vis.clear_geometries()
        self.current_geoms_index = self.current_geoms_index + 1
        if self.current_geoms_index < len(self.geoms_list):
            geoms = self.geoms_list[self.current_geoms_index]
            for geo in geoms:
                vis.add_geometry(geo)
        else:
            vis.destroy_window()

    def start_animation(self):
        vis = o3dv.Visualizer()
        vis.create_window()
        geoms = self.geoms_list[self.current_geoms_index]
        for geo in geoms:
            vis.add_geometry(geo)

        vis.register_animation_callback(self.clear_geoms)
        vis.run()
