# Copyright (c) Facebook, Inc. and its affiliates.
# Code by Samarth Brahmbhatt
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import numpy as np

import init_paths
from utilities.dataset import ContactPose
import utilities.misc as mutils
from utilities.import_open3d import *
# from open3d import utility as o3du

#!/usr/bin/env python3
import glob
# import numpy as np
# import open3d as o3d
# import open3d.visualization.gui as gui
# import open3d.visualization.rendering as rendering
import os
import platform
import sys

isMacOS = (platform.system() == "Darwin")


class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    def __init__(self):
        self.mouse_model = o3dv.gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = o3dv.gui.Color(1, 1, 1)
        self.show_axes = False

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.UNLIT: o3dv.rendering.Material(),
            Settings.LIT: o3dv.rendering.Material(),
            Settings.NORMALS: o3dv.rendering.Material(),
            Settings.DEPTH: o3dv.rendering.Material()
        }
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.UNLIT]

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        assert (self.material.shader == Settings.UNLIT)
        prefab = Settings.PREFAB[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    MATERIAL_NAMES = ["Unlit", "Lit", "Normals", "Depth"]
    MATERIAL_SHADERS = [
        Settings.UNLIT, Settings.LIT, Settings.NORMALS, Settings.DEPTH
    ]

    def __init__(self, width, height):
        self.settings = Settings()
        resource_path = o3dv.gui.Application.instance.resource_path

        self.window = o3dv.gui.Application.instance.create_window(
            "Open3D", width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = o3dv.gui.SceneWidget()
        self._scene.scene = o3dv.rendering.Open3DScene(w.renderer)

        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)
        self._settings_panel = o3dv.gui.Vert(
            0, o3dv.gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # Create a collapsable vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.
        view_ctrls = o3dv.gui.CollapsableVert("View controls", 0.25 * em,
                                              o3dv.gui.Margins(em, 0, 0, 0))

        self._mouse_button = o3dv.gui.Button("Mouse")
        self._mouse_button.horizontal_padding_em = 0.5
        self._mouse_button.vertical_padding_em = 0
        self._mouse_button.set_on_clicked(self._set_mouse_mode_rotate)

        self._model_button = o3dv.gui.Button("Model")
        self._model_button.horizontal_padding_em = 0.5
        self._model_button.vertical_padding_em = 0
        self._model_button.set_on_clicked(self._set_mouse_mode_model)

        view_ctrls.add_child(o3dv.gui.Label("Mouse controls"))
        # We want two rows of buttons, so make two horizontal layouts. We also
        # want the buttons centered, which we can do be putting a stretch item
        # as the first and last item. Stretch items take up as much space as
        # possible, and since there are two, they will each take half the extra
        # space, thus centering the buttons.
        h = o3dv.gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._mouse_button)
        h.add_child(self._model_button)
        h.add_stretch()
        view_ctrls.add_child(h)

        self._bg_color = o3dv.gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)

        grid = o3dv.gui.VGrid(2, 0.25 * em)
        grid.add_child(o3dv.gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        view_ctrls.add_child(grid)

        self._show_axes = o3dv.gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_axes)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)

        self._settings_panel.add_fixed(separation_height)
        material_settings = o3dv.gui.CollapsableVert("Material settings", 0,
                                                     o3dv.gui.Margins(em, 0, 0, 0))

        self._shader = o3dv.gui.Combobox()
        self._shader.add_item(AppWindow.MATERIAL_NAMES[0])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[1])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[2])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[3])
        self._shader.set_on_selection_changed(self._on_shader)
        self._material_prefab = o3dv.gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(
            self._on_material_prefab)
        self._material_color = o3dv.gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = o3dv.gui.Slider(o3dv.gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = o3dv.gui.VGrid(2, 0.25 * em)
        grid.add_child(o3dv.gui.Label("Type"))
        grid.add_child(self._shader)
        grid.add_child(o3dv.gui.Label("Material"))
        grid.add_child(self._material_prefab)
        grid.add_child(o3dv.gui.Label("Color"))
        grid.add_child(self._material_color)
        grid.add_child(o3dv.gui.Label("Point size"))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)
        # ----

        # Normally our user interface can be children of all one layout (usually
        # a vertical layout), which is then the only child of the window. In our
        # case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout
        # callback. The on_layout callback should set the frame
        # (position + size) of every child correctly. After the callback is
        # done the window will layout the grandchildren.
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if o3dv.gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = o3dv.gui.Menu()
                app_menu.add_item("About", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            file_menu = o3dv.gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current Image...",
                               AppWindow.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = o3dv.gui.Menu()
            settings_menu.add_item("Lighting & Materials",
                                   AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = o3dv.gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = o3dv.gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            o3dv.gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

        self._apply_settings()

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background_color(bg_color)
        self._scene.scene.show_axes(self.settings.show_axes)

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._bg_color.color_value = self.settings.bg_color
        self._show_axes.checked = self.settings.show_axes
        self._material_prefab.enabled = (
            self.settings.material.shader == Settings.UNLIT)
        c = o3dv.gui.Color(self.settings.material.base_color[0],
                           self.settings.material.base_color[1],
                           self.settings.material.base_color[2],
                           self.settings.material.base_color[3])
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size

    def _on_layout(self, theme):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * theme.font_size
        height = min(r.height,
                     self._settings_panel.calc_preferred_size(theme).height)
        self._settings_panel.frame = o3dv.gui.Rect(r.get_right() - width, r.y, width,
                                                   height)

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(
            o3dv.gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(
            o3dv.gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_shader(self, name, index):
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_material_prefab(self, name, index):
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color):
        self.settings.material.base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_menu_open(self):
        dlg = o3dv.gui.FileDialog(o3dv.gui.FileDialog.OPEN, "Choose file to load",
                                  self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        dlg = o3dv.gui.FileDialog(o3dv.gui.FileDialog.SAVE, "Choose file to save",
                                  self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        o3dv.gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        o3dv.gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = o3dv.gui.Dialog("About")

        # Add the text
        dlg_layout = o3dv.gui.Vert(em, o3dv.gui.Margins(em, em, em, em))
        dlg_layout.add_child(o3dv.gui.Label("Open3D o3dv.gui Example"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = o3dv.gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = o3dv.gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def load(self, path):
        self._scene.scene.clear_geometry()

        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_mesh(path)
        if mesh is not None:
            if len(mesh.triangles) == 0:
                print(
                    "[WARNING] Contains 0 triangles, will read as point cloud")
                mesh = None
            else:
                mesh.compute_vertex_normals()
                if len(mesh.vertex_colors) == 0:
                    mesh.paint_uniform_color([1, 1, 1])
                geometry = mesh
            # Make sure the mesh has texture coordinates
            if not mesh.has_triangle_uvs():
                uv = np.array([[0.0, 0.0]] * (3 * len(mesh.triangles)))
                mesh.triangle_uvs = o3du.Vector2dVector(uv)
        else:
            print("[Info]", path, "appears to be a point cloud")
            mesh = None

        if geometry is None:
            ioProgressAmount = 0.5
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points", path)
                cloud = None

        if geometry is not None:
            self._scene.scene.add_geometry("__model__", geometry,
                                           self.settings.material)
            bounds = geometry.get_axis_aligned_bounding_box()
            self._scene.setup_camera(60, bounds, bounds.get_center())

    def export_image(self, path, width, height):
        img = None

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)

    def apply_colormap_to_mesh(self, mesh, sigmoid_a=0.05, invert=False):
        colors = np.asarray(mesh.vertex_colors)[:, 0]
        colors = mutils.texture_proc(colors, a=sigmoid_a, invert=invert)
        colors = plt.cm.inferno(colors)[:, :3]
        mesh.vertex_colors = o3du.Vector3dVector(colors)
        return mesh

    def apply_semantic_colormap_to_mesh(self, mesh, semantic_idx, sigmoid_a=0.05,
                                        invert=False):
        colors = np.asarray(mesh.vertex_colors)[:, 0]
        colors = mutils.texture_proc(colors, a=sigmoid_a, invert=invert)

        # apply different colormaps based on finger
        mesh_colors = np.zeros((len(colors), 3))
        cmaps = ['Greys', 'Purples', 'Oranges', 'Greens', 'Blues', 'Reds']
        cmaps = [plt.cm.get_cmap(c) for c in cmaps]
        for semantic_id in np.unique(semantic_idx):
            if (len(cmaps) <= semantic_id):
                print('Not enough colormaps, ignoring semantic id {:d}'.format(
                    semantic_id))
                continue
            idx = semantic_idx == semantic_id
            mesh_colors[idx] = cmaps[semantic_id](colors[idx])[:, :3]
        mesh.vertex_colors = o3du.Vector3dVector(mesh_colors)
        return mesh

    def show_contactmap(self, p_num=53, intent='use', object_name='hammer', mode='simple_hands',
                        joint_sphere_radius_mm=4.0, bone_cylinder_radius_mm=2.5,
                        bone_color=np.asarray([224.0, 172.0, 105.0])/255):
        """
            mode =
            simple: just contact map
            simple_hands: skeleton + contact map
            semantic_hands_fingers: skeleton + contact map colored by finger proximity
            semantic_hands_phalanges: skeleton + contact map colored by phalange proximity
            """
        cp = ContactPose(p_num, intent, object_name)

        # apply simple colormap to the mesh
        if 'simple' in mode:
            pass

        if 'hands' in mode:
            # read hands
            line_ids = mutils.get_hand_line_ids()
            joint_locs = cp.hand_joints()

            # show hands
            hand_colors = [[0, 1, 0], [1, 0, 0]]
            for hand_idx, hand_joints in enumerate(joint_locs):
                if hand_joints is None:
                    continue

                # joint locations
                for j in hand_joints:
                    m = o3dg.TriangleMesh.create_sphere(radius=joint_sphere_radius_mm*1e-3,
                                                        resolution=10)
                    T = np.eye(4)
                    T[:3, 3] = j - hand_joints[0]
                    m.transform(T)
                    m.paint_uniform_color(hand_colors[hand_idx])
                    m.compute_vertex_normals()
                    self._scene.scene.add_geometry("joints"+str(j), m,
                                                   self.settings.material)

                # connecting lines
                for line_idx, (idx0, idx1) in enumerate(line_ids):
                    bone = hand_joints[idx1] - hand_joints[idx0]
                    # print("first: " + str(idx1) + ", sec: " + str(idx0))
                    h = np.linalg.norm(bone)
                    l = o3dg.TriangleMesh.create_cylinder(radius=bone_cylinder_radius_mm*1e-3,
                                                          height=h, resolution=10)
                    T = np.eye(4)
                    # vector direction
                    T[2, 3] = -h/2.0
                    l.transform(T)
                    T = mutils.rotmat_from_vecs(bone, [0, 0, 1])
                    # Translation: draw from each joints
                    T[:3, 3] = hand_joints[idx1] - hand_joints[0]
                    l.transform(T)
                    l.paint_uniform_color(bone_color)
                    l.compute_vertex_normals()
                    self._scene.scene.add_geometry("links"+str(idx1), l,
                                                   self.settings.material)

                    # WE ARE INTEREST THIS PART OF CODE
                    bone = hand_joints[idx1] - hand_joints[idx0]
                    h = np.linalg.norm(bone)
                    T = np.eye(4)
                    # ROTATION
                    T = mutils.rotmat_from_vecs(bone, [0, 0, 1])
                    # TRANSLATION
                    # draw from each joints
                    T[:3, 3] = hand_joints[idx1] - hand_joints[0]
                    # print(T[:3, :3])
                    # print(R.from_matrix(T[:3, :3]).as_rotvec())
                    x = o3dg.TriangleMesh.create_coordinate_frame(
                        size=0.01, origin=(0, 0, 0))
                    # IF YOU WANT TO SPECIFY AXIS DIRECITON THEN DO THIS
                    # x = x.rotate(o3dg.TriangleMesh.get_rotation_matrix_from_xyz((np.pi/2, 0, -np.pi/2)))
                    x.transform(T)
                    self._scene.scene.add_geometry("joints_axis"+str(idx1), x,
                                                   self.settings.material)

            geometry = o3dg.TriangleMesh.create_coordinate_frame(
                size=0.05, origin=(0, 0, 0))
            # IF YOU WANT TO SPECIFY AXIS DIRECITON THEN DO THIS
            # geometry = geometry.rotate(o3dg.TriangleMesh.get_rotation_matrix_from_xyz((gamma, beta, alpha)))
            # geometry = geometry.rotate(o3dg.TriangleMesh.get_rotation_matrix_from_axis_angle(([0.,0.,np.pi*2])))

            if geometry is not None:
                self._scene.scene.add_geometry("__model__", geometry,
                                               self.settings.material)
                bounds = geometry.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())


if __name__ == '__main__':
    # We need to initalize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    o3dv.gui.Application.instance.initialize()

    w = AppWindow(1024, 768)

    w.show_contactmap()

    # Run the event loop. This will not return until the last window is closed.
    o3dv.gui.Application.instance.run()
