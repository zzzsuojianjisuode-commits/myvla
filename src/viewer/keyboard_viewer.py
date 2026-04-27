import glfw
import mujoco
import numpy as np


class KeyboardTeleopViewer:
    def __init__(self, model, data, title="OMY keyboard teleop", width=1400, height=1000):
        self.model = model
        self.data = data
        self._keys_down = set()
        self._keys_pressed = set()
        self._left_text = ""
        self._right_text = ""
        self._overlay_gridpos = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
        self._preview_camera_ids = []
        self._preview_camera_names = []
        self._markers = []
        self._show_view_axes_icon = True
        self._mouse_left = False
        self._mouse_middle = False
        self._mouse_right = False
        self._last_cursor = None

        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW.")

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window.")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

        self.cam = mujoco.MjvCamera()
        self.fixed_cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.scene = mujoco.MjvScene(self.model, maxgeom=50000)
        self.context = mujoco.MjrContext(
            self.model,
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
        )
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultCamera(self.fixed_cam)

    def _key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self._keys_down.add(key)
            self._keys_pressed.add(key)
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
        elif action == glfw.RELEASE:
            self._keys_down.discard(key)

    def _mouse_button_callback(self, window, button, action, mods):
        pressed = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._mouse_left = pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._mouse_middle = pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._mouse_right = pressed
        self._last_cursor = glfw.get_cursor_pos(window)

    def _cursor_pos_callback(self, window, xpos, ypos):
        if self._last_cursor is None:
            self._last_cursor = (xpos, ypos)
            return

        last_x, last_y = self._last_cursor
        self._last_cursor = (xpos, ypos)
        if not (self._mouse_left or self._mouse_middle or self._mouse_right):
            return

        _, height = glfw.get_framebuffer_size(window)
        if height <= 0:
            return
        dx = (xpos - last_x) / height
        dy = (ypos - last_y) / height

        if self._mouse_left:
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ROTATE_H,
                dx,
                0,
                self.scene,
                self.cam,
            )
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                0,
                dy,
                self.scene,
                self.cam,
            )
        elif self._mouse_middle:
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_MOVE_H,
                dx,
                0,
                self.scene,
                self.cam,
            )
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_MOVE_V,
                0,
                dy,
                self.scene,
                self.cam,
            )
        elif self._mouse_right:
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ZOOM,
                0,
                dy,
                self.scene,
                self.cam,
            )

    def _scroll_callback(self, window, x_offset, y_offset):
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0,
            -0.05 * y_offset,
            self.scene,
            self.cam,
        )

    def is_alive(self):
        return not glfw.window_should_close(self.window)

    def is_key_down(self, key):
        return key in self._keys_down

    def consume_key(self, key):
        if key in self._keys_pressed:
            self._keys_pressed.remove(key)
            return True
        return False

    def set_text(self, left="", right="", gridpos=None):
        self._left_text = left
        self._right_text = right
        if gridpos is not None:
            self._overlay_gridpos = gridpos

    def set_camera(self, lookat, distance, azimuth, elevation):
        self.cam.lookat[:] = lookat
        self.cam.distance = distance
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation

    def set_camera_previews(self, camera_names):
        self._preview_camera_ids = []
        self._preview_camera_names = []
        for name in camera_names:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cam_id >= 0:
                self._preview_camera_ids.append(cam_id)
                self._preview_camera_names.append(name)

    def set_view_axes_icon(self, enabled=True):
        self._show_view_axes_icon = enabled

    def add_marker(self, **marker_params):
        self._markers.append(marker_params)

    def plot_sphere(self, pos, radius=0.02, rgba=(1.0, 0.0, 0.0, 0.5), label=""):
        self.add_marker(
            pos=pos,
            size=[radius, radius, radius],
            rgba=rgba,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            label=label,
        )

    def plot_cylinder(
        self,
        pos,
        mat=None,
        radius=0.01,
        half_height=0.18,
        rgba=(0.0, 1.0, 0.0, 0.5),
        label="",
    ):
        if mat is None:
            mat = np.eye(3)
        self.add_marker(
            pos=pos,
            mat=mat,
            size=[radius, radius, half_height],
            rgba=rgba,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            label=label,
        )

    def plot_cylinder_between_points(
        self,
        start,
        end,
        radius=0.01,
        rgba=(0.0, 1.0, 0.0, 0.5),
        label="",
    ):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        length = np.linalg.norm(end - start)
        if length < 1e-8:
            return
        self.add_marker(
            from_pos=start,
            to_pos=end,
            width=radius,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            radius=radius,
            rgba=rgba,
            label=label,
        )

    def capture_fixed_camera_rgb(self, camera_name, width=256, height=256):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Missing camera {camera_name!r}.")

        glfw.make_context_current(self.window)
        self.fixed_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.fixed_cam.fixedcamid = cam_id
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            self.pert,
            self.fixed_cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self.context)
        return np.flipud(rgb)

    def render(self):
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        view_forward = np.asarray(self.scene.camera[0].forward, dtype=np.float64).copy()
        view_up = np.asarray(self.scene.camera[0].up, dtype=np.float64).copy()
        for marker in self._markers:
            self._add_marker_to_scene(marker)
        mujoco.mjr_render(viewport, self.scene, self.context)

        self._render_camera_previews(width, height)

        if self._left_text or self._right_text:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                self._overlay_gridpos,
                viewport,
                self._left_text,
                self._right_text,
                self.context,
            )

        self._render_view_axes_icon(width, height, view_forward, view_up)
        self._markers.clear()
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def _add_marker_to_scene(self, marker):
        if self.scene.ngeom >= self.scene.maxgeom:
            raise RuntimeError(f"Ran out of scene geoms, maxgeom={self.scene.maxgeom}.")

        geom = self.scene.geoms[self.scene.ngeom]
        geom_type = marker.get("type", mujoco.mjtGeom.mjGEOM_BOX)
        rgba = np.asarray(marker.get("rgba", [1.0, 1.0, 1.0, 1.0]), dtype=np.float32)

        if "from_pos" in marker and "to_pos" in marker:
            mujoco.mjv_initGeom(
                geom,
                geom_type,
                np.asarray([0.01, 0.01, 0.01], dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.eye(3, dtype=np.float64).reshape(9),
                rgba,
            )
            mujoco.mjv_connector(
                geom,
                geom_type,
                marker.get("width", 0.01),
                np.asarray(marker["from_pos"], dtype=np.float64),
                np.asarray(marker["to_pos"], dtype=np.float64),
            )
        else:
            size = np.asarray(marker.get("size", [0.1, 0.1, 0.1]), dtype=np.float64)
            pos = np.asarray(marker.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
            mat = np.asarray(marker.get("mat", np.eye(3)), dtype=np.float64).reshape(9)
            mujoco.mjv_initGeom(geom, geom_type, size, pos, mat, rgba)
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        label = marker.get("label", "")
        if label:
            geom.label = label
        self.scene.ngeom += 1

    def _render_view_axes_icon(self, width, height, forward, up):
        if not self._show_view_axes_icon or width <= 0 or height <= 0:
            return

        icon_size = 118
        margin = 14
        x = width - icon_size - margin
        y = margin
        viewport = mujoco.MjrRect(x, y, icon_size, icon_size)
        mujoco.mjr_rectangle(viewport, 0.02, 0.02, 0.02, 0.45)

        forward = self._normalize(forward)
        up = self._normalize(up)
        right = self._normalize(np.cross(forward, up))
        origin = np.array([x + icon_size * 0.50, y + icon_size * 0.42], dtype=np.float64)
        length = icon_size * 0.36

        axes = (
            (np.array([1.0, 0.0, 0.0]), (1.0, 0.08, 0.08), "X"),
            (np.array([0.0, 1.0, 0.0]), (0.1, 0.9, 0.1), "Y"),
            (np.array([0.0, 0.0, 1.0]), (0.15, 0.35, 1.0), "Z"),
        )
        for axis, color, label in axes:
            vec = np.array([np.dot(axis, right), np.dot(axis, up)], dtype=np.float64)
            end = origin + vec * length
            self._draw_icon_line(origin, end, color, thickness=4)
            self._draw_icon_square(end, color, size=8)
            self._draw_icon_label(label, end, color, width, height)

        self._draw_icon_square(origin, (1.0, 1.0, 1.0), size=6)

    def _draw_icon_line(self, start, end, color, thickness=3):
        delta = end - start
        steps = max(int(np.linalg.norm(delta) / 3.0), 1)
        for idx in range(steps + 1):
            point = start + delta * (idx / steps)
            self._draw_icon_square(point, color, size=thickness)

    def _draw_icon_square(self, point, color, size=4):
        half = size * 0.5
        rect = mujoco.MjrRect(
            int(round(point[0] - half)),
            int(round(point[1] - half)),
            int(size),
            int(size),
        )
        mujoco.mjr_rectangle(rect, color[0], color[1], color[2], 0.95)

    def _draw_icon_label(self, label, point, color, width, height):
        x = np.clip((point[0] + 8.0) / width, 0.0, 1.0)
        y = np.clip((point[1] + 8.0) / height, 0.0, 1.0)
        mujoco.mjr_text(
            mujoco.mjtFont.mjFONT_NORMAL,
            label,
            self.context,
            x,
            y,
            color[0],
            color[1],
            color[2],
        )

    @staticmethod
    def _normalize(value):
        value = np.asarray(value, dtype=np.float64)
        norm = np.linalg.norm(value)
        if norm < 1e-8:
            return value
        return value / norm

    @staticmethod
    def _rotation_matrix_between_points(start, end):
        base_axis = np.array([1e-10, -1e-10, 1.0], dtype=np.float64)
        target_axis = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        target_axis = target_axis / np.linalg.norm(target_axis)
        cross = np.cross(base_axis, target_axis)
        cross_norm = np.linalg.norm(cross)
        if cross_norm < 1e-8:
            return np.eye(3)
        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3) + skew + skew @ skew * (
            (1.0 - np.dot(base_axis, target_axis)) / (cross_norm * cross_norm)
        )

    def _render_camera_previews(self, width, height):
        if not self._preview_camera_ids:
            return

        margin = 12
        gap = 10
        preview_w = max(220, int(width * 0.22))
        preview_h = int(preview_w * 0.75)
        self.fixed_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        for idx, cam_id in enumerate(self._preview_camera_ids):
            self.fixed_cam.fixedcamid = cam_id
            x = width - preview_w - margin
            y = height - margin - (idx + 1) * preview_h - idx * gap
            if y < margin:
                break
            viewport = mujoco.MjrRect(x, y, preview_w, preview_h)
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.opt,
                self.pert,
                self.fixed_cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self.scene,
            )
            mujoco.mjr_render(viewport, self.scene, self.context)
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                self._preview_camera_names[idx],
                "",
                self.context,
            )

    def close(self):
        if self.window:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()
