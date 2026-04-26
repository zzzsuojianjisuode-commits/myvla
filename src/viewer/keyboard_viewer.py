import glfw
import mujoco


class KeyboardTeleopViewer:
    def __init__(self, model, data, title="OMY keyboard teleop", width=1400, height=1000):
        self.model = model
        self.data = data
        self._keys_down = set()
        self._keys_pressed = set()
        self._left_text = ""
        self._right_text = ""
        self._preview_camera_ids = []
        self._preview_camera_names = []
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

    def set_text(self, left="", right=""):
        self._left_text = left
        self._right_text = right

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
        mujoco.mjr_render(viewport, self.scene, self.context)

        self._render_camera_previews(width, height)

        if self._left_text or self._right_text:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                self._left_text,
                self._right_text,
                self.context,
            )

        glfw.swap_buffers(self.window)
        glfw.poll_events()

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
