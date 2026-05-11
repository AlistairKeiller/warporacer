from typing import TYPE_CHECKING  # Forward declaration

import warp as wp
from imgui_bundle import imgui
from imgui_bundle.python_backends import pyglet_backend

if TYPE_CHECKING:
    from include.environment import Environment

class ImGuiManager:
    # ImGUI uses Top Left origin while Pyglet is Bottom Left and you have to remember DPI scaling!
    def __init__(self, renderer: wp.render.OpenGLRenderer, env: "Environment"):
        imgui.create_context()
        self.renderer = renderer
        self.env = env

        # Tell the backend NOT to attach its broken callbacks and override them
        self.impl = pyglet_backend.create_renderer(self.renderer.window, attach_callbacks=False)
        self.impl.on_mouse_motion = self.on_mouse_motion
        self.impl.on_mouse_drag = self.on_mouse_drag
        self.impl._attach_callbacks(self.renderer.window)

        # "self.renderer.enable_keyboard_interaction = False" is not working dynamically
        self.renderer.window.push_handlers(on_key_press=self._on_key_press)

    def on_mouse_motion(self, x, y, dx, dy):
        ratio = self.renderer.window.get_pixel_ratio()
        self.impl.io.add_mouse_pos_event(x / ratio, self.impl.io.display_size.y - (y / ratio))
    
    def on_mouse_drag(self, x, y, dx, dy, button, modifiers):
        self.impl._on_mouse_button(button, True)
        self.on_mouse_motion(x, y, dx, dy)
        return self.impl.io.want_capture_mouse
    
    def _on_key_press(self, symbol, modifiers):
        return self.impl.io.want_capture_keyboard

    def _render_frame(self):
        """Renders a single frame of the UI. This should be called from the main render loop."""
        io = imgui.get_io()
        ratio = self.renderer.window.get_pixel_ratio()
        io.display_size = self.renderer.screen_width / ratio, self.renderer.screen_height / ratio

        self.impl.process_inputs()
        imgui.new_frame()

        self._draw_ui()

        imgui.render()
        self.impl.render(imgui.get_draw_data())

    def _draw_ui(self):
        """Draws the UI"""
        io = imgui.get_io()
        imgui.set_next_window_pos(imgui.ImVec2(150, 0), imgui.Cond_.first_use_ever)

        imgui.begin("Environment Data")
        imgui.text(f"Number of cars: {self.env.num_envs}")

        imgui.end()

    def shutdown(self):
        self.impl.shutdown()
