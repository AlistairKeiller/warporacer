import numpy as np
import warp as wp
import warp.render
import pyglet
from imgui_bundle import imgui
from imgui_bundle.python_backends import pyglet_backend

import time

class ImGuiManager:
    # ImGUI uses Top Left origin while Pyglet is Bottom Left and you have to remember DPI scaling!
    def __init__(self, renderer:warp.render.OpenGLRenderer):
        imgui.create_context()
        self.renderer = renderer

        # Tell the backend NOT to attach its broken callbacks and override them
        self.impl = pyglet_backend.create_renderer(self.renderer.window, attach_callbacks=False)
        self.impl.on_mouse_motion = self.on_mouse_motion
        self.impl.on_mouse_drag = self.on_mouse_drag
        self.impl._attach_callbacks(self.renderer.window)

        # "self.renderer.enable_keyboard_interaction = False" is not working dynamically
        self.renderer.window.push_handlers(on_key_press=self.on_key_press)

    def want_capture_mouse(self, *args, **kwargs):
        return self.impl.io.want_capture_mouse

    def on_mouse_motion(self, x, y, dx, dy):
        ratio = self.renderer.window.get_pixel_ratio()
        self.impl.io.add_mouse_pos_event(x / ratio, self.impl.io.display_size.y - (y / ratio))
    
    def on_mouse_drag(self, x, y, dx, dy, button, modifiers):
        self.impl._on_mouse_button(button, True)
        self.on_mouse_motion(x, y, dx, dy)
        return self.impl.io.want_capture_mouse
    
    def on_key_press(self, symbol, modifiers):
        return self.impl.io.want_capture_keyboard

    def render_frame(self):
        """Renders a single frame of the UI. This should be called from the main render loop."""
        io = imgui.get_io()
        ratio = self.renderer.window.get_pixel_ratio()
        io.display_size = self.renderer.screen_width / ratio, self.renderer.screen_height / ratio

        self.impl.process_inputs()
        imgui.new_frame()

        self.draw_ui()

        imgui.render()
        self.impl.render(imgui.get_draw_data())

    def draw_ui(self):
        """Draws the UI"""
        imgui.set_next_window_size(imgui.ImVec2(640, 480), imgui.Cond_.first_use_ever)
        imgui.set_next_window_pos(imgui.ImVec2(0, 0), imgui.Cond_.first_use_ever)

        imgui.begin("Warp Float Values")

        imgui.text(f"A read-only float: {self.renderer.clock_time}")
        imgui.separator()

        imgui.text("Editable floats:")
        imgui.separator()
        imgui.text("File Dialog Examples:")

        ok = "bruh"
        imgui.input_text("Gmaing", ok)

        imgui.end()

    def shutdown(self):
        self.impl.shutdown()

class Visuals:
    # If you see this error and you have more than one GPU (iGPU & eGPU):
    #   "Warp UserWarning: Could not register GL buffer since CUDA/OpenGL interoperability is not available.
    #   Falling back to copy operations between the Warp array and the OpenGL buffer."
    # Then you have to make sure ALL aspects of the Python program is running on GPU. On Windows you find
    # the Python executable and set to "High Performance" in Windows Graphics settings.

    def __init__(self):
        # Init Warp Render
        self.renderer = warp.render.OpenGLRenderer(
            screen_width=1280,
            screen_height=720,
            device="cuda"
        )
        self.renderer.render_ground()

        # Init ImGUI
        self.imgui_manager = ImGuiManager(self.renderer)
        self.renderer.render_2d_callbacks.append(self.imgui_manager.render_frame)

    def render(self):
        # Warp Render Begin
        time = self.renderer.clock_time
        self.renderer.begin_frame(time)

        # Begin Render
        self.renderer.render_cylinder(
            "cylinder",
            [3.2, 1.0, np.sin(time + 0.5)],
            np.array(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.sin(time + 0.5))),
            radius=0.5,
            half_height=0.8,
        )
        self.renderer.render_cone(
            "cone",
            [-1.2, 1.0, 0.0],
            np.array(wp.quat_from_axis_angle(wp.vec3(0.707, 0.707, 0.0), time)),
            radius=0.5,
            half_height=0.8,
        )
        # End Render

        # Warp Render End
        self.renderer.end_frame()

    def clear(self):
        self.imgui_manager.shutdown()
        self.renderer.clear()

if __name__ == "__main__":
    with wp.ScopedDevice("cuda"):
        # Make sure these are on the same device!
        print(f"Warp Device: {wp.get_device().name}")
        print(f"Pyglet Device: {pyglet.gl.gl_info.get_renderer()}")

        vs = Visuals()

        def draw(dt):
            vs.render()

        pyglet.clock.schedule_interval(draw, 1 / 60.0)
        pyglet.app.run()

        vs.clear()