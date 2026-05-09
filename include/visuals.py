import time
from typing import TYPE_CHECKING  # Forward declaration

import numpy as np
import pyglet
import warp as wp
import warp.render  # Must include for OpenGLRenderer
from pyglet.window import key
from scipy.ndimage import binary_dilation

from include.constants import *
from include.imgui_manager import ImGuiManager
from include.map import Map

if TYPE_CHECKING:
    from include.environment import Environment

class Visuals:
    def __init__(self, env: Environment, map: Map):
        self.env = env
        self.map = map
    
        # Init Warp Render
        print(f"Warp Device: {wp.get_device().name}")
        print(f"Pyglet Device: {pyglet.gl.gl_info.get_renderer()}")
        self.renderer = warp.render.OpenGLRenderer( # Changed from wp to warp for VSCode
            title="Warp from Thaumcraft",
            screen_width=1280,
            screen_height=720,
            near_plane=0.1,
            far_plane=1000,
            up_axis="Z", # Cannot change sun direction because it's hard coded in! Cringe!
            background_color=(0,0,0),
            draw_grid=False,
            draw_axis=False,
            draw_sky=False,
            device=wp.get_device()
        )

        # Init ImGUI
        self.imgui_manager = ImGuiManager(self.renderer, env)
        self.renderer.render_2d_callbacks.append(self.imgui_manager._render_frame)

        # Initialize keyboard controls
        self.key_handler = key.KeyStateHandler()
        self.renderer.window.push_handlers(self.key_handler)

        # Initialize map
        self.initialized_all_agents = False
        self.lidar_hit_points = wp.zeros(NUM_LIDAR, dtype=wp.vec3, device=self.env.device)

        self._setup_map()
        self._setup_dynamic_objects()

    def interactive_render_loop(self):
        """While loop for rendering, must be last!"""

        import torch
        user_actions = torch.zeros((self.env.num_envs, ACT_DIM), device=str(self.env.device))

        self.last_render_time = time.perf_counter()
        while self.renderer.is_running():
            current_time = time.perf_counter()
            if current_time - self.last_render_time >= DT:
                self.last_render_time = current_time
                # Take user inputs

                # I = Forward (+1.0), K = Backward/Brake (-1.0)
                throttle = float(self.key_handler[key.I] - self.key_handler[key.K])
                
                # J = Left (+1.0), L = Right (-1.0) 
                steering = float(self.key_handler[key.J] - self.key_handler[key.L])

                # Process inputs and display result
                user_actions[0, 0] = steering
                user_actions[0, 1] = throttle
                self.env.step(user_actions)
                self.render()

        self._clear()

    def render(self):
        # Warp Render Begin
        time = self.env._call * DT # Represent actual simulation time
        self.renderer.begin_frame(time)

        # Begin Render
        # TODO : Make this a ImGUI toggle for performance reasons
        if self.env.num_envs > 1:
            self._render_all_agents()
        else:
            self._render_user_car()

        self._render_user_lidar()
        # End Render

        # Warp Render End
        self.renderer.end_frame()

    def _setup_map(self):
        self._setup_map_ground()
        #self._setup_map_grid() # Disable to remove Moire pattern effect
        self._setup_map_walls()
        self._setup_map_center_line()

    def _setup_map_ground(self):
        physical_width = self.map.w * self.map.res
        physical_length = self.map.h * self.map.res

        center_x = self.map.ox + (physical_width / 2.0)
        center_y = self.map.oy + (physical_length / 2.0)

        self.renderer.render_plane(
            name="map_ground",
            pos=[center_x, center_y, 0.0],
            rot=np.array(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2.0)),
            width=physical_width / 2.0,
            length=physical_length / 2.0,
            color=(0.15, 0.15, 0.15)
        )

    def _setup_map_grid(self):
        pixel_size = self.map.res 
        half_pixel = pixel_size / 2.0  # The shift amount
        
        w = self.map.w
        h = self.map.h
        ox = self.map.ox
        oy = self.map.oy
        z_height = 0.0

        # Vertical lines
        # Shift all X coordinates back by half a pixel
        x_coords = ox + (np.arange(w + 1) * pixel_size) - half_pixel
        
        v_starts = np.column_stack([x_coords, np.full_like(x_coords, oy - half_pixel), np.full_like(x_coords, z_height)])
        v_ends = np.column_stack([x_coords, np.full_like(x_coords, oy + (h * pixel_size) - half_pixel), np.full_like(x_coords, z_height)])
        
        v_verts = np.empty((2 * (w + 1), 3), dtype=np.float32)
        v_verts[0::2] = v_starts
        v_verts[1::2] = v_ends

        # Horizontal lines
        # Shift all Y coordinates back by half a pixel
        y_coords = oy + (np.arange(h + 1) * pixel_size) - half_pixel
        
        h_starts = np.column_stack([np.full_like(y_coords, ox - half_pixel), y_coords, np.full_like(y_coords, z_height)])
        h_ends = np.column_stack([np.full_like(y_coords, ox + (w * pixel_size) - half_pixel), y_coords, np.full_like(y_coords, z_height)])
        
        h_verts = np.empty((2 * (h + 1), 3), dtype=np.float32)
        h_verts[0::2] = h_starts
        h_verts[1::2] = h_ends

        # Combine them
        self.grid_vertices = np.vstack([v_verts, h_verts])
        self.grid_indices = np.arange(len(self.grid_vertices), dtype=np.int32)

        # Render grid
        self.renderer.render_line_list(
            name="pixel_grid",
            vertices=self.grid_vertices,
            indices=self.grid_indices,
            color=(0, 0, 0),
            radius=0.005
        )
    
    def _setup_map_walls(self):
        dilated_free = binary_dilation(self.map.free)

        # The 1-pixel wall boundary is where the space is dilated, 
        # but was NOT part of the original free track.
        boundary_mask = dilated_free & ~self.map.free 

        # Extract the exact row/col coordinates of just that 1-pixel line
        boundary_pixels = np.argwhere(boundary_mask)

        rows = boundary_pixels[:, 0]
        cols = boundary_pixels[:, 1]

        # Convert to physical world coordinates
        wall_x = self.map.ox + cols * self.map.res
        wall_y = self.map.oy + (self.map.h - 1 - rows) * self.map.res

        # Build the 3D vertex array for Warp
        num_wall_points = len(boundary_pixels)
        self.wall_vertices = np.zeros((num_wall_points, 3), dtype=np.float32)
        self.wall_vertices[:, 0] = wall_x
        self.wall_vertices[:, 1] = wall_y
        self.wall_vertices[:, 2] = 0.05

        # Walls
        self.renderer.render_points(
            name="physics_walls",
            points=self.wall_vertices,
            colors=(1.0, 0.0, 0.0),
            radius=0.05
        )

    def _setup_map_center_line(self):
        num_points = len(self.map.centerline)
        self.track_vertices = np.zeros((num_points, 3), dtype=np.float32)
        self.track_vertices[:, 0] = self.map.centerline[:, 0]
        self.track_vertices[:, 1] = self.map.centerline[:, 1]
        self.track_vertices[:, 2] = 0.05
        
        # Center line
        self.renderer.render_points(
            name="center_line",
            points=self.track_vertices,
            colors=(0.0, 1.0, 0.0),
            radius=0.05
        )

    def _setup_dynamic_objects(self):
        # Single agent
        car_state = self.env.cars_buf[0].cpu().numpy()
        car_x, car_y, car_psi = car_state[0], car_state[1], car_state[4]
        car_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(car_psi))
        self.renderer.render_box(
            name="car_0",
            pos=[car_x, car_y, 0.15], 
            rot=np.array(car_rot),
            extents=[LENGTH / 2.0, WIDTH / 2.0, 0.1],
            color=(1.0, 1.0, 0.0)
        )
    
    def _render_all_agents(self):
        if not self.initialized_all_agents:
            all_car_states = self.env.cars_buf.cpu().numpy()
            
            for i in range(self.env.num_envs):
                percent = float(i / self.env.num_envs)
                car_state = all_car_states[i]
                car_x, car_y, car_psi = car_state[0], car_state[1], car_state[4]
                car_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(car_psi))

                car_name = f"car_{i}"

                self.renderer.render_box(
                    name=car_name,
                    pos=[car_x, car_y, 0.15], 
                    rot=np.array(car_rot),
                    extents=[LENGTH / 2.0, WIDTH / 2.0, 0.1]
                )

                # This is a bottleneck for over 8192 agents, but only way to color each agent
                car_color = (1.0 - percent, 1.0 - percent, percent)
                self.renderer.update_shape_instance(
                    name=car_name,
                    color1=car_color,
                    color2=car_color
                )
            
            print("_render_all_agents: self.initialized_all_agents = True")
            self.initialized_all_agents = True

        car_states = self.env.cars_buf.cpu().numpy()
        num_cars = len(car_states)
        
        # Extract arrays of X, Y, and Psi (heading)
        xs = car_states[:, 0]
        ys = car_states[:, 1]
        psis = car_states[:, 4]

        # Vectorize Quaternion Calculation
        # We create an (N, 3) array of Euler angles where only the Z-axis (yaw) is populated
        angles = np.zeros((num_cars, 3))
        angles[:, 2] = psis 
        
        # SciPy converts all 100 angles to quaternions in a single, lightning-fast C call.
        # Note: Physics psi is in radians, so degrees=False
        from scipy.spatial.transform import Rotation as R
        quats = R.from_euler('xyz', angles, degrees=False).as_quat()

        # Fast loop to ONLY update properties
        for i in range(num_cars):
            self.renderer.update_shape_instance(
                name=f"car_{i}",
                pos=[xs[i], ys[i], 0.15],
                rot=np.array(quats[i])
            )

    def _render_user_car(self):
        car_state = self.env.cars_buf[0].cpu().numpy()
        car_x, car_y, car_psi = car_state[0], car_state[1], car_state[4]
        car_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(car_psi))

        self.renderer.update_shape_instance(
            name="car_0",
            pos=[car_x, car_y, 0.15], 
            rot=np.array(car_rot)
        )

    def _render_user_lidar(self):
        """
        Zero-copy GPU rendering of all LiDAR points.
        """
        # We need the PyTorch tensors as Warp arrays for the kernel
        # wp.from_torch() is a zero-copy operation (instantaneous)
        obs_wp = wp.from_torch(self.env.obs_buf.contiguous(), dtype=wp.float32)
        cars_wp = wp.from_torch(self.env.cars_buf.contiguous(), dtype=wp.float32)
        
        # Launch the kernel with 1 thread per ray
        wp.launch(
            kernel=calc_single_lidar_hits_kernel,
            dim=NUM_LIDAR,
            inputs=[
                obs_wp,
                cars_wp,
                self.env.lidar_buf,
                self.lidar_hit_points
            ],
            device=self.env.device
        )
        
        # Pass the Warp array DIRECTLY to the renderer
        # Because we don't call .numpy(), the data never leaves the GPU!
        self.renderer.render_points(
            name="lidar_cloud_0",
            points=self.lidar_hit_points,
            radius=0.05,
            colors=(0.0, 1.0, 1.0)
        )

    def _clear(self):
        self.imgui_manager.shutdown()
        self.renderer.clear()

@wp.kernel
def calc_single_lidar_hits_kernel(
    obs: wp.array2d[wp.float32],
    cars: wp.array2d[wp.float32],
    lidar_dirs: wp.array[wp.vec2],
    hit_points: wp.array[wp.vec3]
):
    # Get the unique thread ID (0 to total_points - 1)
    tid = wp.tid()
    
    # Map the 1D thread ID back to a specific car and ray
    ray_idx = tid
    
    # Extract Car State
    car_x = cars[0, 0]
    car_y = cars[0, 1]
    car_psi = cars[0, 4]
    
    # True LiDAR origin
    lx = car_x + LF * wp.cos(car_psi)
    ly = car_y + LF * wp.sin(car_psi)
    
    # Extract Distance (starts at index 3)
    dist = obs[0, 3 + ray_idx]
    
    # Global Angle Math
    local_dir = lidar_dirs[ray_idx]
    local_angle = wp.atan2(local_dir[1], local_dir[0])
    global_angle = car_psi + local_angle
    
    # Calculate Hit coordinates
    hit_x = lx + dist * wp.cos(global_angle)
    hit_y = ly + dist * wp.sin(global_angle)
    hit_z = 0.150
    
    # Write directly to the render buffer
    hit_points[tid] = wp.vec3(hit_x, hit_y, hit_z)