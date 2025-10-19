import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio # For GIF
from io import BytesIO # To save plot to buffer
matplotlib.use('Agg')

# Import numba-optimized functions with graceful fallback
try:
    from envs.numba_utils import (
        NUMBA_AVAILABLE,
        calculate_plane_features_numba,
        calculate_g_numba,
        create_unpacked_boxes_state_numba,
        calculate_total_packed_volume_numba,
        get_max_height_numba
    )
    if NUMBA_AVAILABLE:
        print("[PackingEnv] Numba JIT compilation with parallel processing enabled - expect 175-350x speedup for plane features")
    else:
        print("[PackingEnv] Warning: Numba not available, using pure Python (slower)")
        calculate_plane_features_numba = None
        calculate_g_numba = None
        create_unpacked_boxes_state_numba = None
        calculate_total_packed_volume_numba = None
        get_max_height_numba = None
except ImportError:
    print("[PackingEnv] Warning: Could not import numba_utils, using pure Python (slower)")
    NUMBA_AVAILABLE = False
    calculate_plane_features_numba = None
    calculate_g_numba = None
    create_unpacked_boxes_state_numba = None
    calculate_total_packed_volume_numba = None
    get_max_height_numba = None

class PackingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi', 'rgb_array'], 'render_fps': 4}

    def __init__(self, container_dims=(100,100), initial_boxes=[], max_boxes=50, render_mode=None, gif_path="packing_process.gif"):
        super().__init__()

        self.container_L, self.container_W = container_dims
        self.initial_boxes_dims = [tuple(map(float,b)) for b in initial_boxes]
        self.max_boxes = len(self.initial_boxes_dims)

        self.render_mode = render_mode
        self.gif_path = gif_path
        self.frames_for_gif = []

        # --- Action Space ---
        self.action_space = spaces.Dict({
            "position": spaces.MultiDiscrete([self.container_L, self.container_W]),
            "box_select": spaces.Discrete(self.max_boxes),
            "orientation": spaces.Discrete(6)
        })

        # --- Observation Space ---
        max_dim_obs = max(self.container_L, self.container_W, 100.0) # Ensure float for Box space high
        self.observation_space = spaces.Dict({
            "unpacked_boxes_state": spaces.Box(
                low=0, high=max_dim_obs,
                shape=(self.max_boxes, 3), dtype=np.float32
            ),
            "container_state": spaces.Box(
                low=-max_dim_obs,
                high=max_dim_obs,
                shape=(self.container_L, self.container_W, 7), dtype=np.float32
            )
        })

        # Internal state
        self.unpacked_boxes = []
        self.packed_boxes_info = []
        self.container_height_map = np.zeros((self.container_L, self.container_W), dtype=np.float32)
        self.current_g = 0.0
        self.steps_taken = 0
        self.max_episode_steps = len(initial_boxes) * 3 # Increased to allow more exploration/failed attempts

        # For matplotlib rendering
        self._fig_render = None
        self._ax3d_render = None
        self._ax2d_render = None

    def _get_oriented_dimensions(self, box_dims, orientation_idx):
        l, w, h = box_dims
        if orientation_idx == 0: return (l, w, h) # (l, w, h)
        if orientation_idx == 1: return (l, h, w) # (l, h, w) - Rotated on X-axis
        if orientation_idx == 2: return (w, l, h) # (w, l, h) - Rotated on Z-axis
        if orientation_idx == 3: return (w, h, l) # (w, h, l) - Rotated on Z then X
        if orientation_idx == 4: return (h, l, w) # (h, l, w) - Rotated on Y-axis
        if orientation_idx == 5: return (h, w, l) # (h, w, l) - Rotated on Y then X
        raise ValueError(f"Invalid orientation index: {orientation_idx}")

    def directional_distance(self, h, w):
        arr = self.container_height_map
        hh, ww = arr.shape
        val = arr[h, w]
        result = {}

        # UP
        dist = 0
        for i in range(h - 1, -1, -1):
            if arr[i, w] != val:
                break
            dist += 1
        result['up'] = dist

        # DOWN 
        dist = 1
        for i in range(h + 1, hh):
            if arr[i, w] != val:
                break
            dist += 1
        result['down'] = dist

        # LEFT
        dist = 0
        for j in range(w - 1, -1, -1):
            if arr[h, j] != val:
                break
            dist += 1
        result['left'] = dist

        # RIGHT
        dist = 1
        for j in range(w + 1, ww):
            if arr[h, j] != val:
                break
            dist += 1
        result['right'] = dist

        # DOWN_NEXT
        dist = 1
        for i in range(h + 1, hh):
            if arr[i, w] > val:
                break
            dist += 1
        result['down_next'] = dist

        # RIGHT_NEXT
        dist = 1
        for j in range(w + 1, ww):
            if arr[h, j] > val:
                break
            dist += 1
        result['right_next'] = dist

        return result

    def _calculate_plane_features(self):
        """
        Calculate plane features for the entire container.

        Uses numba-optimized version if available (10-50x faster),
        otherwise falls back to pure Python implementation.
        """
        # Use numba-optimized version if available
        if NUMBA_AVAILABLE and calculate_plane_features_numba is not None:
            return calculate_plane_features_numba(self.container_height_map)

        # Fallback to pure Python implementation
        plane_features_map = np.zeros((self.container_L, self.container_W, 7), dtype=np.float32)
        for r in range(self.container_L):
            for c in range(self.container_W):
                current_h = self.container_height_map[r, c]
                plane_features_map[r, c, 0] = current_h
                features = self.directional_distance(r, c)
                plane_features_map[r, c, 1] = features['right']
                plane_features_map[r, c, 2] = features['down']
                plane_features_map[r, c, 3] = features['left']
                plane_features_map[r, c, 4] = features['up']
                plane_features_map[r, c, 5] = features['right_next']
                plane_features_map[r, c, 6] = features['down_next']

        return plane_features_map

    def _get_obs(self):
        # Use numba-optimized version if available for unpacked boxes state
        if NUMBA_AVAILABLE and create_unpacked_boxes_state_numba is not None:
            # Convert list to numpy array for numba function
            unpacked_boxes_array = np.array(self.unpacked_boxes, dtype=np.float32) if self.unpacked_boxes else np.empty((0, 3), dtype=np.float32)
            unpacked_state = create_unpacked_boxes_state_numba(unpacked_boxes_array, self.max_boxes)
        else:
            # Fallback to pure Python implementation
            unpacked_state = np.zeros((self.max_boxes, 3), dtype=np.float32)
            for i, box_dims in enumerate(self.unpacked_boxes):
                unpacked_state[i, :] = box_dims

        # Plane features are already optimized with parallel numba
        plane_features = self._calculate_plane_features()
        container_state = plane_features

        return {
            "unpacked_boxes_state": unpacked_state.astype(np.float32),
            "container_state": container_state.astype(np.float32)
        }

    def _get_info(self):
        # Use numba-optimized version if available for max height
        if NUMBA_AVAILABLE and get_max_height_numba is not None:
            max_h = get_max_height_numba(self.container_height_map)
        else:
            max_h = np.max(self.container_height_map) if self.container_height_map.any() else 0.0

        # Use numba-optimized version if available for total packed volume
        if NUMBA_AVAILABLE and calculate_total_packed_volume_numba is not None and self.packed_boxes_info:
            # Extract oriented dimensions into numpy array for numba function
            packed_boxes_dims = np.array([
                b['oriented_dims'] for b in self.packed_boxes_info
            ], dtype=np.float32)
            total_packed_volume = calculate_total_packed_volume_numba(packed_boxes_dims)
        else:
            # Fallback to pure Python implementation
            total_packed_volume = sum(b['oriented_dims'][0] * b['oriented_dims'][1] * b['oriented_dims'][2] for b in self.packed_boxes_info)

        # Calculate utilization rate
        container_volume = self.container_L * self.container_W * max_h
        utilization_rate = (total_packed_volume / container_volume) if container_volume > 0 else 0.0

        return {
            "max_packed_height": float(max_h),
            "num_packed_boxes": len(self.packed_boxes_info),
            "num_unpacked_boxes": len(self.unpacked_boxes),
            "total_packed_volume": float(total_packed_volume),
            "current_g": self.current_g,
            "steps_taken": self.steps_taken,
            "utilization_rate": float(utilization_rate)
        }

    def _calculate_g(self):
        """
        Calculate the gap metric 'g' (unused space).

        Uses numba-optimized version if available for faster computation.
        """
        if not self.packed_boxes_info:
            return 0.0

        # Use numba-optimized version if available
        if NUMBA_AVAILABLE and calculate_g_numba is not None:
            # Extract volumes into numpy array for numba function
            packed_volumes = np.array([
                b['oriented_dims'][0] * b['oriented_dims'][1] * b['oriented_dims'][2]
                for b in self.packed_boxes_info
            ], dtype=np.float32)
            return calculate_g_numba(
                self.container_height_map,
                packed_volumes,
                self.container_L,
                self.container_W
            )

        # Fallback to pure Python implementation
        max_h_stack = np.max(self.container_height_map) if self.container_height_map.any() else 0.0
        total_volume_packed_boxes = sum(b['oriented_dims'][0] * b['oriented_dims'][1] * b['oriented_dims'][2] for b in self.packed_boxes_info)
        return (self.container_L * self.container_W * max_h_stack) - total_volume_packed_boxes

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.unpacked_boxes = copy.deepcopy(self.initial_boxes_dims)
        # np.random.shuffle(self.unpacked_boxes) # Optional

        self.packed_boxes_info = []
        self.container_height_map = np.zeros((self.container_L, self.container_W), dtype=np.float32)
        self.current_g = self._calculate_g()
        self.steps_taken = 0
        self.frames_for_gif.clear()

        #if self.render_mode == "human":
        if self.initial_boxes_dims: # Capture initial empty state for GIF
            self.frames_for_gif.append(self.render(mode='rgb_array'))


        return self._get_obs(), self._get_info()

    def _is_valid_placement(self, box_dims, position):
        box_l, box_w, _ = box_dims # box_h is for updating height_map, not for footprint validity
        pos_x, pos_y = int(position[0]), int(position[1])

        if not (0 <= pos_x < self.container_L and 0 <= pos_y < self.container_W): return False, 0.0
        if not (pos_x + box_l <= self.container_L and pos_y + box_w <= self.container_W): return False, 0.0

        footprint_heights = self.container_height_map[pos_x : pos_x + int(box_l), pos_y : pos_y + int(box_w)]
        if footprint_heights.size == 0: return False, 0.0
        base_z = np.max(footprint_heights)
        return True, float(base_z)

    def step(self, action):
        self.steps_taken += 1
        pos_x, pos_y = int(action["position"][0]), int(action["position"][1])
        box_select_idx = action["box_select"]
        orientation_idx = action["orientation"]

        terminated = False
        reward = -0.01 # Small cost for taking a step
        box_successfully_placed = False

        selected_box_original_dims = self.unpacked_boxes[box_select_idx]
        oriented_dims = self._get_oriented_dimensions(selected_box_original_dims, orientation_idx)
        box_l, box_w, box_h = oriented_dims

        is_valid, base_z = self._is_valid_placement(oriented_dims, (pos_x, pos_y))

        if is_valid:
            new_top_z = base_z + box_h
            self.container_height_map[pos_x : pos_x + int(box_l), pos_y : pos_y + int(box_w)] = new_top_z
            box_to_pack = self.unpacked_boxes.pop(box_select_idx)
            self.packed_boxes_info.append({
                'id': len(self.initial_boxes_dims) - len(self.unpacked_boxes) -1,
                'orig_dims': box_to_pack, 'oriented_dims': oriented_dims,
                'pos': (float(pos_x), float(pos_y), float(base_z))
            })
            box_successfully_placed = True

            g_prev = self.current_g
            self.current_g = self._calculate_g()
            reward += (g_prev - self.current_g) # Positive if g decreased (better packing)

            if not self.unpacked_boxes: terminated = True
        else:
            reward -= np.float32(10e6) # Invalid placement

        if box_successfully_placed:
            self.frames_for_gif.append(self.render(mode='rgb_array'))

        truncated = self.steps_taken >= self.max_episode_steps
        if not self.unpacked_boxes: terminated = True

        observation = self._get_obs()
        info = self._get_info()

        # Standard render call if human mode is active (prints ANSI by default implementation below)
        if self.render_mode == "human":
            self.render()


        return observation, reward, terminated, truncated, info

    def _render_ansi(self):
        s = f"Step: {self.steps_taken}, Reward: ... (see return)\n" # Reward is complex to show here
        s += f"Unpacked: {len(self.unpacked_boxes)}, Packed: {len(self.packed_boxes_info)}\n"
        info = self._get_info()
        s += f"Max Height: {info['max_packed_height']:.2f}, Current g: {info['current_g']:.2f}\n"
        s += "Top view of container heights (rounded):\n"
        for r_idx in range(self.container_L):
            row_str = " ".join([f"{self.container_height_map[r_idx, c_idx]:.0f}" for c_idx in range(self.container_W)])
            s += row_str + "\n"
        s += "---\n"
        return s

    def _plot_3d_state(self, fig, ax, boxes_info, container_dims, title="3D Packing State"):
        ax.clear()
        L, W = container_dims
        ax.set_xlim(0, L)
        ax.set_ylim(0, W)
        max_h = np.max(self.container_height_map) if self.container_height_map.any() else 1.0 # Ensure zlim is reasonable
        ax.set_zlim(0, max(L, max_h)) # Add some padding to z-axis

        ax.set_xlabel('Container Length (X)')
        ax.set_ylabel('Container Width (Y)')
        ax.set_zlabel('Height (Z)')
        ax.set_title(title)

        # Use a colormap for different boxes
        cmap = plt.get_cmap('viridis')
        num_total_boxes = len(self.initial_boxes_dims)
        colors = [cmap(i / num_total_boxes) if num_total_boxes > 0 else cmap(0.5) for i in range(num_total_boxes)]


        for i, box_info in enumerate(boxes_info):
            x, y, z_base = box_info['pos']
            l, w, h = box_info['oriented_dims']
            # matplotlib's bar3d expects dx, dy, dz for depth, width, height
            # and x,y,z for the bottom-left-front corner
            # Our (l,w,h) are (dx,dy,dz) if x aligns with L, y with W
            color_idx = box_info.get('id', i) % len(colors) # Use box ID for consistent color
            ax.bar3d(x, y, z_base, l, w, h, color=colors[color_idx], edgecolor='black', alpha=0.7, shade=True)
        fig.tight_layout()


    def _plot_2d_height_map(self, fig, ax, height_map, title="2D Height Map"):
        ax.clear()
        im = ax.imshow(height_map.T, origin='lower', cmap='cividis', interpolation='nearest') # Transpose for (L,W) intuitive plot
        ax.set_xlabel('Container Length (X)')
        ax.set_ylabel('Container Width (Y)')
        ax.set_title(title)
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Stacked Height')
        fig.tight_layout()


    def render(self, mode=None):
        effective_mode = mode if mode is not None else self.render_mode

        if effective_mode == 'ansi':
            return self._render_ansi()
        elif effective_mode == 'rgb_array':
            # Consistent figsize. DPI can also be set if needed.
            temp_fig = plt.figure(figsize=(8, 6))

            temp_ax3d = temp_fig.add_subplot(111, projection='3d')
            self._plot_3d_state(temp_fig, temp_ax3d, self.packed_boxes_info, (self.container_L, self.container_W))

            image_rgb = None # Initialize

            try:
                temp_fig.canvas.draw()
                width, height = temp_fig.canvas.get_width_height()
                image_rgba = np.frombuffer(temp_fig.canvas.buffer_rgba(), dtype=np.uint8)
                image_rgba = image_rgba.reshape((height, width, 4))  # Ensure shape matches
                image_rgb = image_rgba[:, :, :3]


            except Exception as e1:
                print(f"Error with canvas.buffer_rgba(): {e1}. Falling back to tostring_argb/rgb.")
                try:
                    temp_fig.canvas.draw()
                    width, height = temp_fig.canvas.get_width_height() # Re-get dimensions
                    image_buffer_argb = temp_fig.canvas.tostring_argb() # This is the problematic one

                    # --- MODIFICATION FOR POTENTIAL FLOAT32 ---
                    # Calculate expected float32 buffer size
                    expected_float32_size = height * width * 4 * 4 # height * width * channels * bytes_per_float32
                    # Calculate expected uint16 buffer size
                    expected_uint16_size = height * width * 4 * 2 # height * width * channels * bytes_per_uint16


                    if len(image_buffer_argb) == expected_float32_size:
                        print("Interpreting tostring_argb buffer as float32 ARGB.")
                        image_argb_float32 = np.frombuffer(image_buffer_argb, dtype=np.float32).reshape((height, width, 4))
                        # Normalize from [0.0, 1.0] (typical for float image data) to [0, 255] uint8
                        # And handle ARGB to RGB (Mac ARGB is often BGRA byte order effectively in the buffer)
                        # Assuming standard RGBA float order first, then will adjust if colors are swapped
                        image_rgb_float32 = image_argb_float32[:, :, :3] # Take RGB, drop A
                        if np.max(image_rgb_float32) <= 1.0 and np.min(image_rgb_float32) >=0.0 : # Check if it's normalized
                             image_rgb = (image_rgb_float32 * 255).astype(np.uint8)
                        else: # if not normalized, it might be already in 0-255 range but as float
                             image_rgb = image_rgb_float32.astype(np.uint8)

                    elif len(image_buffer_argb) == expected_uint16_size:
                        print("Interpreting tostring_argb buffer as uint16 ARGB.")
                        image_argb_uint16 = np.frombuffer(image_buffer_argb, dtype=np.uint16).reshape((height, width, 4))
                        # Convert uint16 (0-65535) to uint8 (0-255) by scaling
                        image_rgb_uint16 = image_argb_uint16[:, :, :3] # Take RGB
                        image_rgb = (image_rgb_uint16 / 257).astype(np.uint8) # 65535 / 257 ~= 255
                    else:
                        # Original assumption: uint8 ARGB
                        print("Interpreting tostring_argb buffer as uint8 ARGB (original assumption).")
                        # This is the path that caused the ValueError before
                        image_rgba_uint8 = np.frombuffer(image_buffer_argb, dtype=np.uint8).reshape((height, width, 4))
                        image_rgb = image_rgba_uint8[:, :, :3].copy()


                except AttributeError as e_argb_attr:
                    print(f"tostring_argb attribute error: {e_argb_attr}. Trying tostring_rgb.")
                    try:
                        temp_fig.canvas.draw()
                        width, height = temp_fig.canvas.get_width_height()
                        image_buffer_rgb = temp_fig.canvas.tostring_rgb()
                        image_rgb = np.frombuffer(image_buffer_rgb, dtype=np.uint8).reshape((height, width, 3))
                        # print(f"Success with tostring_rgb: shape {image_rgb.shape}, dtype {image_rgb.dtype}")
                    except Exception as e_rgb:
                        print(f"FATAL: All canvas to image methods failed. Error with tostring_rgb: {e_rgb}")
                        width, height = temp_fig.canvas.get_width_height()
                        image_rgb = np.zeros((height, width, 3), dtype=np.uint8) # Blank image
                except ValueError as ve: # Catch reshape errors specifically for the string methods
                    print(f"ValueError during reshape with tostring_argb: {ve}")
                    print(f"Buffer size: {len(image_buffer_argb) if 'image_buffer_argb' in locals() else 'N/A'}")
                    print(f"Dimensions: w={width}, h={height}")
                    print(f"Expected uint8 ARGB: {height*width*4}, uint16 ARGB: {height*width*4*2}, float32 ARGB: {height*width*4*4}")
                    width, height = temp_fig.canvas.get_width_height()
                    image_rgb = np.zeros((height, width, 3), dtype=np.uint8) # Blank image

            plt.close(temp_fig)
            if image_rgb is None: # Should not happen if fallbacks work
                print("Warning: image_rgb is None, returning blank image.")
                width, height = plt.figure(figsize=(8,6)).canvas.get_width_height() # get default dims
                plt.close() # close the temp figure used for dims
                return np.zeros((height, width, 3), dtype=np.uint8)
            return image_rgb

        elif effective_mode == 'human':
            #print(self._render_ansi())
            return None

    

    # Place the set_axes_equal function somewhere above in your file
    def set_axes_equal(self, ax):
        '''Set 3D plot axes to equal scale (so that spheres appear as spheres, cubes as cubes, etc.)'''
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max([x_range, y_range, z_range])
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
        ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
        ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

    def display_final_state(self):
        """Helper to show final plots. Call after episode ends."""
        if not plt: return # Matplotlib not available

        final_fig = plt.figure(figsize=(12, 5))
        final_ax3d = final_fig.add_subplot(121, projection='3d')
        L, W = self.container_L, self.container_W
        max_h = np.max(self.container_height_map) if self.container_height_map.any() else 1.0
        final_ax2d = final_fig.add_subplot(122)

        self._plot_3d_state(final_fig, final_ax3d, self.packed_boxes_info, (self.container_L, self.container_W), "Final 3D Packing State")
        self._plot_2d_height_map(final_fig, final_ax2d, self.container_height_map, "Final 2D Height Map")

        self.set_axes_equal(final_ax3d)  # <-- Add this line

        plt.show()

    def save_gif(self, gif_path=None):
        if gif_path is not None:
            self.gif_path = gif_path
        """Saves the collected frames as a GIF."""
        if self.frames_for_gif:
            print(f"Saving GIF with {len(self.frames_for_gif)} frames to {self.gif_path}...")
            try:
                imageio.mimsave(self.gif_path, self.frames_for_gif, fps=max(1, int(self.metadata['render_fps'])))
                print(f"GIF saved to {self.gif_path}")
            except Exception as e:
                print(f"Error saving GIF: {e}")
                print("Ensure imageio and its dependencies (like Pillow) are correctly installed.")
        else:
            print("No frames captured for GIF.")


    def close(self):
        if self.render_mode == "human":
            self.save_gif() # Save GIF at the end of the episode if in human mode
            
            final_fig = plt.figure(figsize=(12, 5))
            final_ax3d = final_fig.add_subplot(121, projection='3d')
            L, W = self.container_L, self.container_W
            max_h = np.max(self.container_height_map) if self.container_height_map.any() else 1.0
            final_ax2d = final_fig.add_subplot(122)

            self._plot_3d_state(final_fig, final_ax3d, self.packed_boxes_info, (self.container_L, self.container_W), "Final 3D Packing State")
            self._plot_2d_height_map(final_fig, final_ax2d, self.container_height_map, "Final 2D Height Map")

            self.set_axes_equal(final_ax3d)  
            plt.savefig('final_packing_state.png')  # Save the final state as an image
            # self.display_final_state() # Optionally display final plots automatically on close

        self.frames_for_gif.clear()
        if self._fig_render:
            plt.close(self._fig_render)
            self._fig_render = None
            self._ax3d_render = None
            self._ax2d_render = None

