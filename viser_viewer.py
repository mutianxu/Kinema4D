import time
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm
import viser
import viser.transforms as tf


# =============================================================================
# Utils
# =============================================================================
def _to_rgb01(img: np.ndarray) -> np.ndarray:
    rgb = img.astype(np.float32)
    if rgb.max() > 1.5:
        rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)


def _infer_fov_from_hw_fx(H: int, W: int, fx: float | None) -> float:
    # if fx known: use same formula as your npz version
    if fx is not None and fx > 1e-6:
        return float(2 * np.arctan2(H / 2.0, fx))
    # fallback: keep your pkl default
    return float(np.deg2rad(60.0))


# =============================================================================
# PKL compatibility loader (your existing logic)
# =============================================================================
class _BasePklLoader:
    """Only for pickle loading. Use attributes from saved object."""
    def num_frames(self) -> int:
        if hasattr(self, "_num_frames"):
            return int(self._num_frames)
        if hasattr(self, "rgb_frames"):
            return len(self.rgb_frames)
        if hasattr(self, "xyz_frames"):
            return len(self.xyz_frames)
        if hasattr(self, "pcd"):
            return len(self.pcd)
        raise AttributeError(f"{self.__class__.__name__} has no frame count")

    def get_frame(self, i: int):
        rgb = self.rgb_frames[i]
        if hasattr(self, "xyz_frames"):
            pm = self.xyz_frames[i]
        else:
            pm = self.pcd[i]

        T = getattr(self, "T_world_camera", None)
        if T is None:
            T_i = np.eye(4, dtype=np.float32)
        else:
            T = np.asarray(T)
            T_i = T[i] if (T.ndim == 3 and T.shape[0] > i) else T

        return SimpleNamespace(
            rgb=rgb,
            pcd=pm.reshape(-1, 3),
            pcd_color=rgb.reshape(-1, 3),
            T_world_camera=T_i.astype(np.float32),
        )


class Mp4PointmapLoader(_BasePklLoader):
    pass


class NpzPointmapLoader(_BasePklLoader):
    pass


class _CompatUnpickler(pickle.Unpickler):
    _NAME_MAP = {
        "Mp4PointmapLoader": Mp4PointmapLoader,
        "NpzPointmapLoader": NpzPointmapLoader,
    }

    def find_class(self, module, name):
        if name in self._NAME_MAP:
            return self._NAME_MAP[name]
        return super().find_class(module, name)


def compat_load_pickle(path: str):
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


# =============================================================================
# Unified loader interface
# =============================================================================
class FrameProvider:
    """Abstract provider: returns per-frame (pts_world, colors01, R_wc, t_wc, image_u8)."""
    def __len__(self) -> int: ...
    def get_hw(self) -> tuple[int, int]: ...
    def get_fov_aspect(self) -> tuple[float, float]: ...
    def get(self, t: int) -> SimpleNamespace: ...


class NpzProvider(FrameProvider):
    def __init__(self, npz_path: Path, stride: int = 1, time_stride: int = 1):
        data = np.load(npz_path)
        images = data["images"]
        depths = data["depths"]
        cam_c2w = data["cam_c2w"]
        K = data["intrinsic"].copy()

        T_full, H_full, W_full = depths.shape
        self.T = T_full // time_stride

        self.images = images[::time_stride, ::stride, ::stride, :]
        self.depths = depths[::time_stride, ::stride, ::stride]
        self.cam_c2w = cam_c2w[::time_stride]

        self.H = H_full // stride
        self.W = W_full // stride

        K[0, 0] /= stride
        K[1, 1] /= stride
        K[0, 2] /= stride
        K[1, 2] /= stride

        self.K = K
        self.K_inv = np.linalg.inv(K)

        i, j = np.meshgrid(np.arange(self.W), np.arange(self.H), indexing="xy")
        self.pixels_h = np.stack([i, j, np.ones_like(i)], axis=-1).astype(np.float32)

    def __len__(self) -> int:
        return int(self.T)

    def get_hw(self) -> tuple[int, int]:
        return int(self.H), int(self.W)

    def get_fov_aspect(self) -> tuple[float, float]:
        fx = float(self.K[0, 0])
        fov = _infer_fov_from_hw_fx(self.H, self.W, fx)
        aspect = float(self.W / max(self.H, 1))
        return fov, aspect

    def get(self, t: int) -> SimpleNamespace:
        rgb01 = _to_rgb01(self.images[t])
        depth = self.depths[t].astype(np.float32)
        c2w = self.cam_c2w[t].astype(np.float32)

        H, W = self.H, self.W
        dirs = (self.K_inv @ self.pixels_h.reshape(-1, 3).T).T.reshape(H, W, 3)
        pts_cam = (dirs * depth[..., None]).reshape(-1, 3)
        colors01 = rgb01.reshape(-1, 3)

        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]

        R_wc = c2w[:3, :3]
        t_wc = c2w[:3, 3]
        img_u8 = (rgb01 * 255).astype(np.uint8)

        return SimpleNamespace(
            pts=pts_world.astype(np.float32),
            cols=colors01.astype(np.float32),
            R_wc=R_wc.astype(np.float32),
            t_wc=t_wc.astype(np.float32),
            img_u8=img_u8,
        )


class PklProvider(FrameProvider):
    def __init__(self, pkl_path: Path, max_frames: int = 200):
        self.loader = compat_load_pickle(str(pkl_path))
        self.T = min(int(self.loader.num_frames()), int(max_frames))
        f0 = self.loader.get_frame(0)
        self.H, self.W = int(f0.rgb.shape[0]), int(f0.rgb.shape[1])

    def __len__(self) -> int:
        return int(self.T)

    def get_hw(self) -> tuple[int, int]:
        return int(self.H), int(self.W)

    def get_fov_aspect(self) -> tuple[float, float]:
        fov = _infer_fov_from_hw_fx(self.H, self.W, fx=None)
        aspect = float(self.W / max(self.H, 1))
        return fov, aspect

    def get(self, t: int) -> SimpleNamespace:
        frame = self.loader.get_frame(t)
        rgb01 = _to_rgb01(frame.rgb)

        pts = np.asarray(frame.pcd, dtype=np.float32)
        cols01 = _to_rgb01(np.asarray(frame.pcd_color, dtype=np.float32))

        if cols01.shape[0] != pts.shape[0]:
            mean_c = cols01.reshape(-1, 3).mean(axis=0, keepdims=True)
            cols01 = np.repeat(mean_c, pts.shape[0], axis=0)

        T_wc = np.asarray(frame.T_world_camera, dtype=np.float32)
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]

        img_u8 = (rgb01 * 255).astype(np.uint8)

        return SimpleNamespace(
            pts=pts,
            cols=cols01.astype(np.float32),
            R_wc=R_wc,
            t_wc=t_wc,
            img_u8=img_u8,
        )


# =============================================================================
# Viewer (shared)
# =============================================================================
def run_viewer(
    provider: FrameProvider,
    point_size=0.002,
    cam_scale=0.05,
    axes_scale=0.1,
):
    T = len(provider)
    H, W = provider.get_hw()
    fov, aspect = provider.get_fov_aspect()

    server = viser.ViserServer()
    server.scene.set_up_direction("-z")

    # ---------------- GUI ----------------
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep", min=0, max=T - 1, step=1, initial_value=0, disabled=True
        )
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=30)
        gui_show_all = server.gui.add_checkbox("Show all frames", False)
        gui_stride = server.gui.add_slider(
            "Stride", min=1, max=T, step=1, initial_value=1, disabled=True
        )
        gui_show_frustum = server.gui.add_checkbox("Show camera frustum", True)

    with server.gui.add_folder("Recording"):
        gui_record = server.gui.add_button("Record Scene")

    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(np.array([np.pi / 2, 0, 0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )

    frame_nodes, frustum_nodes = [], []
    

    for t in tqdm(range(T), desc="Building scene"):
        fr = provider.get(t)

        node = server.scene.add_frame(
            f"/frames/t{t}",
            wxyz=tf.SO3.from_matrix(fr.R_wc).wxyz,
            position=fr.t_wc,
            show_axes=False,
        )
        frame_nodes.append(node)

        server.scene.add_point_cloud(
            f"/frames/t{t}/cloud", fr.pts, fr.cols, point_size=point_size
        )

        frustum = server.scene.add_camera_frustum(
            f"/frames/t{t}/frustum",
            fov=fov,
            aspect=aspect,
            scale=cam_scale,
            image=fr.img_u8,
            wxyz=tf.SO3.from_matrix(fr.R_wc).wxyz,
            position=fr.t_wc,
        )
        server.scene.add_frame(
            f"/frames/t{t}/frustum/axes",
            axes_length=cam_scale * axes_scale * 10,
            axes_radius=cam_scale * axes_scale,
        )
        frustum_nodes.append(frustum)

    # ---------------- Visibility control ----------------
    prev_t = int(gui_timestep.value)
    for i, f in enumerate(frame_nodes):
        f.visible = (i == prev_t)
        frustum_nodes[i].visible = bool(gui_show_frustum.value) and (i == prev_t)

    @gui_timestep.on_update
    def _(_):
        nonlocal prev_t
        if not gui_show_all.value:
            with server.atomic():
                frame_nodes[prev_t].visible = False
                frustum_nodes[prev_t].visible = False
                frame_nodes[int(gui_timestep.value)].visible = True
                frustum_nodes[int(gui_timestep.value)].visible = bool(gui_show_frustum.value)
            prev_t = int(gui_timestep.value)

    @gui_playing.on_update
    def _(_):
        gui_timestep.disabled = gui_playing.value

    @gui_show_all.on_update
    def _(_):
        gui_stride.disabled = not gui_show_all.value
        gui_playing.disabled = gui_show_all.value
        gui_timestep.disabled = gui_show_all.value
        with server.atomic():
            for i, f in enumerate(frame_nodes):
                f.visible = (i % int(gui_stride.value) == 0) if gui_show_all.value else (i == int(gui_timestep.value))
                frustum_nodes[i].visible = bool(gui_show_frustum.value) and f.visible

    @gui_stride.on_update
    def _(_):
        if gui_show_all.value:
            with server.atomic():
                for i, f in enumerate(frame_nodes):
                    f.visible = (i % int(gui_stride.value) == 0)
                    frustum_nodes[i].visible = bool(gui_show_frustum.value) and f.visible

    @gui_show_frustum.on_update
    def _(_):
        with server.atomic():
            for i in range(T):
                frustum_nodes[i].visible = bool(gui_show_frustum.value) and frame_nodes[i].visible

    # ---------------- Recording ----------------
    @gui_record.on_click
    def _(_):
        gui_record.disabled = True
        rec = server._start_scene_recording()
        rec.set_loop_start()

        frames_to_record = (
            [i for i in range(T) if i % int(gui_stride.value) == 0]
            if gui_show_all.value
            else list(range(T))
        )

        for i in frames_to_record:
            with server.atomic():
                for j, f in enumerate(frame_nodes):
                    if gui_show_all.value:
                        f.visible = (j % int(gui_stride.value) == 0)
                    else:
                        f.visible = (j == i)
                    frustum_nodes[j].visible = bool(gui_show_frustum.value) and f.visible
            server.flush()
            rec.insert_sleep(1.0 / float(gui_fps.value))

        with server.atomic():
            for f in frame_nodes:
                f.visible = False
            for f in frustum_nodes:
                f.visible = False
        server.flush()

        bs = rec.end_and_serialize()
        save_path = Path(f"./viser_result/recording_{int(time.time())}.viser")
        save_path.parent.mkdir(exist_ok=True)
        save_path.write_bytes(bs)
        print(f"\n✅ Saved: {save_path.resolve()}")
        gui_record.disabled = False

    print("\n[Viser] Open the shown URL in your browser.")
    while True:
        if gui_playing.value and (not gui_show_all.value):
            gui_timestep.value = (int(gui_timestep.value) + 1) % T
        time.sleep(1.0 / float(gui_fps.value))


# =============================================================================
# Entry
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to .npz or .pkl")
    parser.add_argument("--stride", type=int, default=1, help="(npz) spatial stride")
    parser.add_argument("--time_stride", type=int, default=1, help="(npz) temporal stride")
    parser.add_argument("--max_frames", type=int, default=200, help="(pkl) max frames")
    parser.add_argument("--point_size", type=float, default=0.002)
    parser.add_argument("--cam_scale", type=float, default=0.05)
    parser.add_argument("--axes_scale", type=float, default=0.1)
    args = parser.parse_args()

    path = Path(args.input)
    if path.suffix.lower() == ".npz":
        provider = NpzProvider(path, stride=args.stride, time_stride=args.time_stride)
    elif path.suffix.lower() == ".pkl":
        provider = PklProvider(path, max_frames=args.max_frames)
    else:
        raise ValueError(f"Unsupported input suffix: {path.suffix} (need .npz or .pkl)")

    run_viewer(
        provider,
        point_size=args.point_size,
        cam_scale=args.cam_scale,
        axes_scale=args.axes_scale,
    )


if __name__ == "__main__":
    main()
