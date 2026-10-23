//! `SolidCvMimicApp`: Live Webcam-to-3D Humanoid (`SuperMimic` Unitree G1 29-DoF) &
//! 100% Pure-`solidcv` `TrackEverything` + `SolidCvBrepSolid` + `calc_optical_flow_pyr_lk` Studio.

use egui::epaint::Vertex;
use egui::{
    Color32, ColorImage, CornerRadius, FontFamily, FontId, Mesh, Pos2, Rect, RichText, Sense,
    Shape, Stroke, TextureHandle, TextureId, TextureOptions, Ui, Vec2,
};
use egui_plot::{Legend, Line, Plot, PlotPoints};
use solidcv::supermimic::{
    default_g1_joint_pos, mat3_vec3, parse_lafan1_csv, rpy_to_mat, vec3_add, vec3_norm, vec3_scale,
    vec3_sub, G1FkOutput, SolidCvKernelMode, SuperMimicState, WebcamSuperMimicSession,
    G1_BODY_NAMES, G1_JOINT_NAMES, NUM_J, NUM_MOVING_BODIES,
};

#[cfg(not(target_arch = "wasm32"))]
use std::io::Read;
#[cfg(not(target_arch = "wasm32"))]
use std::process::{Command, Stdio};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc, Mutex,
};

const WEBCAM_W: usize = 320;
const WEBCAM_H: usize = 180;
const TRACE_CAP: usize = 600;
const G1_MESH_BYTES: &[u8] = include_bytes!("../assets/g1_mesh_lod.bin");

#[derive(Clone, Copy)]
struct LocalCadTri {
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    normal: [f32; 3],
    shade: u8, // 0 = dark joint/visor (#333333), 1 = light shell (#b3b3b3)
}

struct G1CadMeshAsset {
    bodies: Vec<Vec<LocalCadTri>>,
}

impl G1CadMeshAsset {
    fn load_embedded() -> Self {
        let data = G1_MESH_BYTES;
        let mut off = 0usize;
        let read_u32 = |d: &[u8], o: &mut usize| -> u32 {
            let v = u32::from_le_bytes([d[*o], d[*o + 1], d[*o + 2], d[*o + 3]]);
            *o += 4;
            v
        };
        let read_f32 = |d: &[u8], o: &mut usize| -> f32 {
            let v = f32::from_bits(u32::from_le_bytes([d[*o], d[*o + 1], d[*o + 2], d[*o + 3]]));
            *o += 4;
            v
        };

        let n_bodies = if data.len() >= 4 {
            read_u32(data, &mut off) as usize
        } else {
            0
        };
        let mut bodies = Vec::with_capacity(n_bodies);
        for _ in 0..n_bodies {
            if off + 4 > data.len() {
                break;
            }
            let n_tris = read_u32(data, &mut off) as usize;
            let mut tris = Vec::with_capacity(n_tris);
            for _ in 0..n_tris {
                if off + 49 > data.len() {
                    break;
                }
                let v0 = [
                    read_f32(data, &mut off),
                    read_f32(data, &mut off),
                    read_f32(data, &mut off),
                ];
                let v1 = [
                    read_f32(data, &mut off),
                    read_f32(data, &mut off),
                    read_f32(data, &mut off),
                ];
                let v2 = [
                    read_f32(data, &mut off),
                    read_f32(data, &mut off),
                    read_f32(data, &mut off),
                ];
                let normal = [
                    read_f32(data, &mut off),
                    read_f32(data, &mut off),
                    read_f32(data, &mut off),
                ];
                let shade = data[off];
                off += 1;
                tris.push(LocalCadTri {
                    v0,
                    v1,
                    v2,
                    normal,
                    shade,
                });
            }
            bodies.push(tris);
        }
        Self { bodies }
    }
}

/// AS-000001 Three Appearances (`Dark`, `Light`, `Dim`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsfTheme {
    Dark,
    Light,
    Dim,
}

impl SsfTheme {
    pub fn label(self) -> &'static str {
        match self {
            Self::Dark => "Dark",
            Self::Light => "Light",
            Self::Dim => "Dim",
        }
    }

    pub fn bg(self) -> Color32 {
        match self {
            Self::Dark => Color32::from_rgb(0x0B, 0x0D, 0x10),
            Self::Light => Color32::from_rgb(0xF6, 0xF4, 0xEF),
            Self::Dim => Color32::from_rgb(0x14, 0x18, 0x1F),
        }
    }

    pub fn panel(self) -> Color32 {
        match self {
            Self::Dark => Color32::from_rgb(0x11, 0x14, 0x19),
            Self::Light => Color32::from_rgb(0xED, 0xEA, 0xE2),
            Self::Dim => Color32::from_rgb(0x1B, 0x20, 0x29),
        }
    }

    pub fn raised(self) -> Color32 {
        match self {
            Self::Dark => Color32::from_rgb(0x17, 0x1B, 0x22),
            Self::Light => Color32::from_rgb(0xE3, 0xDF, 0xD5),
            Self::Dim => Color32::from_rgb(0x23, 0x29, 0x35),
        }
    }

    pub fn border(self) -> Color32 {
        match self {
            Self::Dark => Color32::from_rgb(0x26, 0x2C, 0x36),
            Self::Light => Color32::from_rgb(0xC9, 0xC4, 0xB8),
            Self::Dim => Color32::from_rgb(0x31, 0x39, 0x47),
        }
    }

    pub fn ink(self) -> Color32 {
        match self {
            Self::Dark => Color32::from_rgb(0xEC, 0xEF, 0xF4),
            Self::Light => Color32::from_rgb(0x14, 0x17, 0x1C),
            Self::Dim => Color32::from_rgb(0xE5, 0xE9, 0xF0),
        }
    }

    pub fn muted(self) -> Color32 {
        match self {
            Self::Dark => Color32::from_rgb(0x94, 0x9E, 0xAE),
            Self::Light => Color32::from_rgb(0x52, 0x5A, 0x65),
            Self::Dim => Color32::from_rgb(0x9A, 0xA4, 0xB4),
        }
    }

    pub fn live(self) -> Color32 {
        match self {
            Self::Dark => Color32::from_rgb(0x00, 0xD4, 0xAA),
            Self::Light => Color32::from_rgb(0x00, 0x7A, 0x60),
            Self::Dim => Color32::from_rgb(0x20, 0xC9, 0xA0),
        }
    }

    pub fn vp_bg_bottom(self) -> Color32 {
        match self {
            Self::Dark => Color32::from_rgb(0x09, 0x0B, 0x0E),
            Self::Light => Color32::from_rgb(0xEC, 0xE9, 0xE1),
            Self::Dim => Color32::from_rgb(0x10, 0x14, 0x1A),
        }
    }

    pub fn vp_grid(self) -> Color32 {
        match self {
            Self::Dark => Color32::from_rgb(0x22, 0x28, 0x32),
            Self::Light => Color32::from_rgb(0xCE, 0xC9, 0xBD),
            Self::Dim => Color32::from_rgb(0x2B, 0x33, 0x40),
        }
    }
}

/// Active motion source driving the 3D G1 humanoid & scene viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionSourceMode {
    LiveWebcam,
    RecordedReplay,
    LafanDance1,
    LafanJumps1,
    LafanFallAndGetUp1,
}

impl MotionSourceMode {
    pub const ALL: [Self; 5] = [
        Self::LiveWebcam,
        Self::RecordedReplay,
        Self::LafanDance1,
        Self::LafanJumps1,
        Self::LafanFallAndGetUp1,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::LiveWebcam => "Live Webcam (Insta360 Link 2 · Pure solidcv)",
            Self::RecordedReplay => "Replay Recorded Webcam Clip",
            Self::LafanDance1 => "LAFAN1 · dance1_subject1 (G1 29-DoF)",
            Self::LafanJumps1 => "LAFAN1 · jumps1_subject1 (G1 29-DoF)",
            Self::LafanFallAndGetUp1 => "LAFAN1 · fallAndGetUp1_subject1 (G1 29-DoF)",
        }
    }
}

/// Overlay mode for the live webcam Picture-in-Picture HUD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebcamHudOverlay {
    RgbAndSkeleton,
    BitplaneXorMask,
    RealityComberFlow,
}

impl WebcamHudOverlay {
    pub const ALL: [Self; 3] = [
        Self::RgbAndSkeleton,
        Self::BitplaneXorMask,
        Self::RealityComberFlow,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::RgbAndSkeleton => "RGB + LK Corners + 14-Body Skeleton",
            Self::BitplaneXorMask => "XorMintermCarrier Bitplane Mask",
            Self::RealityComberFlow => "RealityComber + Pyramidal LK Flow",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TracePoint {
    pub t: f64,
    pub normalized_return: f64,
    pub e_body_pos_cm: f64,
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeWebcamCapture {
    latest_frame: Arc<Mutex<Option<(Vec<u8>, u64)>>>,
    stop_flag: Arc<AtomicBool>,
    connected: Arc<AtomicBool>,
    child_pid: Arc<AtomicU32>,
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeWebcamCapture {
    fn start() -> Self {
        let latest_frame: Arc<Mutex<Option<(Vec<u8>, u64)>>> = Arc::new(Mutex::new(None));
        let stop_flag = Arc::new(AtomicBool::new(false));
        let connected = Arc::new(AtomicBool::new(false));
        let child_pid = Arc::new(AtomicU32::new(0));

        let frame_clone = Arc::clone(&latest_frame);
        let stop_clone = Arc::clone(&stop_flag);
        let conn_clone = Arc::clone(&connected);
        let pid_clone = Arc::clone(&child_pid);

        std::thread::spawn(move || {
            let child = Command::new("ffmpeg")
                .args([
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-fflags",
                    "nobuffer",
                    "-flags",
                    "low_delay",
                    "-f",
                    "avfoundation",
                    "-framerate",
                    "30",
                    "-video_size",
                    "1280x720",
                    "-pixel_format",
                    "nv12",
                    "-i",
                    "0:none",
                    "-fps_mode",
                    "passthrough",
                    "-vf",
                    "scale=320:180",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-",
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn();

            if let Ok(mut proc) = child {
                pid_clone.store(proc.id(), Ordering::Relaxed);
                if let Some(mut stdout) = proc.stdout.take() {
                    let frame_bytes = WEBCAM_W * WEBCAM_H * 3;
                    let mut buf = vec![0u8; frame_bytes];
                    let mut seq = 0u64;
                    while !stop_clone.load(Ordering::Relaxed) {
                        if stdout.read_exact(&mut buf).is_ok() {
                            seq += 1;
                            conn_clone.store(true, Ordering::Relaxed);
                            if let Ok(mut guard) = frame_clone.lock() {
                                *guard = Some((buf.clone(), seq));
                            }
                        } else {
                            break;
                        }
                    }
                }
                let _ = proc.kill();
                let _ = proc.wait();
            }
            pid_clone.store(0, Ordering::Relaxed);
            conn_clone.store(false, Ordering::Relaxed);
        });

        Self {
            latest_frame,
            stop_flag,
            connected,
            child_pid,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for NativeWebcamCapture {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        let pid = self.child_pid.swap(0, Ordering::Relaxed);
        if pid > 0 {
            let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
        }
    }
}

pub struct SolidCvMimicApp {
    pub theme: SsfTheme,
    pub mode: MotionSourceMode,
    pub hud_overlay: WebcamHudOverlay,
    pub show_pip_skeleton: bool,
    pub session: WebcamSuperMimicSession,
    g1_mesh_asset: G1CadMeshAsset,

    // 3D Viewport camera state
    pub cam_azimuth_deg: f32,
    pub cam_elevation_deg: f32,
    pub cam_distance: f32,
    pub cam_lookat: [f32; 3],
    pub follow_pelvis: bool,

    // Interactive 3D body dragging ("drag the robot — it has to catch itself")
    pub dragged_body: Option<(usize, f32)>, // (body_idx, camera_depth_m)

    // Layer toggles
    pub show_solid_g1: bool,
    pub show_ref_ghost: bool,
    pub show_gt_twin: bool,
    pub show_voxels_3d: bool,
    pub show_brep_mesh_3d: bool,
    pub show_velocity_vectors: bool,
    pub show_joint_axes: bool,
    pub show_webcam_pip: bool,

    // Clips & replay
    pub lafan_dance: Vec<SuperMimicState>,
    pub lafan_jumps: Vec<SuperMimicState>,
    pub lafan_fall: Vec<SuperMimicState>,
    pub clip_cursor: usize,
    pub playback_paused: bool,
    pub playback_speed: f32,

    // Webcam & Telemetry
    pub last_rgb: Vec<u8>,
    pub last_webcam_seq: u64,
    pub webcam_tex: Option<TextureHandle>,
    pub trace: Vec<TracePoint>,
    pub sim_time: f64,
    pub export_status: String,

    #[cfg(not(target_arch = "wasm32"))]
    webcam_capture: NativeWebcamCapture,
}

impl SolidCvMimicApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let theme = SsfTheme::Dark;
        Self::apply_ssf_visuals(&cc.egui_ctx, theme);

        let lafan_dance = Self::load_or_synthesize_clip(
            "/Users/stu/git/supermimic_vibed/data/lafan1_g1/dance1_subject1.csv",
            0,
        );
        let lafan_jumps = Self::load_or_synthesize_clip(
            "/Users/stu/git/supermimic_vibed/data/lafan1_g1/jumps1_subject1.csv",
            1,
        );
        let lafan_fall = Self::load_or_synthesize_clip(
            "/Users/stu/git/supermimic_vibed/data/lafan1_g1/fallAndGetUp1_subject1.csv",
            2,
        );

        Self {
            theme,
            mode: MotionSourceMode::LiveWebcam,
            hud_overlay: WebcamHudOverlay::BitplaneXorMask,
            show_pip_skeleton: true,
            session: WebcamSuperMimicSession::new(),
            g1_mesh_asset: G1CadMeshAsset::load_embedded(),
            cam_azimuth_deg: -14.0,
            cam_elevation_deg: -12.0,
            cam_distance: 2.25,
            cam_lookat: [0.0, 0.0, 0.74],
            follow_pelvis: false,
            dragged_body: None,
            show_solid_g1: true,
            show_ref_ghost: true,
            show_gt_twin: false,
            show_voxels_3d: true,
            show_brep_mesh_3d: true,
            show_velocity_vectors: true,
            show_joint_axes: false,
            show_webcam_pip: true,
            lafan_dance,
            lafan_jumps,
            lafan_fall,
            clip_cursor: 0,
            playback_paused: false,
            playback_speed: 1.0,
            last_rgb: vec![20u8; WEBCAM_W * WEBCAM_H * 3],
            last_webcam_seq: 0,
            webcam_tex: None,
            trace: Vec::with_capacity(TRACE_CAP),
            sim_time: 0.0,
            export_status: String::new(),
            #[cfg(not(target_arch = "wasm32"))]
            webcam_capture: NativeWebcamCapture::start(),
        }
    }

    fn load_or_synthesize_clip(path: &str, variant: usize) -> Vec<SuperMimicState> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Ok(txt) = std::fs::read_to_string(path) {
            let parsed = parse_lafan1_csv(&txt, 1500);
            if !parsed.is_empty() {
                return parsed;
            }
        }
        let _ = path;
        let mut out = Vec::with_capacity(300);
        for f in 0..300 {
            let t = (f as f32) * 0.02;
            let mut st = SuperMimicState::default();
            let mut q = default_g1_joint_pos();
            match variant {
                0 => {
                    st.root_p = [
                        0.12 * (t * 2.2).sin(),
                        0.22 * (t * 1.6).sin(),
                        0.76 + 0.05 * (t * 4.4).cos(),
                    ];
                    st.root_r = rpy_to_mat([
                        0.06 * (t * 1.6).sin(),
                        0.08 * (t * 2.2).cos(),
                        0.45 * (t * 1.1).sin(),
                    ]);
                    q[12] = 0.35 * (t * 2.2).sin();
                    q[15] = -0.85 + 0.75 * (t * 3.2).sin();
                    q[16] = 0.65 + 0.45 * (t * 2.4).cos();
                    q[18] = 0.85 + 0.45 * (t * 3.2).cos();
                    q[22] = -0.85 - 0.75 * (t * 3.2).sin();
                    q[23] = -0.65 - 0.45 * (t * 2.4).cos();
                    q[25] = 0.85 - 0.45 * (t * 3.2).cos();
                }
                1 => {
                    let phase = (t * 2.4).sin();
                    let height_boost = phase.max(0.0) * 0.24;
                    let crouch = (-phase).max(0.0) * 0.18;
                    st.root_p = [0.0, 0.0, 0.78 + height_boost - crouch];
                    q[0] = -0.312 - crouch * 2.2;
                    q[3] = 0.669 + crouch * 3.8;
                    q[4] = -0.363 - crouch * 1.6;
                    q[6] = q[0];
                    q[9] = q[3];
                    q[10] = q[4];
                    q[15] = -0.4 - height_boost * 4.2;
                    q[22] = -0.4 - height_boost * 4.2;
                }
                _ => {
                    let c = 0.5 * (1.0 - (t * 1.5).cos());
                    st.root_p = [0.08 * c, 0.0, 0.78 - 0.26 * c];
                    st.root_r = rpy_to_mat([0.0, 0.38 * c, 0.0]);
                    q[0] = -0.312 - 0.85 * c;
                    q[3] = 0.669 + 1.35 * c;
                    q[4] = -0.363 - 0.50 * c;
                    q[6] = q[0];
                    q[9] = q[3];
                    q[10] = q[4];
                    q[14] = 0.32 * c;
                    q[15] = -0.95 * c;
                    q[22] = -0.95 * c;
                }
            }
            st.q = q;
            out.push(st);
        }
        out
    }

    pub fn apply_ssf_visuals(ctx: &egui::Context, theme: SsfTheme) {
        let mut visuals = match theme {
            SsfTheme::Light => egui::Visuals::light(),
            SsfTheme::Dark | SsfTheme::Dim => egui::Visuals::dark(),
        };
        visuals.panel_fill = theme.panel();
        visuals.window_fill = theme.panel();
        visuals.extreme_bg_color = theme.bg();
        visuals.faint_bg_color = theme.raised();
        visuals.override_text_color = Some(theme.ink());
        visuals.window_corner_radius = CornerRadius::ZERO;
        visuals.menu_corner_radius = CornerRadius::ZERO;
        visuals.window_stroke = Stroke::new(1.0_f32, theme.border());

        visuals.widgets.noninteractive.bg_fill = theme.panel();
        visuals.widgets.noninteractive.weak_bg_fill = theme.panel();
        visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, theme.border());
        visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, theme.ink());
        visuals.widgets.noninteractive.corner_radius = CornerRadius::ZERO;

        visuals.widgets.inactive.bg_fill = theme.raised();
        visuals.widgets.inactive.weak_bg_fill = theme.raised();
        visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, theme.border());
        visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, theme.ink());
        visuals.widgets.inactive.corner_radius = CornerRadius::ZERO;

        visuals.widgets.hovered.bg_fill = theme.raised();
        visuals.widgets.hovered.weak_bg_fill = theme.raised();
        visuals.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, theme.ink());
        visuals.widgets.hovered.fg_stroke = Stroke::new(1.0_f32, theme.ink());
        visuals.widgets.hovered.corner_radius = CornerRadius::ZERO;

        visuals.widgets.active.bg_fill = theme.ink();
        visuals.widgets.active.weak_bg_fill = theme.ink();
        visuals.widgets.active.bg_stroke = Stroke::new(1.0_f32, theme.ink());
        visuals.widgets.active.fg_stroke = Stroke::new(1.0_f32, theme.bg());
        visuals.widgets.active.corner_radius = CornerRadius::ZERO;

        visuals.selection.bg_fill = theme.ink();
        visuals.selection.stroke = Stroke::new(1.0_f32, theme.bg());

        ctx.set_visuals(visuals);
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Space) {
                self.playback_paused = !self.playback_paused;
            }
            if i.key_pressed(egui::Key::R) {
                let def = SuperMimicState::default();
                self.session.tracker.reset_to_reference(&def);
                self.session.reset_background();
            }
            if i.key_pressed(egui::Key::G) {
                self.show_ref_ghost = !self.show_ref_ghost;
            }
            if i.key_pressed(egui::Key::S) {
                self.show_gt_twin = !self.show_gt_twin;
            }
            if i.key_pressed(egui::Key::P) {
                self.session.tracker.policy_enabled = !self.session.tracker.policy_enabled;
            }
            if i.key_pressed(egui::Key::B) {
                self.show_brep_mesh_3d = !self.show_brep_mesh_3d;
            }
            if i.key_pressed(egui::Key::V) {
                self.show_voxels_3d = !self.show_voxels_3d;
            }
        });
    }

    fn step_simulation(&mut self, ctx: &egui::Context, dt: f64) {
        self.handle_shortcuts(ctx);
        if !self.playback_paused {
            self.sim_time += dt;
        }

        let mut got_new_webcam_frame = false;
        #[cfg(not(target_arch = "wasm32"))]
        if let Ok(mut guard) = self.webcam_capture.latest_frame.lock() {
            if let Some((rgb, seq)) = guard.take() {
                if seq != self.last_webcam_seq {
                    self.last_rgb = rgb;
                    self.last_webcam_seq = seq;
                    got_new_webcam_frame = true;
                }
            }
        }

        if !got_new_webcam_frame && self.last_webcam_seq == 0 {
            self.synthesize_fallback_webcam_frame();
            got_new_webcam_frame = true;
        }

        if self.mode == MotionSourceMode::LiveWebcam {
            if got_new_webcam_frame {
                let rgb_copy = self.last_rgb.clone();
                let _ = self.session.process_rgb_frame(&rgb_copy, WEBCAM_W, WEBCAM_H);
                self.update_webcam_texture(ctx);
            } else if !self.playback_paused {
                let target = self.session.tracker.ref_state.clone();
                let reward = self.session.tracker.step_50hz(&target);
                self.session.last_telemetry.reward = reward;
            }
        } else {
            if got_new_webcam_frame {
                self.update_webcam_texture(ctx);
            }
            if !self.playback_paused {
                let active_clip: &[SuperMimicState] = match self.mode {
                    MotionSourceMode::RecordedReplay => &self.session.recorded_frames,
                    MotionSourceMode::LafanDance1 => &self.lafan_dance,
                    MotionSourceMode::LafanJumps1 => &self.lafan_jumps,
                    MotionSourceMode::LafanFallAndGetUp1 => &self.lafan_fall,
                    MotionSourceMode::LiveWebcam => &[],
                };
                if !active_clip.is_empty() {
                    self.clip_cursor = (self.clip_cursor + 1) % active_clip.len();
                    let target = active_clip[self.clip_cursor].clone();
                    let reward = self.session.tracker.step_50hz(&target);
                    self.session.last_telemetry.reward = reward;
                }
            }
        }

        let tel = self.session.last_telemetry;
        self.trace.push(TracePoint {
            t: self.sim_time,
            normalized_return: tel.reward.normalized_return as f64,
            e_body_pos_cm: (tel.reward.e_body_pos.sqrt() * 100.0) as f64,
        });
        if self.trace.len() > TRACE_CAP {
            let drop_n = self.trace.len() - TRACE_CAP;
            self.trace.drain(0..drop_n);
        }
    }

    fn synthesize_fallback_webcam_frame(&mut self) {
        let t = self.sim_time as f32;
        let cx = 0.50 + 0.14 * (t * 1.3).sin();
        let lhx = cx - 0.18 - 0.16 * (t * 2.6).sin().max(0.0);
        let lhy = 0.48 - 0.28 * (t * 2.6).sin();
        let rhx = cx + 0.18 + 0.16 * (t * 2.2).cos().max(0.0);
        let rhy = 0.48 - 0.28 * (t * 2.2).cos();

        for y in 0..WEBCAM_H {
            let v = (y as f32) / (WEBCAM_H as f32);
            for x in 0..WEBCAM_W {
                let u = (x as f32) / (WEBCAM_W as f32);
                let idx = (y * WEBCAM_W + x) * 3;
                let mut r = 24.0 + 18.0 * (1.0 - v);
                let mut g = 28.0 + 20.0 * (1.0 - v);
                let mut b = 34.0 + 24.0 * (1.0 - v);

                if (u - cx).abs() < 0.11 && (v > 0.36 && v < 0.88) {
                    r = 68.0;
                    g = 86.0;
                    b = 112.0;
                }
                let d_head = ((u - cx).powi(2) + ((v - 0.25) * 1.3).powi(2)).sqrt();
                if d_head < 0.075 {
                    r = 215.0;
                    g = 162.0;
                    b = 134.0;
                }
                let d_lh = ((u - lhx).powi(2) + (v - lhy).powi(2)).sqrt();
                let d_rh = ((u - rhx).powi(2) + (v - rhy).powi(2)).sqrt();
                if d_lh < 0.055 || d_rh < 0.055 {
                    r = 222.0;
                    g = 168.0;
                    b = 138.0;
                }
                self.last_rgb[idx] = r as u8;
                self.last_rgb[idx + 1] = g as u8;
                self.last_rgb[idx + 2] = b as u8;
            }
        }
    }

    fn update_webcam_texture(&mut self, ctx: &egui::Context) {
        let n_pix = WEBCAM_W * WEBCAM_H;
        let mut pixels = Vec::with_capacity(n_pix);
        for i in 0..n_pix {
            let base = i * 4;
            if base + 3 < self.session.kernel_rgba.len() {
                pixels.push(Color32::from_rgba_unmultiplied(
                    self.session.kernel_rgba[base],
                    self.session.kernel_rgba[base + 1],
                    self.session.kernel_rgba[base + 2],
                    self.session.kernel_rgba[base + 3],
                ));
            } else {
                pixels.push(Color32::BLACK);
            }
        }

        let color_img = ColorImage {
            size: [WEBCAM_W, WEBCAM_H],
            pixels,
        };
        if let Some(tex) = &mut self.webcam_tex {
            tex.set(color_img, TextureOptions::NEAREST);
        } else {
            self.webcam_tex = Some(ctx.load_texture("webcam_hud", color_img, TextureOptions::NEAREST));
        }
    }
}

impl eframe::App for SolidCvMimicApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let dt = (ctx.input(|i| i.stable_dt as f64)).clamp(0.004, 0.05);
        self.step_simulation(ctx, dt);
        ctx.request_repaint();

        egui::TopBottomPanel::top("ssf_top_bar")
            .exact_height(42.0)
            .show(ctx, |ui| self.render_top_bar(ui));

        egui::SidePanel::left("ssf_left_controls")
            .resizable(true)
            .default_width(315.0)
            .min_width(275.0)
            .show(ctx, |ui| self.render_left_panel(ui));

        egui::SidePanel::right("ssf_right_telemetry")
            .resizable(true)
            .default_width(360.0)
            .min_width(310.0)
            .show(ctx, |ui| self.render_right_panel(ui));

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(self.theme.vp_bg_bottom()))
            .show(ctx, |ui| self.render_center_3d_viewport(ui));
    }
}

impl SolidCvMimicApp {
    fn render_top_bar(&mut self, ui: &mut Ui) {
        ui.horizontal_centered(|ui| {
            ui.add_space(6.0);
            let (dot_rect, _) = ui.allocate_exact_size(Vec2::splat(6.0), Sense::hover());
            let is_webcam_live = {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.webcam_capture.connected.load(Ordering::Relaxed)
                }
                #[cfg(target_arch = "wasm32")]
                {
                    true
                }
            };
            let dot_col = if is_webcam_live {
                self.theme.live()
            } else {
                self.theme.muted()
            };
            ui.painter().rect_filled(dot_rect, CornerRadius::ZERO, dot_col);

            ui.label(
                RichText::new("solidSF")
                    .font(FontId::new(15.0, FontFamily::Proportional))
                    .strong()
                    .color(self.theme.ink()),
            );
            ui.label(RichText::new("·").color(self.theme.muted()));
            ui.label(
                RichText::new("solidcv 3D × SuperMimic Whole-Body Studio (0-GPU Pure Rust)")
                    .font(FontId::new(13.5, FontFamily::Proportional))
                    .color(self.theme.ink()),
            );

            ui.separator();
            let status_txt = if is_webcam_live {
                format!(
                    "LIVE WEBCAM · Insta360 Link 2 (#{} · {} LK pts)",
                    self.last_webcam_seq,
                    self.session.last_telemetry.lk_tracked_points
                )
            } else {
                "SYNTHETIC STUDIO STREAM".to_string()
            };
            ui.label(
                RichText::new(status_txt)
                    .font(FontId::new(11.5, FontFamily::Monospace))
                    .color(dot_col),
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(8.0);
                for th in [SsfTheme::Dim, SsfTheme::Light, SsfTheme::Dark] {
                    let selected = self.theme == th;
                    let btn = egui::Button::new(
                        RichText::new(th.label())
                            .font(FontId::new(12.0, FontFamily::Proportional))
                            .color(if selected { self.theme.bg() } else { self.theme.ink() }),
                    )
                    .fill(if selected {
                        self.theme.ink()
                    } else {
                        self.theme.raised()
                    })
                    .corner_radius(CornerRadius::ZERO);
                    if ui.add(btn).clicked() {
                        self.theme = th;
                        Self::apply_ssf_visuals(ui.ctx(), th);
                    }
                }
                ui.label(
                    RichText::new("Appearance:")
                        .font(FontId::new(12.0, FontFamily::Proportional))
                        .color(self.theme.muted()),
                );
            });
        });
    }

    fn render_left_panel(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(4.0);
            ui.label(
                RichText::new("MOTION SOURCE & LIVE CAPTURE")
                    .font(FontId::new(11.0, FontFamily::Proportional))
                    .color(self.theme.muted()),
            );
            ui.add_space(4.0);

            egui::ComboBox::from_id_salt("motion_source_combo")
                .width(ui.available_width())
                .selected_text(self.mode.label())
                .show_ui(ui, |ui| {
                    for m in MotionSourceMode::ALL {
                        if ui.selectable_value(&mut self.mode, m, m.label()).clicked() {
                            self.clip_cursor = 0;
                        }
                    }
                });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let rec_label = if self.session.is_recording {
                    "Stop Recording"
                } else {
                    "Record Webcam"
                };
                let rec_btn = egui::Button::new(
                    RichText::new(rec_label).color(if self.session.is_recording {
                        self.theme.bg()
                    } else {
                        self.theme.ink()
                    }),
                )
                .fill(if self.session.is_recording {
                    self.theme.live()
                } else {
                    self.theme.raised()
                });
                if ui.add(rec_btn).clicked() {
                    if !self.session.is_recording {
                        self.session.recorded_frames.clear();
                        self.session.is_recording = true;
                        self.mode = MotionSourceMode::LiveWebcam;
                    } else {
                        self.session.is_recording = false;
                        if !self.session.recorded_frames.is_empty() {
                            self.mode = MotionSourceMode::RecordedReplay;
                            self.clip_cursor = 0;
                        }
                    }
                }

                if ui.button("Reset State (R)").clicked() {
                    let def = SuperMimicState::default();
                    self.session.tracker.reset_to_reference(&def);
                    self.session.reset_background();
                }
            });

            ui.label(
                RichText::new(format!(
                    "Recorded buffer: {} frames ({:.1} s @ 50 Hz)",
                    self.session.recorded_frames.len(),
                    (self.session.recorded_frames.len() as f32) * 0.02
                ))
                .font(FontId::new(11.5, FontFamily::Monospace))
                .color(self.theme.muted()),
            );

            ui.add_space(8.0);
            ui.separator();
            ui.label(
                RichText::new("WORLD MODEL & PHYSICS DOCK (SUPERMIMIC)")
                    .font(FontId::new(11.0, FontFamily::Proportional))
                    .color(self.theme.muted()),
            );
            ui.horizontal(|ui| {
                let pol_on = self.session.tracker.policy_enabled;
                let pol_btn = egui::Button::new(
                    RichText::new(if pol_on { "Policy: ON (P)" } else { "Policy: OFF (P)" }).color(
                        if pol_on {
                            self.theme.bg()
                        } else {
                            self.theme.ink()
                        },
                    ),
                )
                .fill(if pol_on {
                    self.theme.live()
                } else {
                    self.theme.raised()
                });
                if ui.add(pol_btn).clicked() {
                    self.session.tracker.policy_enabled = !pol_on;
                }

                let play_lbl = if self.playback_paused {
                    "Resume (Space)"
                } else {
                    "Pause (Space)"
                };
                if ui.button(play_lbl).clicked() {
                    self.playback_paused = !self.playback_paused;
                }
            });

            ui.add(
                egui::Slider::new(&mut self.session.tracker.action_noise, 0.0..=0.25)
                    .text("Action noise"),
            );
            ui.add(
                egui::Slider::new(&mut self.session.tracker.floor_friction, 0.05..=1.0)
                    .text("Floor friction"),
            );
            ui.add(
                egui::Slider::new(&mut self.session.tracker.stiffness_scale, 0.4..=3.0)
                    .text("BeyondMimic PD stiffness"),
            );

            if self.mode != MotionSourceMode::LiveWebcam {
                ui.add_space(6.0);
                let clip_len = match self.mode {
                    MotionSourceMode::RecordedReplay => self.session.recorded_frames.len().max(1),
                    MotionSourceMode::LafanDance1 => self.lafan_dance.len().max(1),
                    MotionSourceMode::LafanJumps1 => self.lafan_jumps.len().max(1),
                    MotionSourceMode::LafanFallAndGetUp1 => self.lafan_fall.len().max(1),
                    MotionSourceMode::LiveWebcam => 1,
                };
                let mut cursor = self.clip_cursor.min(clip_len - 1);
                if ui
                    .add(egui::Slider::new(&mut cursor, 0..=(clip_len - 1)).text("Clip frame"))
                    .changed()
                {
                    self.clip_cursor = cursor;
                }
            }

            ui.add_space(10.0);
            ui.separator();
            ui.label(
                RichText::new("3D VIEWPORT LAYERS & CAMERA")
                    .font(FontId::new(11.0, FontFamily::Proportional))
                    .color(self.theme.muted()),
            );
            ui.checkbox(
                &mut self.show_solid_g1,
                "World Model · 30-Body Unitree G1 CAD Mesh",
            );
            ui.checkbox(
                &mut self.show_ref_ghost,
                "Goal Pose (G) · Reference IK / Mocap Ghost",
            );
            ui.checkbox(
                &mut self.show_gt_twin,
                "Ground-Truth Twin (S) · Friction/Slip Physics",
            );
            ui.checkbox(
                &mut self.show_brep_mesh_3d,
                "solidcv BRep (B) · Watertight 3D Manifold",
            );
            ui.checkbox(
                &mut self.show_voxels_3d,
                "TrackEverything (V) · 3D Worldtube Cloud",
            );
            ui.checkbox(&mut self.show_velocity_vectors, "3D Worldtube velocity vectors");
            ui.checkbox(&mut self.show_joint_axes, "Show 29-DoF local joint axes");
            ui.checkbox(&mut self.follow_pelvis, "Lock camera look-at to G1 pelvis");

            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                if ui.button("Front Mirror").clicked() {
                    self.cam_azimuth_deg = 0.0;
                    self.cam_elevation_deg = -9.0;
                    self.cam_distance = 2.20;
                }
                if ui.button("3/4 Studio").clicked() {
                    self.cam_azimuth_deg = -18.0;
                    self.cam_elevation_deg = -12.0;
                    self.cam_distance = 2.30;
                }
                if ui.button("Side Profile").clicked() {
                    self.cam_azimuth_deg = 90.0;
                    self.cam_elevation_deg = -10.0;
                    self.cam_distance = 2.40;
                }
                if ui.button("Top Down").clicked() {
                    self.cam_azimuth_deg = 0.0;
                    self.cam_elevation_deg = -78.0;
                    self.cam_distance = 3.00;
                }
            });

            ui.add_space(10.0);
            ui.separator();
            ui.label(
                RichText::new("SOLIDCV 6-STAGE KERNEL & IK RETARGETING")
                    .font(FontId::new(11.0, FontFamily::Proportional))
                    .color(self.theme.muted()),
            );
            ui.checkbox(&mut self.session.mirror_webcam, "Mirror webcam horizontally");
            ui.checkbox(&mut self.show_webcam_pip, "Show Live Kernel Output Viewport (320x180)");
            ui.checkbox(&mut self.show_pip_skeleton, "Overlay 14-Body Skeleton on Kernel Viewport");

            egui::ComboBox::from_id_salt("hud_overlay_combo")
                .width(ui.available_width())
                .selected_text(self.session.kernel_mode.label())
                .show_ui(ui, |ui| {
                    for ov in SolidCvKernelMode::ALL {
                        ui.selectable_value(&mut self.session.kernel_mode, ov, ov.label());
                    }
                });

            let mut xor_th = self.session.xor_threshold as u32;
            if ui
                .add(egui::Slider::new(&mut xor_th, 4..=64).text("Temporal XOR threshold"))
                .changed()
            {
                self.session.xor_threshold = xor_th as u8;
            }
            ui.add(
                egui::Slider::new(&mut self.session.motion_gain, 0.5..=3.5)
                    .text("XorMinterm motion gain"),
            );
            ui.add(
                egui::Slider::new(&mut self.session.arm_reach_scale, 0.6..=2.5)
                    .text("Arm IK reach scale"),
            );

            ui.add_space(10.0);
            ui.separator();
            ui.label(
                RichText::new("CAD / GLB & TRAJECTORY EXPORT")
                    .font(FontId::new(11.0, FontFamily::Proportional))
                    .color(self.theme.muted()),
            );
            ui.horizontal(|ui| {
                if ui.button("Export 3D Scene (.glb)").clicked() {
                    self.export_glb_to_scratch();
                }
                if ui.button("Export G1 (.csv)").clicked() {
                    self.export_csv_to_scratch();
                }
            });
            if !self.export_status.is_empty() {
                ui.label(
                    RichText::new(&self.export_status)
                        .font(FontId::new(11.0, FontFamily::Monospace))
                        .color(self.theme.live()),
                );
            }
        });
    }

    fn export_glb_to_scratch(&mut self) {
        if let Ok(bytes) = self.session.export_current_scene_glb() {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let dir = "/Users/stu/solidsf-scratch/supermimic-webcam";
                let _ = std::fs::create_dir_all(dir);
                let path = format!("{dir}/solidcv_supermimic_scene.glb");
                if std::fs::write(&path, &bytes).is_ok() {
                    self.export_status = format!("Wrote {} bytes -> {}", bytes.len(), path);
                    return;
                }
            }
            self.export_status = format!("Built binary glTF 2.0 ({} bytes)", bytes.len());
        }
    }

    fn export_csv_to_scratch(&mut self) {
        let csv = self.session.export_recorded_csv();
        #[cfg(not(target_arch = "wasm32"))]
        {
            let dir = "/Users/stu/solidsf-scratch/supermimic-webcam";
            let _ = std::fs::create_dir_all(dir);
            let path = format!("{dir}/solidcv_supermimic_traj.csv");
            if std::fs::write(&path, csv.as_bytes()).is_ok() {
                self.export_status = format!("Wrote {} B -> {}", csv.len(), path);
                return;
            }
        }
        self.export_status = format!("Exported {} bytes of 36-col G1 CSV", csv.len());
    }

    fn render_right_panel(&mut self, ui: &mut Ui) {
        let tel = self.session.last_telemetry;
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(4.0);
            ui.label(
                RichText::new("SOLIDCV 3D & SUPERMIMIC TELEMETRY")
                    .font(FontId::new(11.0, FontFamily::Proportional))
                    .color(self.theme.muted()),
            );
            ui.add_space(4.0);

            egui::Grid::new("telemetry_kpi_grid")
                .num_columns(2)
                .spacing([16.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Compute vs. SONIC (128 GPUs)");
                    ui.label(
                        RichText::new("0 GPUs · Local Apple Silicon")
                            .font(FontId::new(11.5, FontFamily::Monospace))
                            .color(self.theme.live()),
                    );
                    ui.end_row();

                    ui.label("SuperMimic return (0..1)");
                    ui.label(
                        RichText::new(format!("{:.3}", tel.reward.normalized_return))
                            .font(FontId::new(12.5, FontFamily::Monospace))
                            .color(self.theme.live()),
                    );
                    ui.end_row();

                    ui.label("14-body pos error (RMSE)");
                    ui.label(
                        RichText::new(format!("{:.2} cm", tel.reward.e_body_pos.sqrt() * 100.0))
                            .font(FontId::new(12.0, FontFamily::Monospace)),
                    );
                    ui.end_row();

                    ui.label("TrackEverything 3D tracks");
                    ui.label(
                        RichText::new(format!(
                            "{} total ({} dyn)",
                            tel.track_everything_tracks, tel.track_everything_dynamic
                        ))
                        .font(FontId::new(12.0, FontFamily::Monospace)),
                    );
                    ui.end_row();

                    ui.label("Pyramidal LK corners");
                    ui.label(
                        RichText::new(format!("{} active pts", tel.lk_tracked_points))
                            .font(FontId::new(12.0, FontFamily::Monospace)),
                    );
                    ui.end_row();

                    ui.label("SolidCvBrepSolid 3D mesh");
                    ui.label(
                        RichText::new(format!("{} manifold tris", tel.brep_triangles))
                            .font(FontId::new(12.0, FontFamily::Monospace)),
                    );
                    ui.end_row();

                    ui.label("Betti topology (b0, b1, b2)");
                    ui.label(
                        RichText::new(format!(
                            "({}, {}, {})",
                            tel.betti_numbers.0, tel.betti_numbers.1, tel.betti_numbers.2
                        ))
                        .font(FontId::new(12.0, FontFamily::Monospace)),
                    );
                    ui.end_row();

                    ui.label("XorMinterm active pixels");
                    ui.label(
                        RichText::new(format!(
                            "{} / {}",
                            tel.xor_changed_pixels,
                            tel.total_pixels.max(WEBCAM_W * WEBCAM_H)
                        ))
                        .font(FontId::new(12.0, FontFamily::Monospace)),
                    );
                    ui.end_row();

                    ui.label("RealityComber static pruned");
                    ui.label(
                        RichText::new(format!("{:.1}%", tel.pruned_static_blocks_pct))
                            .font(FontId::new(12.0, FontFamily::Monospace)),
                    );
                    ui.end_row();
                });

            ui.add_space(8.0);
            ui.separator();
            ui.label(
                RichText::new("TRACKING RETURN & 14-BODY ERROR SCOPE")
                    .font(FontId::new(11.0, FontFamily::Proportional))
                    .color(self.theme.muted()),
            );

            let ret_pts: PlotPoints<'_> = self
                .trace
                .iter()
                .map(|p| [p.t, p.normalized_return])
                .collect();
            let err_pts: PlotPoints<'_> = self
                .trace
                .iter()
                .map(|p| [p.t, p.e_body_pos_cm * 0.05])
                .collect();

            Plot::new("supermimic_return_plot")
                .height(145.0)
                .legend(Legend::default())
                .include_y(0.0)
                .include_y(1.05)
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(ret_pts)
                            .name("Return r_track")
                            .color(Color32::from_rgb(0x00, 0xD4, 0xAA))
                            .width(1.8_f32),
                    );
                    plot_ui.line(
                        Line::new(err_pts)
                            .name("Body error (x20 cm)")
                            .color(Color32::from_rgb(0xE0, 0x6C, 0x75))
                            .width(1.4_f32),
                    );
                });

            ui.add_space(6.0);
            ui.separator();
            ui.label(
                RichText::new("UNITREE G1 · 29-DOF JOINT ANGLES (RAD)")
                    .font(FontId::new(11.0, FontFamily::Proportional))
                    .color(self.theme.muted()),
            );

            egui::Grid::new("g1_29dof_grid")
                .num_columns(3)
                .spacing([8.0, 3.0])
                .striped(true)
                .show(ui, |ui| {
                    for j in 0..NUM_J {
                        let name = G1_JOINT_NAMES[j].trim_end_matches("_joint");
                        let q_sim = self.session.tracker.sim_state.q[j];
                        let q_ref = self.session.tracker.ref_state.q[j];
                        let lo = self.session.tracker.kin.q_lo[j];
                        let hi = self.session.tracker.kin.q_hi[j];
                        let frac = ((q_sim - lo) / (hi - lo).max(1e-4)).clamp(0.0, 1.0);

                        ui.label(
                            RichText::new(name)
                                .font(FontId::new(10.5, FontFamily::Monospace))
                                .color(self.theme.muted()),
                        );
                        ui.label(
                            RichText::new(format!("{:+5.2}", q_sim))
                                .font(FontId::new(10.5, FontFamily::Monospace)),
                        );
                        let (bar_rect, _) =
                            ui.allocate_exact_size(Vec2::new(84.0, 8.0), Sense::hover());
                        ui.painter()
                            .rect_filled(bar_rect, CornerRadius::ZERO, self.theme.raised());
                        let fill_w = bar_rect.width() * frac;
                        let fill_rect = Rect::from_min_size(
                            bar_rect.min,
                            Vec2::new(fill_w, bar_rect.height()),
                        );
                        ui.painter()
                            .rect_filled(fill_rect, CornerRadius::ZERO, self.theme.ink());

                        let ref_frac = ((q_ref - lo) / (hi - lo).max(1e-4)).clamp(0.0, 1.0);
                        let rx = bar_rect.left() + bar_rect.width() * ref_frac;
                        ui.painter().line_segment(
                            [Pos2::new(rx, bar_rect.top()), Pos2::new(rx, bar_rect.bottom())],
                            Stroke::new(1.5_f32, Color32::from_rgb(0x4C, 0x9A, 0xFF)),
                        );
                        ui.end_row();
                    }
                });
        });
    }

    fn render_center_3d_viewport(&mut self, ui: &mut Ui) {
        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, Sense::click_and_drag());

        let sim_st = self.session.tracker.sim_state.clone();
        let gt_st = self.session.tracker.gt_state.clone();
        let ref_st = self.session.tracker.ref_state.clone();
        if self.follow_pelvis {
            self.cam_lookat[0] = sim_st.root_p[0];
            self.cam_lookat[1] = sim_st.root_p[1];
            self.cam_lookat[2] = 0.74;
        }

        let az = self.cam_azimuth_deg.to_radians();
        let el = self.cam_elevation_deg.to_radians();
        let cam_offset = [
            self.cam_distance * el.cos() * az.cos(),
            self.cam_distance * el.cos() * az.sin(),
            self.cam_distance * (-el.sin()),
        ];
        let cam_pos = vec3_add(self.cam_lookat, cam_offset);
        let fwd_raw = vec3_sub(self.cam_lookat, cam_pos);
        let fwd = vec3_scale(fwd_raw, 1.0 / vec3_norm(fwd_raw).max(1e-6));
        let world_up = [0.0f32, 0.0, 1.0];
        let right_raw = [
            fwd[1] * world_up[2] - fwd[2] * world_up[1],
            fwd[2] * world_up[0] - fwd[0] * world_up[2],
            fwd[0] * world_up[1] - fwd[1] * world_up[0],
        ];
        let right = vec3_scale(right_raw, 1.0 / vec3_norm(right_raw).max(1e-6));
        let up = [
            right[1] * fwd[2] - right[2] * fwd[1],
            right[2] * fwd[0] - right[0] * fwd[2],
            right[0] * fwd[1] - right[1] * fwd[0],
        ];

        let focal = rect.height() * 0.92;
        let center = rect.center();
        let project = |pt: [f32; 3]| -> Option<(Pos2, f32)> {
            let rel = vec3_sub(pt, cam_pos);
            let depth = rel[0] * fwd[0] + rel[1] * fwd[1] + rel[2] * fwd[2];
            if depth <= 0.12 {
                return None;
            }
            let rx = rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2];
            let ry = rel[0] * up[0] + rel[1] * up[1] + rel[2] * up[2];
            let sx = center.x + (rx / depth) * focal;
            let sy = center.y - (ry / depth) * focal;
            Some((Pos2::new(sx, sy), depth))
        };

        let unproject = |scr: Pos2, depth: f32| -> [f32; 3] {
            let rx = ((scr.x - center.x) / focal) * depth;
            let ry = ((center.y - scr.y) / focal) * depth;
            let mut p = cam_pos;
            for k in 0..3 {
                p[k] += fwd[k] * depth + right[k] * rx + up[k] * ry;
            }
            p
        };

        // Compute Forward Kinematics for Reference Ghost, Ground-Truth Twin & Simulated Policy Robot
        let kin = &self.session.tracker.kin;
        let sim_fk = kin.forward(
            sim_st.root_p,
            sim_st.root_r,
            sim_st.root_v,
            sim_st.root_w,
            &sim_st.q,
            &sim_st.qd,
        );
        let gt_fk = kin.forward(
            gt_st.root_p,
            gt_st.root_r,
            gt_st.root_v,
            gt_st.root_w,
            &gt_st.q,
            &gt_st.qd,
        );
        let ref_fk = kin.forward(
            ref_st.root_p,
            ref_st.root_r,
            ref_st.root_v,
            ref_st.root_w,
            &ref_st.q,
            &ref_st.qd,
        );

        // Handle Interactive 3D Body Dragging vs. Camera Orbit
        if response.drag_started_by(egui::PointerButton::Primary) {
            if let Some(ptr) = response.interact_pointer_pos() {
                let mut best_body: Option<(usize, f32, f32)> = None;
                for b_idx in 0..NUM_MOVING_BODIES {
                    if let Some((scr, depth)) = project(sim_fk.body_p[b_idx]) {
                        let dist_px = scr.distance(ptr);
                        if dist_px < 42.0 {
                            if best_body.map_or(true, |(_, _, best_d)| dist_px < best_d) {
                                best_body = Some((b_idx, depth, dist_px));
                            }
                        }
                    }
                }
                if let Some((b_idx, depth, _)) = best_body {
                    self.dragged_body = Some((b_idx, depth));
                }
            }
        }

        if response.dragged_by(egui::PointerButton::Primary) {
            if let Some((b_idx, depth)) = self.dragged_body {
                if let Some(ptr) = response.interact_pointer_pos() {
                    let target_3d = unproject(ptr, depth);
                    self.session.tracker.drag_perturbation = Some((b_idx, target_3d));
                }
            } else {
                let delta = response.drag_delta();
                self.cam_azimuth_deg -= delta.x * 0.45;
                self.cam_elevation_deg =
                    (self.cam_elevation_deg - delta.y * 0.35).clamp(-85.0, 25.0);
            }
        } else {
            self.dragged_body = None;
            self.session.tracker.drag_perturbation = None;
        }

        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            self.cam_lookat[1] += delta.x * 0.003;
            self.cam_lookat[2] = (self.cam_lookat[2] + delta.y * 0.003).clamp(0.1, 1.8);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.1 {
                self.cam_distance = (self.cam_distance * (1.0 - scroll * 0.0015)).clamp(0.9, 6.5);
            }
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, CornerRadius::ZERO, self.theme.vp_bg_bottom());

        // 1. Studio Ground Grid (z = 0)
        let grid_col = self.theme.vp_grid();
        let gx_center = (self.cam_lookat[0] * 2.0).round() * 0.5;
        let gy_center = (self.cam_lookat[1] * 2.0).round() * 0.5;
        for i in -8..=8 {
            let offset = (i as f32) * 0.25;
            if let (Some((p0, _)), Some((p1, _))) = (
                project([gx_center - 2.0, gy_center + offset, 0.0]),
                project([gx_center + 2.0, gy_center + offset, 0.0]),
            ) {
                painter.line_segment([p0, p1], Stroke::new(1.0_f32, grid_col));
            }
            if let (Some((p0, _)), Some((p1, _))) = (
                project([gx_center + offset, gy_center - 2.0, 0.0]),
                project([gx_center + offset, gy_center + 2.0, 0.0]),
            ) {
                painter.line_segment([p0, p1], Stroke::new(1.0_f32, grid_col));
            }
        }

        if let Some((o, _)) = project([0.0, 0.0, 0.002]) {
            if let Some((px, _)) = project([0.22, 0.0, 0.002]) {
                painter.line_segment(
                    [o, px],
                    Stroke::new(2.0_f32, Color32::from_rgb(0xE0, 0x52, 0x52)),
                );
            }
            if let Some((py, _)) = project([0.0, 0.22, 0.002]) {
                painter.line_segment(
                    [o, py],
                    Stroke::new(2.0_f32, Color32::from_rgb(0x4E, 0xC9, 0x78)),
                );
            }
            if let Some((pz, _)) = project([0.0, 0.0, 0.22]) {
                painter.line_segment(
                    [o, pz],
                    Stroke::new(2.0_f32, Color32::from_rgb(0x4C, 0x9A, 0xFF)),
                );
            }
        }

        // 2. Draw `solidcv::cad_export::SolidCvBrepSolid` 3D Manifold Mesh
        if self.show_brep_mesh_3d {
            self.draw_solidcv_brep_mesh(&painter, &project);
        }

        // 3. Draw `TrackEverything` 3D Voxelean & Worldtube Point Cloud
        if self.show_voxels_3d {
            self.draw_voxelean_point_cloud(&painter, &project, focal);
        }

        // 4. Draw Translucent Goal / Reference Ghost (30-Body Unitree G1 CAD Mesh)
        if self.show_ref_ghost {
            self.draw_g1_cad_mesh(&painter, &project, cam_pos, focal, &ref_fk, G1RenderStyle::GoalGhost);
        }

        // 5. Draw Ground-Truth Physics Twin if enabled
        if self.show_gt_twin {
            self.draw_g1_cad_mesh(&painter, &project, cam_pos, focal, &gt_fk, G1RenderStyle::GroundTruthSolid);
        }

        // 6. Draw World Model G1 Robot (30-Body Unitree G1 CAD Mesh)
        if self.show_solid_g1 {
            let style = if self.show_gt_twin {
                G1RenderStyle::WorldModelBlue
            } else {
                G1RenderStyle::StudioMetallic
            };
            self.draw_g1_cad_mesh(&painter, &project, cam_pos, focal, &sim_fk, style);
        }

        // 7. Draw Interactive 3D Drag Tether if dragging a body link
        if let Some((b_idx, target_xyz)) = self.session.tracker.drag_perturbation {
            if let (Some((p0, _)), Some((p1, _))) =
                (project(sim_fk.body_p[b_idx]), project(target_xyz))
            {
                painter.line_segment(
                    [p0, p1],
                    Stroke::new(2.5_f32, Color32::from_rgb(0x00, 0xD4, 0xAA)),
                );
                painter.rect_filled(
                    Rect::from_center_size(p0, Vec2::splat(7.0)),
                    CornerRadius::ZERO,
                    Color32::from_rgb(0x00, 0xD4, 0xAA),
                );
                painter.rect_filled(
                    Rect::from_center_size(p1, Vec2::splat(7.0)),
                    CornerRadius::ZERO,
                    Color32::from_rgb(0x4C, 0x9A, 0xFF),
                );
                painter.text(
                    Pos2::new(p1.x + 8.0, p1.y - 8.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("DRAGGING: {}", G1_BODY_NAMES[b_idx]),
                    FontId::new(11.0, FontFamily::Monospace),
                    self.theme.live(),
                );
            }
        }

        // 8. Draw Joint Axes if enabled
        if self.show_joint_axes {
            for i in 1..NUM_MOVING_BODIES {
                let bp = sim_fk.body_p[i];
                let br = sim_fk.body_r[i];
                let ax_w = mat3_vec3(br, kin.axes[i]);
                let tip = vec3_add(bp, vec3_scale(ax_w, 0.08));
                if let (Some((p0, _)), Some((p1, _))) = (project(bp), project(tip)) {
                    painter.line_segment(
                        [p0, p1],
                        Stroke::new(1.5_f32, Color32::from_rgb(0xF5, 0xC2, 0x42)),
                    );
                }
            }
        }

        // 9. Draw Live Webcam Picture-in-Picture HUD
        if self.show_webcam_pip {
            self.draw_webcam_pip_hud(&painter, rect);
        }

        // 10. Draw Viewport HUD Header & Legend Overlay
        self.draw_viewport_hud_badges(&painter, rect);
    }

    fn draw_solidcv_brep_mesh(
        &self,
        painter: &egui::Painter,
        project: &impl Fn([f32; 3]) -> Option<(Pos2, f32)>,
    ) {
        if let Some(brep) = &self.session.live_brep_solid {
            let verts = &brep.vertices;
            let stroke = Stroke::new(
                1.0_f32,
                Color32::from_rgba_unmultiplied(0x00, 0xD4, 0xAA, 95),
            );
            for (idx, face) in brep.faces.iter().enumerate() {
                if idx > 320 || face.outer_loop.len() < 3 {
                    break;
                }
                let map_v = |v: [f64; 3]| -> [f32; 3] {
                    [
                        (v[0] as f32) - 0.55,
                        (v[1] as f32) - 0.45,
                        (v[2] as f32) + 0.12,
                    ]
                };
                for k in 0..face.outer_loop.len() {
                    let i0 = face.outer_loop[k];
                    let i1 = face.outer_loop[(k + 1) % face.outer_loop.len()];
                    if i0 < verts.len() && i1 < verts.len() {
                        if let (Some((p0, _)), Some((p1, _))) =
                            (project(map_v(verts[i0])), project(map_v(verts[i1])))
                        {
                            painter.line_segment([p0, p1], stroke);
                        }
                    }
                }
            }
        }
    }

    fn draw_voxelean_point_cloud(
        &self,
        painter: &egui::Painter,
        project: &impl Fn([f32; 3]) -> Option<(Pos2, f32)>,
        focal: f32,
    ) {
        for vox in &self.session.voxels_3d {
            let p_world = [vox.pos[0] - 0.35, vox.pos[1], vox.pos[2]];
            if let Some((scr, depth)) = project(p_world) {
                let rad = ((0.012 * focal) / depth).clamp(1.8, 6.0);
                let col = if vox.is_dynamic {
                    Color32::from_rgba_unmultiplied(
                        vox.rgb[0].saturating_add(30),
                        vox.rgb[1].saturating_add(55),
                        vox.rgb[2].saturating_add(45),
                        235,
                    )
                } else {
                    Color32::from_rgba_unmultiplied(vox.rgb[0], vox.rgb[1], vox.rgb[2], 125)
                };
                painter.rect_filled(
                    Rect::from_center_size(scr, Vec2::splat(rad * 1.6)),
                    CornerRadius::ZERO,
                    col,
                );

                if self.show_velocity_vectors && vox.is_dynamic {
                    let tip_world = [
                        p_world[0] + vox.vel[0] * 0.22,
                        p_world[1] + vox.vel[1] * 0.22,
                        p_world[2] + vox.vel[2] * 0.22,
                    ];
                    if let Some((tip_scr, _)) = project(tip_world) {
                        painter.line_segment(
                            [scr, tip_scr],
                            Stroke::new(
                                1.2_f32,
                                Color32::from_rgba_unmultiplied(0x00, 0xD4, 0xAA, 200),
                            ),
                        );
                    }
                }
            }
        }
    }

    fn draw_g1_cad_mesh(
        &self,
        painter: &egui::Painter,
        project: &impl Fn([f32; 3]) -> Option<(Pos2, f32)>,
        cam_pos: [f32; 3],
        focal: f32,
        fk: &G1FkOutput,
        style: G1RenderStyle,
    ) {
        // Soft ground contact shadow under pelvis & feet
        if matches!(
            style,
            G1RenderStyle::StudioMetallic | G1RenderStyle::WorldModelBlue
        ) {
            for &(b_idx, rad_m) in &[(0usize, 0.14f32), (6, 0.08), (12, 0.08)] {
                let bp = fk.body_p[b_idx];
                if let Some((sp, depth)) = project([bp[0], bp[1], 0.002]) {
                    let r = ((rad_m * focal) / depth).clamp(5.0, 42.0);
                    painter.circle_filled(sp, r, Color32::from_rgba_unmultiplied(0, 0, 0, 58));
                }
            }
        }

        // Studio 3-point lighting directions in Z-up world frame
        let key_light = [0.48f32, -0.42, 0.77];
        let fill_light = [-0.62f32, 0.52, 0.58];

        let mut visible_tris: Vec<(f32, Pos2, Pos2, Pos2, Color32)> = Vec::with_capacity(4800);

        for (b_idx, body_tris) in self.g1_mesh_asset.bodies.iter().enumerate() {
            if b_idx >= NUM_MOVING_BODIES {
                break;
            }
            // For Ghost mode, subsample every 2nd triangle so it stays crisp and translucent
            let step = if style == G1RenderStyle::GoalGhost { 2 } else { 1 };
            let bp = fk.body_p[b_idx];
            let br = fk.body_r[b_idx];

            for (t_idx, tri) in body_tris.iter().enumerate() {
                if t_idx % step != 0 {
                    continue;
                }
                let nw = mat3_vec3(br, tri.normal);
                let w0 = vec3_add(bp, mat3_vec3(br, tri.v0));
                let view_dir = vec3_sub(cam_pos, w0);
                // Back-face culling
                let facing = nw[0] * view_dir[0] + nw[1] * view_dir[1] + nw[2] * view_dir[2];
                if facing <= 0.0 {
                    continue;
                }
                let w1 = vec3_add(bp, mat3_vec3(br, tri.v1));
                let w2 = vec3_add(bp, mat3_vec3(br, tri.v2));

                if let (Some((p0, d0)), Some((p1, d1)), Some((p2, d2))) =
                    (project(w0), project(w1), project(w2))
                {
                    let depth = (d0 + d1 + d2) * (1.0 / 3.0);
                    let ndotl_key =
                        (nw[0] * key_light[0] + nw[1] * key_light[1] + nw[2] * key_light[2]).max(0.0);
                    let ndotl_fill = (nw[0] * fill_light[0]
                        + nw[1] * fill_light[1]
                        + nw[2] * fill_light[2])
                        .max(0.0);
                    let shade_f = (0.30 + 0.58 * ndotl_key + 0.24 * ndotl_fill).clamp(0.18, 1.15);

                    let color = match style {
                        G1RenderStyle::StudioMetallic => {
                            let (base_r, base_g, base_b) = if tri.shade == 1 {
                                (198.0, 203.0, 212.0) // Brushed light metallic armor (#C6CBD4)
                            } else {
                                (58.0, 64.0, 74.0) // Dark carbon actuator / visor (#3A404A)
                            };
                            Color32::from_rgb(
                                (base_r * shade_f).clamp(0.0, 255.0) as u8,
                                (base_g * shade_f).clamp(0.0, 255.0) as u8,
                                (base_b * shade_f).clamp(0.0, 255.0) as u8,
                            )
                        }
                        G1RenderStyle::WorldModelBlue => {
                            let (base_r, base_g, base_b) = if tri.shade == 1 {
                                (93.0, 161.0, 229.0) // SuperMimic World-Model Blue (#5DA1E5)
                            } else {
                                (47.0, 112.0, 185.0)
                            };
                            Color32::from_rgba_unmultiplied(
                                (base_r * shade_f).clamp(0.0, 255.0) as u8,
                                (base_g * shade_f).clamp(0.0, 255.0) as u8,
                                (base_b * shade_f).clamp(0.0, 255.0) as u8,
                                215,
                            )
                        }
                        G1RenderStyle::GroundTruthSolid => {
                            let (base_r, base_g, base_b) = if tri.shade == 1 {
                                (182.0, 186.0, 192.0)
                            } else {
                                (54.0, 58.0, 66.0)
                            };
                            Color32::from_rgb(
                                (base_r * shade_f).clamp(0.0, 255.0) as u8,
                                (base_g * shade_f).clamp(0.0, 255.0) as u8,
                                (base_b * shade_f).clamp(0.0, 255.0) as u8,
                            )
                        }
                        G1RenderStyle::GoalGhost => Color32::from_rgba_unmultiplied(
                            (110.0 * shade_f).clamp(0.0, 255.0) as u8,
                            (185.0 * shade_f).clamp(0.0, 255.0) as u8,
                            (245.0 * shade_f).clamp(0.0, 255.0) as u8,
                            72,
                        ),
                    };

                    visible_tris.push((depth, p0, p1, p2, color));
                }
            }
        }

        // Painter's algorithm sort (far to near) and single-batch GPU `egui::Mesh` submission
        visible_tris.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut mesh = Mesh::with_texture(TextureId::default());
        mesh.vertices.reserve(visible_tris.len() * 3);
        mesh.indices.reserve(visible_tris.len() * 3);

        for (_, p0, p1, p2, color) in visible_tris {
            let base = mesh.vertices.len() as u32;
            mesh.vertices.push(Vertex {
                pos: p0,
                uv: Pos2::ZERO,
                color,
            });
            mesh.vertices.push(Vertex {
                pos: p1,
                uv: Pos2::ZERO,
                color,
            });
            mesh.vertices.push(Vertex {
                pos: p2,
                uv: Pos2::ZERO,
                color,
            });
            mesh.indices.push(base);
            mesh.indices.push(base + 1);
            mesh.indices.push(base + 2);
        }

        painter.add(Shape::mesh(mesh));
    }

    fn draw_webcam_pip_hud(&self, painter: &egui::Painter, vp_rect: Rect) {
        let pip_w = 384.0;
        let pip_h = 216.0;
        let sx = pip_w / (WEBCAM_W as f32);
        let sy = pip_h / (WEBCAM_H as f32);
        let header_h = 22.0;
        let footer_h = 22.0;

        let header_rect = Rect::from_min_size(
            Pos2::new(vp_rect.right() - pip_w - 16.0, vp_rect.top() + 16.0),
            Vec2::new(pip_w, header_h),
        );
        let pip_rect = Rect::from_min_size(
            Pos2::new(header_rect.left(), header_rect.bottom()),
            Vec2::new(pip_w, pip_h),
        );
        let caption_rect = Rect::from_min_size(
            Pos2::new(pip_rect.left(), pip_rect.bottom()),
            Vec2::new(pip_w, footer_h),
        );

        // Header bar matching app.solidsf.com SolidCv.tsx
        painter.rect_filled(header_rect, CornerRadius::ZERO, self.theme.panel());
        painter.rect_stroke(
            header_rect,
            CornerRadius::ZERO,
            Stroke::new(1.0_f32, self.theme.border()),
            egui::StrokeKind::Inside,
        );
        painter.text(
            Pos2::new(header_rect.left() + 8.0, header_rect.center().y),
            egui::Align2::LEFT_CENTER,
            "Live Kernel Output Viewport (320x180)",
            FontId::new(10.5, FontFamily::Proportional),
            self.theme.ink(),
        );
        painter.text(
            Pos2::new(header_rect.right() - 8.0, header_rect.center().y),
            egui::Align2::RIGHT_CENTER,
            format!("Seal: 0x{:016x}", self.session.last_telemetry.zobrist_feistel_seal),
            FontId::new(9.5, FontFamily::Monospace),
            self.theme.muted(),
        );

        painter.rect_filled(pip_rect, CornerRadius::ZERO, Color32::from_rgb(0x0B, 0x0E, 0x13));
        if let Some(tex) = &self.webcam_tex {
            painter.image(
                tex.id(),
                pip_rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }

        let to_pip = |uv: [f32; 2]| -> Pos2 {
            Pos2::new(
                (pip_rect.left() + uv[0] * sx).clamp(pip_rect.left(), pip_rect.right()),
                (pip_rect.top() + uv[1] * sy).clamp(pip_rect.top(), pip_rect.bottom()),
            )
        };

        // 1. Draw Foveated ROI boxes (`#20c9a0` / `#00d4aa`)
        let mode = self.session.kernel_mode;
        if matches!(
            mode,
            SolidCvKernelMode::Composite
                | SolidCvKernelMode::OrbB1Mma
                | SolidCvKernelMode::Se2Uplift
                | SolidCvKernelMode::BitPlaneXor
        ) {
            for (idx, roi) in self.session.foveated_rois.iter().enumerate() {
                let r_min = to_pip([roi.x as f32, roi.y as f32]);
                let r_max = to_pip([(roi.x + roi.w) as f32, (roi.y + roi.h) as f32]);
                let r_rect = Rect::from_min_max(r_min, r_max);
                let col = if roi.independent_motion {
                    self.theme.live()
                } else {
                    Color32::from_rgb(0x4C, 0x9A, 0xFF)
                };
                painter.rect_stroke(
                    r_rect,
                    CornerRadius::ZERO,
                    Stroke::new(1.2_f32, col),
                    egui::StrokeKind::Inside,
                );
                let lbl = format!("ROI#{} H={:.2}b", idx + 1, roi.entropy);
                let tag_rect = Rect::from_min_size(
                    Pos2::new(r_rect.left(), (r_rect.top() - 12.0).max(pip_rect.top())),
                    Vec2::new(82.0, 11.0),
                );
                painter.rect_filled(
                    tag_rect,
                    CornerRadius::ZERO,
                    Color32::from_rgba_unmultiplied(11, 14, 19, 220),
                );
                painter.text(
                    Pos2::new(tag_rect.left() + 3.0, tag_rect.center().y),
                    egui::Align2::LEFT_CENTER,
                    lbl,
                    FontId::new(8.5, FontFamily::Monospace),
                    col,
                );
            }
        }

        // 2. Draw ORB keypoints (`#ffb020` matched, `#9aa3b2` unmatched)
        if matches!(mode, SolidCvKernelMode::Composite | SolidCvKernelMode::OrbB1Mma) {
            for kp in &self.session.orb_keypoints {
                let p = to_pip([kp.x as f32, kp.y as f32]);
                let col = if kp.matched {
                    Color32::from_rgb(0xFF, 0xB0, 0x20)
                } else {
                    Color32::from_rgb(0x9A, 0xA3, 0xB2)
                };
                painter.rect_stroke(
                    Rect::from_center_size(p, Vec2::splat(4.0)),
                    CornerRadius::ZERO,
                    Stroke::new(1.0_f32, col),
                    egui::StrokeKind::Inside,
                );
            }
        }

        // 3. Draw 14-Body Skeleton + Raised Hand Trackers (`L-HAND`, `R-HAND`)
        if self.show_pip_skeleton {
            let sk = &self.session.skeleton_2d;
            let bones = [
                (sk.head, sk.torso),
                (sk.torso, sk.pelvis),
                (sk.torso, sk.left_shoulder),
                (sk.left_shoulder, sk.left_elbow),
                (sk.left_elbow, sk.left_wrist),
                (sk.torso, sk.right_shoulder),
                (sk.right_shoulder, sk.right_elbow),
                (sk.right_elbow, sk.right_wrist),
            ];
            for (a, b) in bones {
                painter.line_segment([to_pip(a), to_pip(b)], Stroke::new(2.0_f32, self.theme.live()));
            }
            for (pt, label) in [
                (sk.head, "HEAD"),
                (sk.left_wrist, "L-HAND"),
                (sk.right_wrist, "R-HAND"),
            ] {
                let pos = to_pip(pt);
                painter.rect_filled(
                    Rect::from_center_size(pos, Vec2::splat(6.0)),
                    CornerRadius::ZERO,
                    self.theme.live(),
                );
                painter.text(
                    Pos2::new(pos.x + 6.0, pos.y - 5.0),
                    egui::Align2::LEFT_CENTER,
                    label,
                    FontId::new(8.5, FontFamily::Monospace),
                    Color32::WHITE,
                );
            }
        }

        painter.rect_stroke(
            pip_rect,
            CornerRadius::ZERO,
            Stroke::new(1.0_f32, self.theme.border()),
            egui::StrokeKind::Inside,
        );

        // Footer bar matching app.solidsf.com SolidCv.tsx
        painter.rect_filled(caption_rect, CornerRadius::ZERO, self.theme.panel());
        painter.rect_stroke(
            caption_rect,
            CornerRadius::ZERO,
            Stroke::new(1.0_f32, self.theme.border()),
            egui::StrokeKind::Inside,
        );
        let tel = self.session.last_telemetry;
        painter.text(
            Pos2::new(caption_rect.left() + 8.0, caption_rect.center().y),
            egui::Align2::LEFT_CENTER,
            format!(
                "Frame #{:06} | Compute: {:.2} ms | Motion: {:.1}% | ORB: {}/{} | ROIs: {}",
                tel.frame_index,
                tel.compute_ms,
                tel.active_motion_ratio * 100.0,
                tel.matched_track_count,
                tel.orb_keypoint_count.max(1),
                tel.roi_count
            ),
            FontId::new(9.5, FontFamily::Monospace),
            self.theme.ink(),
        );
    }

    fn draw_viewport_hud_badges(&self, painter: &egui::Painter, vp_rect: Rect) {
        let top_badge = Rect::from_min_size(
            Pos2::new(vp_rect.left() + 16.0, vp_rect.top() + 16.0),
            Vec2::new(510.0, 52.0),
        );
        painter.rect_filled(top_badge, CornerRadius::ZERO, self.theme.panel());
        painter.rect_stroke(
            top_badge,
            CornerRadius::ZERO,
            Stroke::new(1.0_f32, self.theme.border()),
            egui::StrokeKind::Inside,
        );

        let tel = self.session.last_telemetry;
        painter.text(
            Pos2::new(top_badge.left() + 10.0, top_badge.top() + 16.0),
            egui::Align2::LEFT_CENTER,
            format!(
                "SOURCE: {}   |   STEP #{}   |   RETURN = {:.3}",
                self.mode.label(),
                self.session.tracker.step_count,
                tel.reward.normalized_return
            ),
            FontId::new(11.0, FontFamily::Monospace),
            self.theme.ink(),
        );
        painter.text(
            Pos2::new(top_badge.left() + 10.0, top_badge.top() + 36.0),
            egui::Align2::LEFT_CENTER,
            format!(
                "SOLIDCV: TRACKEVERYTHING ({} TRKS) · PYR-LK ({} PTS) · BREP ({} TRIS) · DRAG ANY ROBOT LINK",
                tel.track_everything_tracks,
                tel.lk_tracked_points,
                tel.brep_triangles
            ),
            FontId::new(10.0, FontFamily::Monospace),
            self.theme.muted(),
        );

        let bot_badge = Rect::from_min_size(
            Pos2::new(vp_rect.left() + 16.0, vp_rect.bottom() - 38.0),
            Vec2::new(720.0, 24.0),
        );
        painter.rect_filled(bot_badge, CornerRadius::ZERO, self.theme.panel());
        painter.rect_stroke(
            bot_badge,
            CornerRadius::ZERO,
            Stroke::new(1.0_f32, self.theme.border()),
            egui::StrokeKind::Inside,
        );
        painter.text(
            Pos2::new(bot_badge.left() + 10.0, bot_badge.center().y),
            egui::Align2::LEFT_CENTER,
            "DRAG ROBOT LINK: Perturb balance   |   [G] Goal Ghost   |   [S] Physics Twin   |   [P] Policy ON/OFF   |   [B] solidcv BRep",
            FontId::new(10.5, FontFamily::Monospace),
            self.theme.ink(),
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum G1RenderStyle {
    StudioMetallic,
    WorldModelBlue,
    GroundTruthSolid,
    GoalGhost,
}
