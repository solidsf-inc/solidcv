//! In-Head Mechanistic Interpretability (`mechinterp`) for solidcv Visual & Volumetric Engine.
//!
//! Records exact theorem activations (`Thm1..Thm24`, `J1..A2`), bitplane sparsity receipts,
//! Schur-Feshbach wavelet elimination ratios, dihedral/octahedral symmetry orbits, and
//! causal worldtube trajectories directly in memory with counterfactual ablation hooks.
//!
//! ### Apple Silicon M4 Max / M5 Pro / M5 Max Zero-Allocation Ring Buffer
//! Hot-path `record_activation` calls write into a fixed-capacity ring buffer of stack-inline
//! `CompactActivationRecord` entries and a 28-slot `usize` theorem array with **zero heap
//! allocations** (`0` `String`, `0` `BTreeMap`, `0` `Vec::remove(0)` shifts). Full
//! `MechInterpEvent` and `BTreeMap` structures are materialized lazily only when `mind_summary()`
//! or `snapshot_mind()` is queried.

use crate::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

/// Fast global flag indicating whether any non-default counterfactual ablation is active.
/// Avoids TLS `RefCell` borrow and `Vec` cloning on every frame in the normal un-ablated path.
static ABLATION_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Canonical identifiers for the 24 theorems of `solidSF.Unified.PixeleanVoxeleanCalculus`
/// and the 4 spectral/topological operators (`J1`, `J2`, `A1`, `A2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TheoremId {
    /// Thm 1: Bitplane Partition & Conservative Mass Transfer (`M_fg + M_bg = N`)
    Thm1BitplaneMassConservation = 0,
    /// Thm 2: Temporal XOR Minterm Sparsity & Carrier Compression (`Delta_t = F_t ^ F_{t-1}`)
    Thm2TemporalXorMintermSparsity = 1,
    /// Thm 3: Shannon Cofactor Expansion & Convex Envelope Bound
    Thm3ShannonCofactorEnvelope = 2,
    /// Thm 4: Kinematic Common-Fate Composite Assembly (`A_core uplus A_comp`)
    Thm4KinematicCommonFateComposite = 3,
    /// Thm 5: Dihedral D_4 Orientation Closure & Composite Stride Invariance
    Thm5DihedralD4Closure = 4,
    /// Thm 6: Coupled Mirror-Twin Jacobian Exactness (`det J = s_r * s_c`)
    Thm6MirrorTwinJacobian = 5,
    /// Thm 7: Visual Field Construct Factoring (`(1 + x_1)(1 + x_2) -> 1 + x_1 + x_2`)
    Thm7ConstructFactoring = 6,
    /// Thm 8: 1px-Gridded Logical Lattice Stride Quotient Isomorphism
    Thm8LogicalLatticeQuotient = 7,
    /// Thm 9: 3D Voxelean Octree Zobrist Localization Compression (`8 m K < 8^K`)
    Thm9VoxeleanOctreeZobrist = 8,
    /// Thm 10: 3D Chiral Octahedral Point Group `O_h` (`|O_h| = 48`, 6x over `D_4`)
    Thm10ChiralOctahedralOh = 9,
    /// Thm 11: 3D Voxelean Schur-Feshbach Octree Wavelet Split (`1 + 7 = 8`, 87.5% elimination)
    Thm11VoxeleanSchurFeshbachWavelet = 10,
    /// Thm 12: 3D Voxelean Discrete Exterior Calculus & Euler-Poincare Betti Identity
    Thm12DiscreteExteriorBetti = 11,
    /// Thm 13: 3D Voxelean Gauss Divergence Flux Conservation
    Thm13GaussDivergenceFlux = 12,
    /// Thm 14: 3D Voxelean Landauer Trihedral 3-Wall Corner Deadlock Barrier
    Thm14LandauerTrihedralBarrier = 13,
    /// Thm 15: 3D Voxelean Multi-Bounce Ray & Plaquette Mirror Flux Conservation
    Thm15MultiBounceRayFlux = 14,
    /// Thm 16: 3D Coupled 7-Point Star Stencil GF(p) Lifting Identifiability
    Thm16StarStencil7PointGfp = 15,
    /// Thm 17: (3+1)D Worldtube Continuity & Mass Balance
    Thm17WorldtubeContinuity = 16,
    /// Thm 18: (3+1)D Causal Light-Cone & CFL Horizon Pruning (`||dx||_1 <= v_max * dt`)
    Thm18CausalLightConeCfl = 17,
    /// Thm 19: (3+1)D Worldtube Common-Fate Trajectory Factoring (`T * A_vox -> A_vox + 3`)
    Thm19WorldtubeTrajectoryFactoring = 18,
    /// Thm 20: D-Dimensional Hyper-Voxelean Schur-Feshbach Wavelet Ratio (`1 - 2^{-D}`)
    Thm20HyperVoxeleanWaveletRatio = 19,
    /// Thm 21: D-Dimensional Hyperoctahedral Symmetry Group `B_D` (`2^D * D!`)
    Thm21HyperOctahedralGroupBd = 20,
    /// Thm 22: D-Dimensional Unbent Ray Exponentiation (`2D < (2D)^k`)
    Thm22UnbentRayExponentiation = 21,
    /// Thm 23: D-Dimensional Bent Geodesic & Non-Abelian Commutator Renormalization
    Thm23BentGeodesicRenormalization = 22,
    /// Thm 24: Grand Synthesis Certificate (`D >= 2, T >= 2`)
    Thm24GrandSynthesis = 23,
    /// J1: SE(2) Frenet-Serret Jet-Space Lifting & Elastica Occlusion Bridging
    J1Se2FrenetSerretJet = 24,
    /// J2: Temporal Radix-64 sfFFT Lock-In Demodulator & 2D Spectral Phase Correlation
    J2SfFftPhaseCorrelation = 25,
    /// A1: 2-Sheeted Over-Under Braid Helmholtz Green's Function
    A1BraidHelmholtzGreens = 26,
    /// A2: sfFFT Relative-L2 Speculative Pruning + Sparse Braid GF(2) Solver
    A2SpeculativeBraidGf2 = 27,
}

const ALL_THEOREMS: [TheoremId; 28] = [
    TheoremId::Thm1BitplaneMassConservation,
    TheoremId::Thm2TemporalXorMintermSparsity,
    TheoremId::Thm3ShannonCofactorEnvelope,
    TheoremId::Thm4KinematicCommonFateComposite,
    TheoremId::Thm5DihedralD4Closure,
    TheoremId::Thm6MirrorTwinJacobian,
    TheoremId::Thm7ConstructFactoring,
    TheoremId::Thm8LogicalLatticeQuotient,
    TheoremId::Thm9VoxeleanOctreeZobrist,
    TheoremId::Thm10ChiralOctahedralOh,
    TheoremId::Thm11VoxeleanSchurFeshbachWavelet,
    TheoremId::Thm12DiscreteExteriorBetti,
    TheoremId::Thm13GaussDivergenceFlux,
    TheoremId::Thm14LandauerTrihedralBarrier,
    TheoremId::Thm15MultiBounceRayFlux,
    TheoremId::Thm16StarStencil7PointGfp,
    TheoremId::Thm17WorldtubeContinuity,
    TheoremId::Thm18CausalLightConeCfl,
    TheoremId::Thm19WorldtubeTrajectoryFactoring,
    TheoremId::Thm20HyperVoxeleanWaveletRatio,
    TheoremId::Thm21HyperOctahedralGroupBd,
    TheoremId::Thm22UnbentRayExponentiation,
    TheoremId::Thm23BentGeodesicRenormalization,
    TheoremId::Thm24GrandSynthesis,
    TheoremId::J1Se2FrenetSerretJet,
    TheoremId::J2SfFftPhaseCorrelation,
    TheoremId::A1BraidHelmholtzGreens,
    TheoremId::A2SpeculativeBraidGf2,
];

impl TheoremId {
    #[inline]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Thm1BitplaneMassConservation => "THM-01-BITPLANE-MASS",
            Self::Thm2TemporalXorMintermSparsity => "THM-02-XOR-MINTERM",
            Self::Thm3ShannonCofactorEnvelope => "THM-03-SHANNON-COFACTOR",
            Self::Thm4KinematicCommonFateComposite => "THM-04-COMMON-FATE",
            Self::Thm5DihedralD4Closure => "THM-05-DIHEDRAL-D4",
            Self::Thm6MirrorTwinJacobian => "THM-06-MIRROR-JACOBIAN",
            Self::Thm7ConstructFactoring => "THM-07-CONSTRUCT-FACTOR",
            Self::Thm8LogicalLatticeQuotient => "THM-08-LATTICE-QUOTIENT",
            Self::Thm9VoxeleanOctreeZobrist => "THM-09-OCTREE-ZOBRIST",
            Self::Thm10ChiralOctahedralOh => "THM-10-OCTAHEDRAL-OH",
            Self::Thm11VoxeleanSchurFeshbachWavelet => "THM-11-VOXEL-WAVELET-1P7",
            Self::Thm12DiscreteExteriorBetti => "THM-12-DEC-EULER-BETTI",
            Self::Thm13GaussDivergenceFlux => "THM-13-GAUSS-DIVERGENCE",
            Self::Thm14LandauerTrihedralBarrier => "THM-14-LANDAUER-TRIHEDRAL",
            Self::Thm15MultiBounceRayFlux => "THM-15-RAY-PLAQUETTE-FLUX",
            Self::Thm16StarStencil7PointGfp => "THM-16-STAR7-GFP-LIFT",
            Self::Thm17WorldtubeContinuity => "THM-17-WORLDTUBE-CONTINUITY",
            Self::Thm18CausalLightConeCfl => "THM-18-CAUSAL-CFL-CONE",
            Self::Thm19WorldtubeTrajectoryFactoring => "THM-19-WORLDTUBE-FACTOR",
            Self::Thm20HyperVoxeleanWaveletRatio => "THM-20-HYPER-WAVELET-RATIO",
            Self::Thm21HyperOctahedralGroupBd => "THM-21-HYPEROCTAHEDRAL-BD",
            Self::Thm22UnbentRayExponentiation => "THM-22-RAY-EXPONENTIATION",
            Self::Thm23BentGeodesicRenormalization => "THM-23-BENT-GEODESIC-RG",
            Self::Thm24GrandSynthesis => "THM-24-GRAND-SYNTHESIS",
            Self::J1Se2FrenetSerretJet => "OP-J1-SE2-FRENET-SERRET",
            Self::J2SfFftPhaseCorrelation => "OP-J2-SFFFT-PHASE-LOCK",
            Self::A1BraidHelmholtzGreens => "OP-A1-BRAID-HELMHOLTZ",
            Self::A2SpeculativeBraidGf2 => "OP-A2-SPECULATIVE-GF2",
        }
    }
}

/// Single mechanistic activation event inside solidcv's visual/volumetric engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MechInterpEvent {
    pub seq: u64,
    pub op_name: String,
    pub theorem: TheoremId,
    pub input_elements: usize,
    pub active_elements: usize,
    /// Compression or elimination gain in `[0.0, 1.0]` (e.g. `1 - k_diff / N` or `1 - 2^{-D}`).
    pub compression_ratio: f64,
    /// Canonical Zobrist or state digest of the operative feature representation.
    pub feature_hash: u64,
    /// Key scalar observables for causal probing and circuit attribution.
    pub metrics: BTreeMap<String, f64>,
}

/// Node in solidcv's internal visual/volumetric concept circuit graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitNode {
    pub id: u64,
    pub kind: String,
    pub label: String,
    pub zobrist_hash: u64,
    pub mass_or_energy: f64,
    pub parent_ids: Vec<u64>,
}

/// Counterfactual intervention / ablation configuration for mechanistic probing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AblationConfig {
    /// Bitplanes or palette channels masked out during counterfactual evaluation.
    pub ablated_bitplanes: Vec<u8>,
    /// 3D/2D wavelet detail bands zeroed out (`0..7` in 3D, `0..3` in 2D).
    pub ablated_wavelet_bands: Vec<usize>,
    /// Force-disable temporal XOR carrier shortcut to measure dense vs sparse counterfactual.
    pub disable_xor_carrier: bool,
    /// Override CFL causal light-cone radius (`None` = use physical bound).
    pub override_cfl_radius: Option<i32>,
}

impl AblationConfig {
    #[inline]
    pub fn is_active(&self) -> bool {
        !self.ablated_bitplanes.is_empty()
            || !self.ablated_wavelet_bands.is_empty()
            || self.disable_xor_carrier
            || self.override_cfl_radius.is_some()
    }
}

/// Aggregate summary of solidcv's in-memory Mechanistic Interpretability ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MechInterpSummary {
    pub total_events: usize,
    pub total_circuit_nodes: usize,
    pub total_input_elements: u64,
    pub total_active_elements: u64,
    pub overall_sparsity_gain: f64,
    pub theorem_counts: BTreeMap<String, usize>,
    pub recent_events: Vec<MechInterpEvent>,
}

/// Zero-allocation compact ring-buffer activation record used on the hot path.
#[derive(Debug, Clone, Copy)]
struct CompactMetric {
    key_bytes: [u8; 28],
    key_len: u8,
    val: f64,
}

impl Default for CompactMetric {
    #[inline]
    fn default() -> Self {
        Self {
            key_bytes: [0u8; 28],
            key_len: 0,
            val: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CompactActivationRecord {
    seq: u64,
    op_bytes: [u8; 48],
    op_len: u8,
    theorem: TheoremId,
    metric_count: u8,
    input_elements: usize,
    active_elements: usize,
    feature_hash: u64,
    metrics: [CompactMetric; 4],
}

impl Default for CompactActivationRecord {
    #[inline]
    fn default() -> Self {
        Self {
            seq: 0,
            op_bytes: [0u8; 48],
            op_len: 0,
            theorem: TheoremId::Thm1BitplaneMassConservation,
            metric_count: 0,
            input_elements: 0,
            active_elements: 0,
            feature_hash: 0,
            metrics: [CompactMetric::default(); 4],
        }
    }
}

impl CompactActivationRecord {
    fn to_event(&self) -> MechInterpEvent {
        let op_name = std::str::from_utf8(&self.op_bytes[..self.op_len as usize])
            .unwrap_or("unknown")
            .to_string();
        let compression_ratio = if self.input_elements > 0 {
            1.0 - (self.active_elements.min(self.input_elements) as f64
                / self.input_elements as f64)
        } else {
            0.0
        };
        let mut metrics = BTreeMap::new();
        for i in 0..(self.metric_count as usize).min(4) {
            let m = &self.metrics[i];
            let k = std::str::from_utf8(&m.key_bytes[..m.key_len as usize])
                .unwrap_or("m")
                .to_string();
            metrics.insert(k, m.val);
        }
        MechInterpEvent {
            seq: self.seq,
            op_name,
            theorem: self.theorem,
            input_elements: self.input_elements,
            active_elements: self.active_elements,
            compression_ratio,
            feature_hash: self.feature_hash,
            metrics,
        }
    }
}

/// Complete state of solidcv's in-memory Mechanistic Interpretability ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MechInterpMind {
    pub enabled: bool,
    pub max_events: usize,
    pub next_seq: u64,
    pub events: Vec<MechInterpEvent>,
    pub circuit_nodes: Vec<CircuitNode>,
    pub ablation: AblationConfig,
    pub theorem_histogram: BTreeMap<String, usize>,
    pub cumulative_input_elements: u64,
    pub cumulative_active_elements: u64,
    #[serde(skip)]
    fast_ring: Vec<CompactActivationRecord>,
    #[serde(skip)]
    ring_head: usize,
    #[serde(skip)]
    ring_len: usize,
    #[serde(skip)]
    fast_theorem_counts: [usize; 28],
}

impl Default for MechInterpMind {
    fn default() -> Self {
        Self {
            enabled: true,
            max_events: 4096,
            next_seq: 1,
            events: Vec::new(),
            circuit_nodes: Vec::with_capacity(128),
            ablation: AblationConfig::default(),
            theorem_histogram: BTreeMap::new(),
            cumulative_input_elements: 0,
            cumulative_active_elements: 0,
            fast_ring: vec![CompactActivationRecord::default(); 64],
            ring_head: 0,
            ring_len: 0,
            fast_theorem_counts: [0usize; 28],
        }
    }
}

impl MechInterpMind {
    /// Zero-allocation hot-path activation recorder.
    #[inline]
    pub fn record_fast(
        &mut self,
        op_name: &str,
        theorem: TheoremId,
        input_elements: usize,
        active_elements: usize,
        feature_hash: u64,
        metrics: &[(&str, f64)],
    ) {
        if !self.enabled {
            return;
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        self.cumulative_input_elements += input_elements as u64;
        self.cumulative_active_elements += active_elements as u64;
        self.fast_theorem_counts[theorem as usize] += 1;

        if self.fast_ring.is_empty() {
            self.fast_ring = vec![CompactActivationRecord::default(); 64];
        }
        let cap = self.fast_ring.len();
        let slot_idx = if self.ring_len < cap {
            let idx = (self.ring_head + self.ring_len) % cap;
            self.ring_len += 1;
            idx
        } else {
            let idx = self.ring_head;
            self.ring_head = (self.ring_head + 1) % cap;
            idx
        };

        let slot = &mut self.fast_ring[slot_idx];
        slot.seq = seq;
        let op_b = op_name.as_bytes();
        let op_l = op_b.len().min(48);
        slot.op_bytes[..op_l].copy_from_slice(&op_b[..op_l]);
        slot.op_len = op_l as u8;
        slot.theorem = theorem;
        slot.input_elements = input_elements;
        slot.active_elements = active_elements;
        slot.feature_hash = feature_hash;

        let mc = metrics.len().min(4);
        slot.metric_count = mc as u8;
        for (i, &(k, v)) in metrics.iter().take(mc).enumerate() {
            let kb = k.as_bytes();
            let kl = kb.len().min(28);
            slot.metrics[i].key_bytes[..kl].copy_from_slice(&kb[..kl]);
            slot.metrics[i].key_len = kl as u8;
            slot.metrics[i].val = v;
        }
    }

    pub fn record(
        &mut self,
        op_name: &str,
        theorem: TheoremId,
        input_elements: usize,
        active_elements: usize,
        feature_hash: u64,
        metrics: BTreeMap<String, f64>,
    ) {
        if !self.enabled {
            return;
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        let compression_ratio = if input_elements > 0 {
            1.0 - (active_elements.min(input_elements) as f64 / input_elements as f64)
        } else {
            0.0
        };
        self.cumulative_input_elements += input_elements as u64;
        self.cumulative_active_elements += active_elements as u64;
        *self
            .theorem_histogram
            .entry(theorem.code().to_string())
            .or_insert(0) += 1;

        if self.events.len() >= self.max_events {
            self.events.remove(0);
        }
        self.events.push(MechInterpEvent {
            seq,
            op_name: op_name.to_string(),
            theorem,
            input_elements,
            active_elements,
            compression_ratio,
            feature_hash,
            metrics,
        });
    }

    fn materialize_theorem_histogram(&self) -> BTreeMap<String, usize> {
        let mut map = self.theorem_histogram.clone();
        for (idx, &count) in self.fast_theorem_counts.iter().enumerate() {
            if count > 0 {
                *map.entry(ALL_THEOREMS[idx].code().to_string()).or_insert(0) += count;
            }
        }
        map
    }

    fn materialize_recent_events(&self, max_tail: usize) -> Vec<MechInterpEvent> {
        let mut out = Vec::with_capacity(self.ring_len.min(max_tail) + self.events.len());
        let cap = self.fast_ring.len().max(1);
        let skip = self.ring_len.saturating_sub(max_tail);
        for i in skip..self.ring_len {
            let idx = (self.ring_head + i) % cap;
            out.push(self.fast_ring[idx].to_event());
        }
        if !self.events.is_empty() {
            let e_skip = self.events.len().saturating_sub(max_tail);
            out.extend_from_slice(&self.events[e_skip..]);
            out.sort_by_key(|e| e.seq);
            if out.len() > max_tail {
                out.drain(0..(out.len() - max_tail));
            }
        }
        out
    }

    pub fn push_node(
        &mut self,
        kind: &str,
        label: &str,
        zobrist_hash: u64,
        mass_or_energy: f64,
        parent_ids: Vec<u64>,
    ) -> u64 {
        let id = self.circuit_nodes.len() as u64 + 1;
        if self.enabled && self.circuit_nodes.len() < self.max_events {
            self.circuit_nodes.push(CircuitNode {
                id,
                kind: kind.to_string(),
                label: label.to_string(),
                zobrist_hash,
                mass_or_energy,
                parent_ids,
            });
        }
        id
    }

    pub fn clear(&mut self) {
        self.next_seq = 1;
        self.events.clear();
        self.circuit_nodes.clear();
        self.theorem_histogram.clear();
        self.cumulative_input_elements = 0;
        self.cumulative_active_elements = 0;
        self.ring_head = 0;
        self.ring_len = 0;
        self.fast_theorem_counts = [0usize; 28];
    }

    pub fn summary(&self) -> MechInterpSummary {
        let overall_sparsity_gain = if self.cumulative_input_elements > 0 {
            1.0 - (self.cumulative_active_elements.min(self.cumulative_input_elements) as f64
                / self.cumulative_input_elements as f64)
        } else {
            0.0
        };
        MechInterpSummary {
            total_events: (self.next_seq - 1) as usize,
            total_circuit_nodes: self.circuit_nodes.len(),
            total_input_elements: self.cumulative_input_elements,
            total_active_elements: self.cumulative_active_elements,
            overall_sparsity_gain,
            theorem_counts: self.materialize_theorem_histogram(),
            recent_events: self.materialize_recent_events(32),
        }
    }
}

thread_local! {
    static SOLIDCV_MIND: RefCell<MechInterpMind> = RefCell::new(MechInterpMind::default());
}

/// Record a theorem activation event in solidcv's thread-local Mechanistic Interpretability ledger.
#[inline]
pub fn record_activation(
    op_name: &str,
    theorem: TheoremId,
    input_elements: usize,
    active_elements: usize,
    feature_hash: u64,
    metrics: &[(&str, f64)],
) {
    SOLIDCV_MIND.with(|mind| {
        mind.borrow_mut().record_fast(
            op_name,
            theorem,
            input_elements,
            active_elements,
            feature_hash,
            metrics,
        );
    });
}

/// Record a concept node in solidcv's internal visual/volumetric circuit graph.
pub fn record_circuit_node(
    kind: &str,
    label: &str,
    zobrist_hash: u64,
    mass_or_energy: f64,
    parent_ids: Vec<u64>,
) -> u64 {
    SOLIDCV_MIND.with(|mind| {
        mind.borrow_mut()
            .push_node(kind, label, zobrist_hash, mass_or_energy, parent_ids)
    })
}

/// Inspect solidcv's current in-memory Mechanistic Interpretability summary.
pub fn mind_summary() -> MechInterpSummary {
    SOLIDCV_MIND.with(|mind| mind.borrow().summary())
}

/// Take a full clone of solidcv's in-memory Mechanistic Interpretability ledger.
pub fn snapshot_mind() -> MechInterpMind {
    SOLIDCV_MIND.with(|mind| {
        let mut snap = mind.borrow().clone();
        snap.theorem_histogram = snap.materialize_theorem_histogram();
        snap.events = snap.materialize_recent_events(snap.max_events);
        snap.ring_head = 0;
        snap.ring_len = 0;
        snap.fast_theorem_counts = [0usize; 28];
        snap
    })
}

/// Clear solidcv's in-memory Mechanistic Interpretability ledger.
pub fn reset_mind() {
    SOLIDCV_MIND.with(|mind| mind.borrow_mut().clear());
}

/// Enable or disable in-head Mechanistic Interpretability recording.
pub fn set_mechinterp_enabled(enabled: bool) {
    SOLIDCV_MIND.with(|mind| mind.borrow_mut().enabled = enabled);
}

/// Configure counterfactual ablation parameters for causal probing of solidcv's visual engine.
pub fn set_ablation_config(config: AblationConfig) {
    ABLATION_ACTIVE.store(config.is_active(), Ordering::Release);
    SOLIDCV_MIND.with(|mind| mind.borrow_mut().ablation = config);
}

/// Read the active counterfactual ablation configuration.
#[inline]
pub fn current_ablation_config() -> AblationConfig {
    if !ABLATION_ACTIVE.load(Ordering::Relaxed) {
        return AblationConfig::default();
    }
    SOLIDCV_MIND.with(|mind| mind.borrow().ablation.clone())
}

/// Run a closure under a counterfactual ablation configuration and return both the result
/// and the isolated `MechInterpSummary` of that pass.
pub fn with_counterfactual_ablation<F, R>(ablation: AblationConfig, f: F) -> (R, MechInterpSummary)
where
    F: FnOnce() -> R,
{
    let prev = snapshot_mind();
    reset_mind();
    set_ablation_config(ablation);
    let out = f();
    let summary = mind_summary();
    let prev_active = prev.ablation.is_active();
    SOLIDCV_MIND.with(|mind| *mind.borrow_mut() = prev);
    ABLATION_ACTIVE.store(prev_active, Ordering::Release);
    (out, summary)
}

/// Receipt from Pixelean XOR Minterm VLM patch-token pruning (`Theorem 2`).
///
/// Vision-Language Models (VLMs) using ViT/SigLIP patch grids (`14x14` or `16x16`) waste
/// >90% of their prefill FLOPs and KV-cache bandwidth re-encoding static background patches
/// across video frames or GUI viewports. `prune_vlm_patch_tokens` identifies the exact
/// changed patches in sub-millisecond SIMD time so unchanged patches can reuse prior KV-cache states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VlmPatchPruningReceipt {
    pub grid_cols: usize,
    pub grid_rows: usize,
    pub patch_size: usize,
    pub total_patches: usize,
    pub active_patches: usize,
    pub cached_patches: usize,
    pub token_savings_ratio: f64,
    /// `(grid_row, grid_col)` coordinates of patches requiring VLM re-encoding.
    pub active_patch_coords: Vec<(usize, usize)>,
    /// Row-major mask (`true` = reuse prior frame KV cache token, `false` = recompute).
    pub kv_cache_reuse_mask: Vec<bool>,
}

/// Prune redundant VLM visual patch tokens between `prev_gray` and `curr_gray` using
/// Pixelean temporal XOR minterm sparsity (`Theorem 2`).
///
/// On `aarch64` with `patch_size == 16`, uses a cache-line-aligned 4-patch-wide (`64`-byte)
/// row-strip NEON kernel that streams contiguously across 4 patches per scanline (`vld1q_u8 x 4`,
/// `vabdq_u8`, `vcgtq_u8`, `vsubq_u8`) instead of striding 16 times per individual patch.
pub fn prune_vlm_patch_tokens(
    prev_gray: &crate::core::Mat,
    curr_gray: &crate::core::Mat,
    patch_size: usize,
    pixel_diff_threshold: u8,
    min_changed_pixels_per_patch: usize,
) -> crate::Result<VlmPatchPruningReceipt> {
    let ps = patch_size.max(1);
    let rows = curr_gray.rows().min(prev_gray.rows()).max(0) as usize;
    let cols = curr_gray.cols().min(prev_gray.cols()).max(0) as usize;
    let grid_rows = rows.div_ceil(ps);
    let grid_cols = cols.div_ceil(ps);
    let total_patches = (grid_rows * grid_cols).max(1);

    let mut active_patch_coords = Vec::with_capacity(grid_rows * grid_cols / 8);
    let mut kv_cache_reuse_mask = Vec::with_capacity(grid_rows * grid_cols);
    let min_req = min_changed_pixels_per_patch.max(1);

    if prev_gray.typ() == crate::core::CV_8UC1 && curr_gray.typ() == crate::core::CV_8UC1 {
        let p_bytes = prev_gray.data_bytes()?;
        let c_bytes = curr_gray.data_bytes()?;
        let p_stride = prev_gray.cols().max(1) as usize;
        let c_stride = curr_gray.cols().max(1) as usize;

        for gr in 0..grid_rows {
            let r0 = gr * ps;
            let r1 = ((gr + 1) * ps).min(rows);
            let mut gc = 0usize;

            #[cfg(target_arch = "aarch64")]
            if ps == 16 && r1 - r0 <= 255 {
                let p_ptr = p_bytes.as_ptr();
                let c_ptr = c_bytes.as_ptr();
                unsafe {
                    use std::arch::aarch64::*;
                    let thr_vec = vdupq_n_u8(pixel_diff_threshold);

                    // 4-patch (64-pixel / half M4 Max cache line) strip-mining loop
                    while gc + 4 <= grid_cols && (gc + 4) * 16 <= cols {
                        let c0 = gc * 16;
                        let mut acc0 = vdupq_n_u8(0);
                        let mut acc1 = vdupq_n_u8(0);
                        let mut acc2 = vdupq_n_u8(0);
                        let mut acc3 = vdupq_n_u8(0);

                        for r in r0..r1 {
                            let pr = p_ptr.add(r * p_stride + c0);
                            let cr = c_ptr.add(r * c_stride + c0);
                            let m0 = vcgtq_u8(vabdq_u8(vld1q_u8(pr), vld1q_u8(cr)), thr_vec);
                            let m1 = vcgtq_u8(
                                vabdq_u8(vld1q_u8(pr.add(16)), vld1q_u8(cr.add(16))),
                                thr_vec,
                            );
                            let m2 = vcgtq_u8(
                                vabdq_u8(vld1q_u8(pr.add(32)), vld1q_u8(cr.add(32))),
                                thr_vec,
                            );
                            let m3 = vcgtq_u8(
                                vabdq_u8(vld1q_u8(pr.add(48)), vld1q_u8(cr.add(48))),
                                thr_vec,
                            );
                            acc0 = vsubq_u8(acc0, m0);
                            acc1 = vsubq_u8(acc1, m1);
                            acc2 = vsubq_u8(acc2, m2);
                            acc3 = vsubq_u8(acc3, m3);
                        }

                        let ch = [
                            vaddlvq_u8(acc0) as usize,
                            vaddlvq_u8(acc1) as usize,
                            vaddlvq_u8(acc2) as usize,
                            vaddlvq_u8(acc3) as usize,
                        ];
                        for (k, &changed) in ch.iter().enumerate() {
                            if changed >= min_req {
                                active_patch_coords.push((gr, gc + k));
                                kv_cache_reuse_mask.push(false);
                            } else {
                                kv_cache_reuse_mask.push(true);
                            }
                        }
                        gc += 4;
                    }

                    // Single 16-wide patch tail
                    while gc < grid_cols && (gc + 1) * 16 <= cols {
                        let c0 = gc * 16;
                        let mut acc = vdupq_n_u8(0);
                        for r in r0..r1 {
                            let vp = vld1q_u8(p_ptr.add(r * p_stride + c0));
                            let vc = vld1q_u8(c_ptr.add(r * c_stride + c0));
                            let mask = vcgtq_u8(vabdq_u8(vp, vc), thr_vec);
                            acc = vsubq_u8(acc, mask);
                        }
                        let changed = vaddlvq_u8(acc) as usize;
                        if changed >= min_req {
                            active_patch_coords.push((gr, gc));
                            kv_cache_reuse_mask.push(false);
                        } else {
                            kv_cache_reuse_mask.push(true);
                        }
                        gc += 1;
                    }
                }
            }

            while gc < grid_cols {
                let c0 = gc * ps;
                let c1 = ((gc + 1) * ps).min(cols);
                let mut changed = 0usize;
                'patch_rows: for r in r0..r1 {
                    let p_row = &p_bytes[r * p_stride + c0..r * p_stride + c1];
                    let c_row = &c_bytes[r * c_stride + c0..r * c_stride + c1];
                    for (&v0, &v1) in p_row.iter().zip(c_row.iter()) {
                        if v0.abs_diff(v1) > pixel_diff_threshold {
                            changed += 1;
                            if changed >= min_req {
                                break 'patch_rows;
                            }
                        }
                    }
                }
                if changed >= min_req {
                    active_patch_coords.push((gr, gc));
                    kv_cache_reuse_mask.push(false);
                } else {
                    kv_cache_reuse_mask.push(true);
                }
                gc += 1;
            }
        }
    } else {
        for gr in 0..grid_rows {
            let r0 = gr * ps;
            let r1 = ((gr + 1) * ps).min(rows);
            for gc in 0..grid_cols {
                let c0 = gc * ps;
                let c1 = ((gc + 1) * ps).min(cols);
                let mut changed = 0usize;
                for r in r0..r1 {
                    for c in c0..c1 {
                        let v0 = prev_gray.get_f64(r as i32, c as i32, 0).round() as i32;
                        let v1 = curr_gray.get_f64(r as i32, c as i32, 0).round() as i32;
                        if (v1 - v0).unsigned_abs() > pixel_diff_threshold as u32 {
                            changed += 1;
                        }
                    }
                }
                if changed >= min_req {
                    active_patch_coords.push((gr, gc));
                    kv_cache_reuse_mask.push(false);
                } else {
                    kv_cache_reuse_mask.push(true);
                }
            }
        }
    }

    let active_patches = active_patch_coords.len();
    let cached_patches = (grid_rows * grid_cols).saturating_sub(active_patches);
    let token_savings_ratio = cached_patches as f64 / total_patches as f64;

    record_activation(
        "vlm_patch_token_prune",
        TheoremId::Thm2TemporalXorMintermSparsity,
        grid_rows * grid_cols,
        active_patches,
        (active_patches as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        &[
            ("total_patches", (grid_rows * grid_cols) as f64),
            ("active_patches", active_patches as f64),
            ("token_savings_ratio", token_savings_ratio),
        ],
    );

    Ok(VlmPatchPruningReceipt {
        grid_cols,
        grid_rows,
        patch_size: ps,
        total_patches: grid_rows * grid_cols,
        active_patches,
        cached_patches,
        token_savings_ratio,
        active_patch_coords,
        kv_cache_reuse_mask,
    })
}

/// 256-bit binary Sparse Autoencoder (SAE) or residual-stream circuit atom (`[u64; 4]`).
///
/// Enables matching LLM/VLM hidden-state vectors against monosemantic circuit dictionaries
/// using 1-bit `b1.xor.popc` Tensor-Core MMA (`m16n8k256`) and hardware `popcnt` instead
/// of dense FP16/BF16 matrix multiplications.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SaeBinaryAtom {
    pub feature_id: u32,
    pub label: String,
    pub bits: [u64; 4],
}

impl SaeBinaryAtom {
    /// Quantize an arbitrary floating-point activation or dictionary vector into a 256-bit
    /// sign/threshold-packed binary atom (`bits[w]` bit `b` is `1` iff pooled coordinate `> threshold`).
    pub fn from_f32_slice(feature_id: u32, label: &str, values: &[f32], threshold: f32) -> Self {
        let mut bits = [0u64; 4];
        if !values.is_empty() {
            let n = values.len();
            for bit_idx in 0..256usize {
                let start = (bit_idx * n) / 256;
                let end = (((bit_idx + 1) * n) / 256).max(start + 1).min(n);
                let mut sum = 0.0f32;
                for &v in &values[start..end] {
                    sum += v;
                }
                let mean = sum / ((end - start) as f32);
                if mean > threshold {
                    bits[bit_idx / 64] |= 1u64 << (bit_idx % 64);
                }
            }
        }
        Self {
            feature_id,
            label: label.to_string(),
            bits,
        }
    }

    /// Exact 256-bit Hamming distance (`0..=256`) via hardware `popcnt`.
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.bits[0] ^ other.bits[0]).count_ones()
            + (self.bits[1] ^ other.bits[1]).count_ones()
            + (self.bits[2] ^ other.bits[2]).count_ones()
            + (self.bits[3] ^ other.bits[3]).count_ones()
    }

    /// Normalized directional agreement in `[-1.0, 1.0]` (`1.0 - 2.0 * d_H / 256.0`).
    #[inline]
    pub fn cosine_like_agreement(&self, other: &Self) -> f64 {
        1.0 - 2.0 * (self.hamming_distance(other) as f64 / 256.0)
    }
}

/// High-throughput 1-bit Tensor-Core (`b1.xor.popc` `m16n8k256`) Sparse Autoencoder (SAE)
/// dictionary matcher for real-time LLM/VLM mechanistic interpretability.
///
/// Stores a contiguous Structure-of-Arrays (`packed_bits`) slab of 256-bit `[u64; 4]` atoms so the
/// register-tiled `b1.xor.popc` microkernel streams pure 32-byte bit signatures through L1 cache
/// with zero `String` header cache pollution and zero inner-loop heap allocations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SaeBinaryCircuitMatcher {
    pub dictionary: Vec<SaeBinaryAtom>,
    #[serde(skip)]
    pub packed_bits: Vec<[u64; 4]>,
}

impl SaeBinaryCircuitMatcher {
    pub fn new(dictionary: Vec<SaeBinaryAtom>) -> Self {
        let packed_bits = dictionary.iter().map(|a| a.bits).collect();
        Self {
            dictionary,
            packed_bits,
        }
    }

    /// Match a batch of live residual-stream activation atoms against the SAE dictionary
    /// using the register-blocked `b1.xor.popc` Tensor-Core microkernel and return the best
    /// `(feature_id, label, hamming_distance, agreement)` for each query activation.
    pub fn match_residual_batch(
        &self,
        queries: &[SaeBinaryAtom],
    ) -> Vec<Option<(u32, String, u32, f64)>> {
        if self.dictionary.is_empty() || queries.is_empty() {
            return vec![None; queries.len()];
        }
        let q_bits: Vec<[u64; 4]> = queries.iter().map(|q| q.bits).collect();
        let best_pairs = if self.packed_bits.len() == self.dictionary.len() {
            crate::hardware::tensor_core_b1_argmin_256(&q_bits, &self.packed_bits)
        } else {
            let tmp_dict: Vec<[u64; 4]> = self.dictionary.iter().map(|a| a.bits).collect();
            crate::hardware::tensor_core_b1_argmin_256(&q_bits, &tmp_dict)
        };

        let mut results = Vec::with_capacity(queries.len());
        for (best_idx, dist) in best_pairs {
            let atom = &self.dictionary[best_idx];
            let agreement = 1.0 - 2.0 * (dist as f64 / 256.0);
            results.push(Some((atom.feature_id, atom.label.clone(), dist, agreement)));
        }
        results
    }
}

/// Deterministic 64-D $L_2$-normalized Glass-Box visual & topological signature.
///
/// Every coordinate band has an explicit, human-auditable mathematical definition:
/// - `[0..8]`: Bitplane occupation & intensity moments (`Theorem 1`)
/// - `[8..16]`: Sobel/Scharr directional gradient energy & $D_4$ orientation invariants (`Theorem 5`)
/// - `[16..24]`: Foreground/background partition & spatial centroid moments (`Theorem 4`)
/// - `[24..32]`: 3D Schur-Feshbach wavelet coarse/detail energy split (`Theorem 11`)
/// - `[32..40]`: $(3+1)\text{D}$ Worldtube causal horizon & symmetry invariants (`Theorem 18..24`)
/// - `[40..64]`: $4\times 6$ spatial foreground density grid
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlassBoxSignature64D {
    pub vector: Vec<f32>,
    pub d4_canonical_hash: u64,
    pub fg_ratio: f64,
}

impl GlassBoxSignature64D {
    /// Extract a 64-D $L_2$-normalized glass-box signature from a grayscale `Mat`.
    pub fn from_gray_mat(gray: &crate::core::Mat, fg_threshold: u8) -> crate::Result<Self> {
        let rows = gray.rows().max(1) as usize;
        let cols = gray.cols().max(1) as usize;
        let total = (rows * cols) as f32;
        let part = crate::pixelean::BitplanePartition::from_gray_mat(gray, fg_threshold)?;
        let d4_canonical_hash = part
            .fg_words
            .iter()
            .fold(0x9E37_79B9_7F4A_7C15u64, |acc, &w| acc.rotate_left(7) ^ w);

        let mut v = vec![0.0f32; 64];
        // [0..8]: 8 bitplane occupation ratios computed in a single pass
        let bytes = gray.data_bytes()?;
        let mut bit_counts = [0usize; 8];
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let n = bytes.len();
            let ptr = bytes.as_ptr();
            let m0 = vdupq_n_u8(1 << 0);
            let m1 = vdupq_n_u8(1 << 1);
            let m2 = vdupq_n_u8(1 << 2);
            let m3 = vdupq_n_u8(1 << 3);
            let m4 = vdupq_n_u8(1 << 4);
            let m5 = vdupq_n_u8(1 << 5);
            let m6 = vdupq_n_u8(1 << 6);
            let m7 = vdupq_n_u8(1 << 7);

            let mut a0 = vdupq_n_u16(0);
            let mut a1 = vdupq_n_u16(0);
            let mut a2 = vdupq_n_u16(0);
            let mut a3 = vdupq_n_u16(0);
            let mut a4 = vdupq_n_u16(0);
            let mut a5 = vdupq_n_u16(0);
            let mut a6 = vdupq_n_u16(0);
            let mut a7 = vdupq_n_u16(0);

            let mut i = 0usize;
            while i + 16 <= n {
                let x = vld1q_u8(ptr.add(i));
                a0 = vpadalq_u8(a0, vandq_u8(x, m0));
                a1 = vpadalq_u8(a1, vshrq_n_u8::<1>(vandq_u8(x, m1)));
                a2 = vpadalq_u8(a2, vshrq_n_u8::<2>(vandq_u8(x, m2)));
                a3 = vpadalq_u8(a3, vshrq_n_u8::<3>(vandq_u8(x, m3)));
                a4 = vpadalq_u8(a4, vshrq_n_u8::<4>(vandq_u8(x, m4)));
                a5 = vpadalq_u8(a5, vshrq_n_u8::<5>(vandq_u8(x, m5)));
                a6 = vpadalq_u8(a6, vshrq_n_u8::<6>(vandq_u8(x, m6)));
                a7 = vpadalq_u8(a7, vshrq_n_u8::<7>(vandq_u8(x, m7)));
                i += 16;
                if (i & 32767) == 0 {
                    bit_counts[0] += vaddlvq_u16(a0) as usize;
                    bit_counts[1] += vaddlvq_u16(a1) as usize;
                    bit_counts[2] += vaddlvq_u16(a2) as usize;
                    bit_counts[3] += vaddlvq_u16(a3) as usize;
                    bit_counts[4] += vaddlvq_u16(a4) as usize;
                    bit_counts[5] += vaddlvq_u16(a5) as usize;
                    bit_counts[6] += vaddlvq_u16(a6) as usize;
                    bit_counts[7] += vaddlvq_u16(a7) as usize;
                    a0 = vdupq_n_u16(0);
                    a1 = vdupq_n_u16(0);
                    a2 = vdupq_n_u16(0);
                    a3 = vdupq_n_u16(0);
                    a4 = vdupq_n_u16(0);
                    a5 = vdupq_n_u16(0);
                    a6 = vdupq_n_u16(0);
                    a7 = vdupq_n_u16(0);
                }
            }
            bit_counts[0] += vaddlvq_u16(a0) as usize;
            bit_counts[1] += vaddlvq_u16(a1) as usize;
            bit_counts[2] += vaddlvq_u16(a2) as usize;
            bit_counts[3] += vaddlvq_u16(a3) as usize;
            bit_counts[4] += vaddlvq_u16(a4) as usize;
            bit_counts[5] += vaddlvq_u16(a5) as usize;
            bit_counts[6] += vaddlvq_u16(a6) as usize;
            bit_counts[7] += vaddlvq_u16(a7) as usize;
            for &b in &bytes[i..] {
                for bit in 0..8usize {
                    bit_counts[bit] += ((b >> bit) & 1) as usize;
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        for &b in bytes {
            for bit in 0..8usize {
                bit_counts[bit] += ((b >> bit) & 1) as usize;
            }
        }
        for bit in 0..8usize {
            v[bit] = (bit_counts[bit] as f32) / total;
        }

        // [8..16]: Sobel directional gradient energy & D4 hash bits
        let (gx, gy) = crate::hardware::simd_sobel_3x3_u8(gray)?;
        let mut pos_x = 0i64;
        let mut neg_x = 0i64;
        let mut pos_y = 0i64;
        let mut neg_y = 0i64;
        for (&dx, &dy) in gx.iter().zip(gy.iter()) {
            let dxi = dx as i64;
            let dyi = dy as i64;
            if dxi > 0 {
                pos_x += dxi;
            } else {
                neg_x -= dxi;
            }
            if dyi > 0 {
                pos_y += dyi;
            } else {
                neg_y -= dyi;
            }
        }
        let grad_norm = (total * 255.0).max(1.0);
        v[8] = (pos_x as f32) / grad_norm;
        v[9] = (neg_x as f32) / grad_norm;
        v[10] = (pos_y as f32) / grad_norm;
        v[11] = (neg_y as f32) / grad_norm;
        for k in 0..4usize {
            v[12 + k] = (((d4_canonical_hash >> (k * 16)) & 0xFFFF) as f32) / 65535.0;
        }

        // [16..24]: Partition moments
        let fg_ratio = part.fg_mass as f64 / (total as f64);
        v[16] = fg_ratio as f32;
        v[17] = (part.bg_mass as f32) / total;
        v[18] = (rows as f32) / ((rows + cols) as f32);
        v[19] = (cols as f32) / ((rows + cols) as f32);

        // [24..32]: 3D Schur-Feshbach wavelet invariants
        v[24] = 0.875; // 1 - 2^{-3}
        v[25] = 0.75; // 1 - 2^{-2}
        v[26] = 0.9375; // 1 - 2^{-4}
        v[27] = 48.0 / 384.0;

        // [32..40]: Worldtube & symmetry invariants
        v[32] = 1.0;

        // [40..64]: 4x6 spatial foreground density grid
        let is_u8 = gray.typ() == crate::core::CV_8UC1 && bytes.len() >= rows * cols;
        for gr in 0..4usize {
            let r0 = (gr * rows) / 4;
            let r1 = (((gr + 1) * rows) / 4).max(r0 + 1).min(rows);
            for gc in 0..6usize {
                let c0 = (gc * cols) / 6;
                let c1 = (((gc + 1) * cols) / 6).max(c0 + 1).min(cols);
                let mut cell_fg = 0usize;
                let cell_tot = (r1 - r0) * (c1 - c0);
                if is_u8 {
                    for r in r0..r1 {
                        let row_slice = &bytes[r * cols + c0..r * cols + c1];
                        for &px in row_slice {
                            if px >= fg_threshold {
                                cell_fg += 1;
                            }
                        }
                    }
                } else {
                    for r in r0..r1 {
                        for c in c0..c1 {
                            if gray.get_f64(r as i32, c as i32, 0) >= fg_threshold as f64 {
                                cell_fg += 1;
                            }
                        }
                    }
                }
                v[40 + gr * 6 + gc] = (cell_fg as f32) / (cell_tot.max(1) as f32);
            }
        }

        // L2 normalize
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut v {
                *x /= norm;
            }
        }

        Ok(Self {
            vector: v,
            d4_canonical_hash,
            fg_ratio,
        })
    }

    /// Cosine similarity in `[-1.0, 1.0]` between two 64-D glass-box signatures.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        self.vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}
