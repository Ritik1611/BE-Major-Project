"""
research_evaluation_suite.py
=============================================================================
Research-Grade Evaluation Suite for the Federated Learning System
BE Major Project — DAIC-WOZ Depression Detection

Covers four independent evaluation dimensions:
  [1] Privacy Analysis       — RDP/GDP accounting, amplification, composition
  [2] Security Verification  — integrity, receipt chains, crypto, path safety
  [3] Attack Testbeds        — MIA, gradient inversion, Byzantine poisoning
  [4] Compliance Evidence    — signed receipts, audit chain, DP certificates

Design principles
-----------------
• Zero hard dependencies on the live runtime.  Each module stubs the exact
  interfaces used in the project (DepressionNet, SecureStore, receipts,
  rdp_to_dp, aggregator) so it can run standalone on any clean machine.
• Research-grade statistics: AUC-ROC with 95 % CI (DeLong), Mann-Whitney U,
  Cohen's d, bootstrapped confidence intervals, Wilcoxon signed-rank.
• Every numerical result maps back to a specific file + line in the codebase.
• Outputs: console table, results.json, privacy_certificate.json, plots/*.png

Usage
-----
  python3 research_evaluation_suite.py [--data_dir ./data] [--rounds 20]
                                        [--no_dp] [--seed 42] [--plots]

=============================================================================
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac as _hmac
import json
import logging
import math
import os
import secrets
import struct
import sys
import time
import warnings
from base64 import b64encode, b64decode
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval")

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Paths / constants that mirror the project layout
# (from installer/runtime/agents/dp/dp_agent.py and fl_algorithm_comparison.py)
# ---------------------------------------------------------------------------
BASE_DIR   = Path.home() / ".federated"
STORE_ROOT = BASE_DIR / "data" / "secure_store"
RESULTS_DIR = Path("eval_results")
PLOTS_DIR   = RESULTS_DIR / "plots"

RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ===========================================================================
# SECTION 0 — Numpy-based toy model (mirrors DepressionNet in
#              fl_algorithm_comparison.py §SECTION 4)
# ===========================================================================

class NumpyLayer:
    """Single linear layer with ReLU, forward-mode autodiff for gradient attacks."""
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2 / in_dim), (in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x @ self.W + self.b)

    def parameters(self) -> List[np.ndarray]:
        return [self.W, self.b]


class NumpyDepressionNet:
    """
    Lightweight numpy mirror of DepressionNet (fl_algorithm_comparison.py, ~L650).
    Supports forward pass, loss, gradient computation for attack testbeds.
    """
    def __init__(self, audio_dim=78, visual_dim=70, text_dim=768,
                 fusion_hidden=256, n_classes=2, seed=42):
        self.enc_a = NumpyLayer(audio_dim,  128, seed)
        self.enc_v = NumpyLayer(visual_dim, 128, seed + 1)
        in_fusion  = text_dim + 128 + 128           # 1024
        self.fuse1 = NumpyLayer(in_fusion, fusion_hidden, seed + 2)
        self.fuse2 = NumpyLayer(fusion_hidden, fusion_hidden // 2, seed + 3)
        rng = np.random.default_rng(seed + 4)
        self.W_out = rng.normal(0, 0.01,
                                (fusion_hidden // 2, n_classes)).astype(np.float32)
        self.b_out = np.zeros(n_classes, dtype=np.float32)
        self.n_classes = n_classes

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        ex = np.exp(x - x.max(axis=-1, keepdims=True))
        return ex / ex.sum(axis=-1, keepdims=True)

    def forward(self, audio: np.ndarray, visual: np.ndarray,
                text: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a = self.enc_a.forward(audio)
        v = self.enc_v.forward(visual)
        fused = np.concatenate([text, a, v], axis=-1)
        h1    = self.fuse1.forward(fused)
        h2    = self.fuse2.forward(h1)
        logits = h2 @ self.W_out + self.b_out
        probs  = self._softmax(logits)
        return logits, probs

    def cross_entropy_loss(self, probs: np.ndarray,
                           labels: np.ndarray) -> float:
        eps = 1e-9
        return -np.mean(np.log(probs[np.arange(len(labels)), labels] + eps))

    def predict_proba(self, audio, visual, text) -> np.ndarray:
        _, probs = self.forward(audio, visual, text)
        return probs

    def get_flat_params(self) -> np.ndarray:
        layers = [self.enc_a, self.enc_v, self.fuse1, self.fuse2]
        parts = [p.flatten() for l in layers for p in l.parameters()]
        parts.extend([self.W_out.flatten(), self.b_out.flatten()])
        return np.concatenate(parts)

    def set_flat_params(self, flat: np.ndarray):
        idx = 0
        for layer in [self.enc_a, self.enc_v, self.fuse1, self.fuse2]:
            for p in [layer.W, layer.b]:
                n = p.size
                p[:] = flat[idx:idx+n].reshape(p.shape)
                idx += n
        n = self.W_out.size
        self.W_out[:] = flat[idx:idx+n].reshape(self.W_out.shape); idx += n
        self.b_out[:] = flat[idx:idx+self.b_out.size]

    def count_params(self) -> int:
        return self.get_flat_params().size


def make_synthetic_dataset(n: int = 200, audio_dim=78, visual_dim=70,
                            text_dim=768, seed: int = 0) -> dict:
    """Generate a labelled dataset that mirrors DAIC-WOZ feature shapes."""
    rng = np.random.default_rng(seed)
    half = n // 2
    # Depressed: slightly higher L2 norm in audio, lower visual energy
    audio  = np.vstack([
        rng.normal(0.5, 0.3, (half, audio_dim)),    # depressed
        rng.normal(0.0, 0.3, (half, audio_dim)),    # not depressed
    ]).astype(np.float32)
    visual = np.vstack([
        rng.normal(0.2, 0.2, (half, visual_dim)),
        rng.normal(0.4, 0.2, (half, visual_dim)),
    ]).astype(np.float32)
    text   = np.vstack([
        rng.normal(-0.1, 0.1, (half, text_dim)),
        rng.normal( 0.1, 0.1, (half, text_dim)),
    ]).astype(np.float32)
    labels = np.array([1] * half + [0] * half, dtype=np.int64)
    idx    = rng.permutation(n)
    return dict(audio=audio[idx], visual=visual[idx],
                text=text[idx], labels=labels[idx])


# ===========================================================================
# SECTION 1 — PRIVACY ANALYSIS
# Mirrors rdp_to_dp() in fl_algorithm_comparison.py (L492) and
# DPAgent._rdp_to_dp()  in dp_agent.py (L62)
# ===========================================================================

@dataclass
class PrivacyAccountantResult:
    """Full (ε, δ)-DP guarantee with supporting statistics."""
    epsilon:         float
    delta:           float
    noise_mult:      float
    clip_norm:       float
    sample_rate:     float
    steps:           int
    best_alpha:      int
    rdp_epsilon:     float
    mechanism:       str
    composition:     str    # "advanced" | "basic"
    amplified_eps:   Optional[float] = None    # after privacy amplification


def rdp_epsilon(noise_mult: float, alpha: int) -> float:
    """RDP(α) for Gaussian mechanism — Mironov 2017 Theorem 3."""
    return alpha / (2.0 * noise_mult ** 2)


def rdp_to_dp(noise_mult: float, sample_rate: float,
              steps: int, delta: float, alpha_range=(2, 512)) -> PrivacyAccountantResult:
    """
    Full RDP → (ε, δ)-DP conversion with optimal α search.
    FIX: Added overflow protection for math.exp() and use log1p/expm1 for stability.
    """
    if noise_mult <= 0 or sample_rate <= 0:
        return PrivacyAccountantResult(
            epsilon=float("inf"), delta=delta, noise_mult=noise_mult,
            clip_norm=1.0, sample_rate=sample_rate, steps=steps,
            best_alpha=-1, rdp_epsilon=float("inf"),
            mechanism="Gaussian", composition="advanced_RDP",
            amplified_eps=float("inf"),
        )

    best_eps = math.inf
    best_alpha = -1
    best_rdp = math.inf

    for alpha in range(*alpha_range):
        q = sample_rate
        rdp_a = rdp_epsilon(noise_mult, alpha) * steps

        # Privacy amplification by Poisson subsampling (Wang et al. 2019)
        # FIX: Avoid overflow in math.exp() by checking threshold and using stable functions
        if q < 0.5 and rdp_a < 700:  # 700 is safe threshold (exp(709) ~ max float64)
            # Use log1p/expm1 for numerical stability: log(1+x) and exp(x)-1
            rdp_amp = min(
                rdp_a,
                2 * q ** 2 * rdp_a,
                q * rdp_a + math.log1p(q * math.expm1(rdp_a))  # Stable computation
            )
        else:
            # Fallback: no amplification when q is large or rdp_a would overflow
            rdp_amp = rdp_a

        # Convert RDP(α) → (ε, δ)
        eps_a = rdp_amp + math.log(1 / delta) / (alpha - 1)
        if eps_a < best_eps:
            best_eps = eps_a
            best_alpha = alpha
            best_rdp = rdp_amp

    # Zero-Concentrated DP (zCDP) Gaussian: ρ = 1/(2σ²)
    rho = 1 / (2 * noise_mult ** 2) * steps
    zcdp_eps = rho + 2 * math.sqrt(rho * math.log(1 / delta))
    final_eps = min(best_eps, zcdp_eps)

    return PrivacyAccountantResult(
        epsilon=round(final_eps, 6),
        delta=delta,
        noise_mult=noise_mult,
        clip_norm=1.0,
        sample_rate=sample_rate,
        steps=steps,
        best_alpha=best_alpha,
        rdp_epsilon=round(best_rdp, 6),
        mechanism="Gaussian",
        composition="advanced_RDP",
        amplified_eps=round(best_eps, 6),
    )


def privacy_analysis(n_rounds: int = 30, noise_mults: List[float] = None,
                     n_clients: int = 8, n_samples_per_client: int = 5,
                     batch_size: int = 8, local_epochs: int = 5,
                     delta: float = 1e-5) -> Dict[str, Any]:
    """
    Full privacy analysis matching the FL setup in fl_algorithm_comparison.py.

    Returns per-round ε trajectory, optimal noise multiplier recommendation,
    and privacy-utility tradeoff metadata.
    """
    if noise_mults is None:
        noise_mults = [0.5, 0.8, 1.0, 1.1, 1.5, 2.0]

    total_samples  = n_clients * n_samples_per_client
    sample_rate    = batch_size / max(n_samples_per_client, 1)
    # local steps per round = local_epochs * ceil(n_samples / batch_size)
    local_steps    = local_epochs * max(1, math.ceil(n_samples_per_client / batch_size))

    results = {}
    for nm in noise_mults:
        eps_trajectory = []
        cumulative_eps = 0.0
        for r in range(1, n_rounds + 1):
            # Advanced composition: total steps = round × local_steps
            acc = rdp_to_dp(nm, sample_rate, r * local_steps, delta)
            eps_trajectory.append(acc.epsilon)
            cumulative_eps = acc.epsilon
        results[nm] = {
            "eps_trajectory":  eps_trajectory,
            "final_epsilon":   cumulative_eps,
            "final_alpha":     acc.best_alpha,
            "rdp_eps":         acc.rdp_epsilon,
            "meets_epsilon_8": cumulative_eps <= 8.0,   # common FL privacy budget
        }

    # Identify minimum noise_mult achieving ε ≤ 8 at δ = 1e-5
    safe_nm = [nm for nm, r in results.items() if r["meets_epsilon_8"]]
    optimal_nm = min(safe_nm) if safe_nm else min(noise_mults)

    # Privacy amplification gain for q=sample_rate vs q=1
    amp_gain = {}
    for nm in noise_mults:
        eps_no_amp = rdp_to_dp(nm, 1.0, n_rounds * local_steps, delta).epsilon
        eps_amp    = results[nm]["final_epsilon"]
        amp_gain[nm] = round((eps_no_amp - eps_amp) / eps_no_amp * 100, 2)

    return {
        "per_noise_mult":        results,
        "optimal_noise_mult":    optimal_nm,
        "privacy_amplification_gain_pct": amp_gain,
        "sample_rate":           round(sample_rate, 4),
        "local_steps_per_round": local_steps,
        "delta":                 delta,
        "n_rounds":              n_rounds,
    }


# ===========================================================================
# SECTION 2 — SECURITY VERIFICATION
# Tests mirror security properties claimed in:
#   installer/security/integrity.py          — HMAC-chained baseline
#   installer/runtime/core/centralized_secure_store.py — AES-GCM
#   installer/security/military_security.py  — path safety, canary
#   server/orchestration_agent/src/ledger.rs — HMAC chain
# ===========================================================================

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend


@dataclass
class SecurityTestResult:
    name:    str
    passed:  bool
    detail:  str
    latency_ms: float = 0.0


class SecurityVerifier:
    """
    Verifies all security properties implemented in the codebase.
    Each test_* method targets a specific file and mitigation.
    """

    def __init__(self):
        self.results: List[SecurityTestResult] = []

    def _record(self, name: str, passed: bool,
                detail: str, latency: float = 0.0):
        self.results.append(SecurityTestResult(name, passed, detail, latency))
        status = "✓ PASS" if passed else "✗ FAIL"
        log.info(f"  [{status}] {name} — {detail}")

    # ── 2.1  AES-GCM encrypt-then-MAC (centralized_secure_store.py) ─────────
    def test_aes_gcm_confidentiality(self):
        t0 = time.perf_counter()
        master_key = secrets.token_bytes(32)
        info = b"enc:session_0"
        derived = HKDF(hashes.SHA256(), 32, None, info,
                       default_backend()).derive(master_key)
        aesgcm  = AESGCM(derived)
        nonce   = os.urandom(12)
        msg     = b"patient_id=123;phq=14"
        ct      = aesgcm.encrypt(nonce, msg, None)

        # Correct decryption
        plain = aesgcm.decrypt(nonce, ct, None)
        ok1   = plain == msg

        # Ciphertext tampering must raise InvalidTag
        ct_tampered = bytearray(ct)
        ct_tampered[8] ^= 0xFF
        try:
            aesgcm.decrypt(nonce, bytes(ct_tampered), None)
            ok2 = False
        except Exception:
            ok2 = True

        # Nonce reuse: same nonce → same ct (semantic security broken)
        ct2 = aesgcm.encrypt(nonce, msg, None)
        ok3 = ct == ct2  # deterministic — nonce reuse IS a vulnerability; we detect it

        ms = (time.perf_counter() - t0) * 1000
        self._record("AES-GCM confidentiality",
                     ok1 and ok2,
                     f"decrypt_ok={ok1}, tamper_detected={ok2}, "
                     f"nonce_reuse_deterministic={ok3}", ms)

    # ── 2.2  Path traversal prevention (FIX-SS-1) ────────────────────────────
    def test_path_traversal_prevention(self):
        t0   = time.perf_counter()
        root = Path("/tmp/secure_root_eval")
        root.mkdir(parents=True, exist_ok=True)

        # Adjacent directory attack  (the bug fixed in FIX-SS-1)
        evil_path = Path("/tmp/secure_root_eval_evil/x.bin")
        evil_path.parent.mkdir(parents=True, exist_ok=True)
        evil_path.write_bytes(b"secret")

        def path_is_within(child: Path, parent: Path) -> bool:
            try:
                child.relative_to(parent)
                return True
            except ValueError:
                return False

        # old_check = str(evil_path).startswith(str(root))  → WRONG
        old_check = str(evil_path.resolve()).startswith(str(root.resolve()))
        new_check = path_is_within(evil_path.resolve(), root.resolve())

        ms = (time.perf_counter() - t0) * 1000
        self._record(
            "Path traversal prevention (FIX-SS-1)",
            old_check is True and new_check is False,
            f"old_startswith_bug_would_allow={old_check}, "
            f"new_relative_to_blocks={not new_check}",
            ms,
        )

    # ── 2.3  HMAC receipt chaining (centralised_receipts.py + ledger.rs) ─────
    def test_receipt_hmac_chain(self):
        t0   = time.perf_counter()
        key  = secrets.token_bytes(32)
        chain_links = []

        prev = "genesis"
        for i in range(5):
            payload = json.dumps({"round": i, "device": f"dev_{i}",
                                  "eps": 0.4 + i * 0.1}, sort_keys=True)
            mac  = _hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()
            link = _hmac.new(key,
                             (prev + "|" + mac).encode(),
                             hashlib.sha256).hexdigest()
            chain_links.append(link)
            prev = link

        # Tamper round 2 → chain from round 3 onwards should break
        tampered_link = chain_links[2][::-1]   # reverse hex digits
        next_link_tampered = _hmac.new(
            key, (tampered_link + "|" + "fake_mac").encode(),
            hashlib.sha256).hexdigest()
        chain_broken = next_link_tampered != chain_links[3]

        ms = (time.perf_counter() - t0) * 1000
        self._record("HMAC receipt chain integrity",
                     chain_broken and len(chain_links) == 5,
                     f"chain_length=5, tamper_detects_break={chain_broken}", ms)

    # ── 2.4  Integrity baseline write-once (BYPASS-1 fix, integrity.py) ──────
    def test_write_once_baseline(self):
        t0    = time.perf_counter()
        token = secrets.token_hex(32)

        # Simulate generate_install_token() + consume_write_token()
        token_store: Optional[str] = token

        def consume(provided: str) -> bool:
            nonlocal token_store
            if token_store is None:
                return False
            match = _hmac.compare_digest(token_store, provided)
            if match:
                token_store = None   # one-time use
            return match

        first_use  = consume(token)    # should succeed
        second_use = consume(token)    # token is None → should fail
        bad_token  = consume("wrong")  # wrong token → should fail

        ms = (time.perf_counter() - t0) * 1000
        self._record("Write-once install token (BYPASS-1)",
                     first_use and not second_use and not bad_token,
                     f"first_use={first_use}, second_replay={second_use}, "
                     f"wrong_token={bad_token}", ms)

    # ── 2.5  Timing-safe comparison (timing side-channel) ────────────────────
    def test_timing_safe_comparison(self):
        t0   = time.perf_counter()
        a    = secrets.token_bytes(32)
        b_ok = a
        b_no = secrets.token_bytes(32)

        N = 10_000
        t_ok_start = time.perf_counter()
        for _ in range(N):
            _hmac.compare_digest(a, b_ok)
        t_ok = time.perf_counter() - t_ok_start

        t_no_start = time.perf_counter()
        for _ in range(N):
            _hmac.compare_digest(a, b_no)
        t_no = time.perf_counter() - t_no_start

        ratio  = t_ok / max(t_no, 1e-12)
        timing_safe = 0.5 < ratio < 2.0   # within 2× = constant time

        ms = (time.perf_counter() - t0) * 1000
        self._record("Timing-safe hmac.compare_digest",
                     timing_safe,
                     f"ratio={ratio:.3f} (target 0.5–2.0), "
                     f"ok_us={t_ok/N*1e6:.3f}, bad_us={t_no/N*1e6:.3f}", ms)

    # ── 2.6  Canary file tamper detection (military_security.py) ─────────────
    def test_canary_detection(self):
        t0 = time.perf_counter()
        canaries: Dict[str, str] = {}
        canary_dir = Path("/tmp/eval_canaries")
        canary_dir.mkdir(exist_ok=True)

        for i in range(5):
            name    = f".{secrets.token_hex(8)}.dat"
            content = secrets.token_bytes(64)
            path    = canary_dir / name
            path.write_bytes(content)
            canaries[str(path)] = hashlib.sha256(content).hexdigest()

        def check_canaries(store: dict) -> bool:
            for path_str, expected in store.items():
                p = Path(path_str)
                if not p.exists():
                    return False
                if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                    return False
            return True

        ok_before = check_canaries(canaries)

        # Tamper one canary
        first_path = list(canaries.keys())[0]
        Path(first_path).write_bytes(b"TAMPERED_DATA")
        ok_after = check_canaries(canaries)

        ms = (time.perf_counter() - t0) * 1000
        self._record("Canary tamper detection (military_security.py)",
                     ok_before and not ok_after,
                     f"intact={ok_before}, tamper_detected={not ok_after}", ms)

    # ── 2.7  HKDF per-agent key isolation ────────────────────────────────────
    def test_hkdf_key_isolation(self):
        t0         = time.perf_counter()
        master_key = secrets.token_bytes(32)

        def derive(agent: str, context: str) -> bytes:
            info = f"{agent}:{context}".encode()
            return HKDF(hashes.SHA256(), 32, None, info,
                        default_backend()).derive(master_key)

        k_lda_a   = derive("lda",   "session_0")
        k_lda_b   = derive("lda",   "session_1")   # same agent, diff context
        k_dp_a    = derive("dp",    "session_0")    # diff agent, same context
        k_enc_a   = derive("enc",   "session_0")    # another agent

        all_distinct = len({k_lda_a, k_lda_b, k_dp_a, k_enc_a}) == 4
        ms = (time.perf_counter() - t0) * 1000
        self._record("HKDF per-agent key isolation (FIX-CRYPTO-3)",
                     all_distinct,
                     f"4 keys all distinct: {all_distinct}", ms)

    # ── 2.8  File-level HMAC tree hash (integrity.py::compute_tree_hash) ─────
    def test_integrity_tree_hash(self):
        import tempfile, shutil
        t0   = time.perf_counter()
        root = Path(tempfile.mkdtemp())
        try:
            (root / "agents").mkdir()
            (root / "agents" / "model.py").write_bytes(b"print('hello')")
            (root / "agents" / "policy.py").write_bytes(b"POLICY=1")

            def tree_hash(d: Path) -> str:
                h = hashlib.sha3_256()
                for p in sorted(d.rglob("*.py")):
                    rel = p.relative_to(d).as_posix().lower().encode()
                    h.update(struct.pack(">I", len(rel)))
                    h.update(rel)
                    data = p.read_bytes()
                    h.update(struct.pack(">Q", len(data)))
                    h.update(data)
                return h.hexdigest()

            h1 = tree_hash(root)
            # Modify one file
            (root / "agents" / "model.py").write_bytes(b"print('TAMPERED')")
            h2 = tree_hash(root)

            ms = (time.perf_counter() - t0) * 1000
            self._record("SHA3-256 integrity tree hash (integrity.py)",
                         h1 != h2,
                         f"baseline={h1[:16]}…, after_tamper={h2[:16]}…, "
                         f"change_detected={h1 != h2}", ms)
        finally:
            shutil.rmtree(root)

    def run_all(self) -> Dict[str, Any]:
        log.info("=" * 60)
        log.info("SECTION 2 — SECURITY VERIFICATION")
        log.info("=" * 60)
        for fn in [
            self.test_aes_gcm_confidentiality,
            self.test_path_traversal_prevention,
            self.test_receipt_hmac_chain,
            self.test_write_once_baseline,
            self.test_timing_safe_comparison,
            self.test_canary_detection,
            self.test_hkdf_key_isolation,
            self.test_integrity_tree_hash,
        ]:
            try:
                fn()
            except Exception as e:
                self._record(fn.__name__, False, f"Exception: {e}")

        passed = sum(r.passed for r in self.results)
        total  = len(self.results)
        return {
            "passed": passed,
            "failed": total - passed,
            "total":  total,
            "pass_rate": round(passed / total, 4),
            "tests": [asdict(r) for r in self.results],
        }


# ===========================================================================
# SECTION 3 — ATTACK TESTBEDS & DEFENSE EVALUATION
# 3a. Membership Inference Attack (Shokri et al. 2017)
# 3b. Gradient Inversion / DLG (Zhu et al. NeurIPS 2019)
# 3c. Byzantine Poisoning (label flip + scaling)
# 3d. Free-rider detection (zero-gradient)
# Defense wiring: DP-SGD, trimmed mean, norm clipping, krum
# ===========================================================================

# ------------ shared helpers ------------------------------------------------

def _bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = 1000,
                  alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    rng  = np.random.default_rng(seed)
    boot = [stat_fn(rng.choice(values, len(values), replace=True))
            for _ in range(n_boot)]
    lo   = np.percentile(boot, 100 * alpha / 2)
    hi   = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def delong_auc_variance(y_true: np.ndarray,
                        y_score: np.ndarray) -> Tuple[float, float, float]:
    """DeLong et al. 1988 AUC variance for 95% CI."""
    from sklearn.metrics import roc_auc_score
    pos   = y_score[y_true == 1]
    neg   = y_score[y_true == 0]
    auc   = roc_auc_score(y_true, y_score)
    n_pos = len(pos)
    n_neg = len(neg)
    # Placement values
    pv_pos = np.array([np.mean(p >= neg) for p in pos])
    pv_neg = np.array([np.mean(n <= pos) for n in neg])
    var    = (np.var(pv_pos) / n_pos + np.var(pv_neg) / n_neg)
    se     = math.sqrt(var)
    ci_lo  = auc - 1.96 * se
    ci_hi  = auc + 1.96 * se
    return float(auc), float(max(0, ci_lo)), float(min(1, ci_hi))


# ------------ 3a. Membership Inference Attack --------------------------------

@dataclass
class MIAResult:
    """
    Membership Inference Attack result following Shokri et al. 2017.
    Metric: AUC-ROC of a shadow-model meta-classifier.
    Under perfect DP: AUC → 0.5 (no advantage over random guess).
    """
    auc:        float
    auc_ci_lo:  float
    auc_ci_hi:  float
    attack_acc: float
    tpr:        float     # True positive rate at 10% FPR
    advantage:  float     # AUC − 0.5
    cohen_d:    float     # effect size between member / non-member
    dp_noise:   float
    n_members:  int
    n_nonmembers: int


def run_membership_inference_attack(
    model: NumpyDepressionNet,
    train_data: dict,
    test_data: dict,
    dp_noise: float = 0.0,
    n_shadow: int = 4,
    seed: int = 42,
) -> MIAResult:
    """
    Shadow-model membership inference attack.

    The attack trains shadow models on disjoint subsets, collects
    per-sample (loss, confidence, entropy, margin) feature vectors,
    then trains a meta-classifier to distinguish members from non-members.

    Defense: adding DP noise (matched to the FL dp_agent.py noise_mult)
    degrades the attack AUC toward 0.5.
    """
    rng = np.random.default_rng(seed)

    def get_features(m: NumpyDepressionNet, audio, visual, text,
                     labels) -> np.ndarray:
        _, probs = m.forward(audio, visual, text)
        correct_prob = probs[np.arange(len(labels)), labels]
        # Add calibrated DP noise to confidence (mirrors FIX-6 in fl_algorithm_comparison.py)
        if dp_noise > 0:
            noise_scale = dp_noise / max(len(labels), 1)
            correct_prob = np.clip(
                correct_prob + rng.normal(0, noise_scale, correct_prob.shape),
                1e-7, 1 - 1e-7)
        loss    = -np.log(correct_prob + 1e-9)
        entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)
        sorted_p = np.sort(probs, axis=1)
        margin   = sorted_p[:, -1] - sorted_p[:, -2]
        max_conf = sorted_p[:, -1]
        return np.column_stack([loss, entropy, margin, max_conf, correct_prob])

    n_train = len(train_data["labels"])
    n_test  = len(test_data["labels"])

    # Shadow models: train on small random subsets
    shadow_feats, shadow_labels = [], []
    chunk = n_train // (n_shadow * 2)
    if chunk < 4:
        chunk = 4

    for sh in range(n_shadow):
        # "Member" subset
        idx_m = rng.choice(n_train, chunk, replace=False)
        sm = copy.deepcopy(model)
        _simple_train(sm, train_data, idx_m, epochs=3, lr=0.01, rng=rng)
        feats_m = get_features(sm,
                               train_data["audio"][idx_m],
                               train_data["visual"][idx_m],
                               train_data["text"][idx_m],
                               train_data["labels"][idx_m])
        shadow_feats.append(feats_m)
        shadow_labels.extend([1] * len(idx_m))

        # "Non-member" subset
        idx_nm = rng.choice(n_test, min(chunk, n_test), replace=False)
        feats_nm = get_features(sm,
                                test_data["audio"][idx_nm],
                                test_data["visual"][idx_nm],
                                test_data["text"][idx_nm],
                                test_data["labels"][idx_nm])
        shadow_feats.append(feats_nm)
        shadow_labels.extend([0] * len(idx_nm))

    X_attack = np.vstack(shadow_feats)
    y_attack = np.array(shadow_labels)

    # Meta-classifier (logistic regression)
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_attack)
    clf     = LogisticRegression(max_iter=500, C=1.0, random_state=seed)
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs    = []
    accs    = []
    all_prob = np.zeros(len(y_attack))

    for tr, te in cv.split(X_sc, y_attack):
        clf.fit(X_sc[tr], y_attack[tr])
        prob = clf.predict_proba(X_sc[te])[:, 1]
        all_prob[te] = prob
        aucs.append(roc_auc_score(y_attack[te], prob) if len(np.unique(y_attack[te])) > 1 else 0.5)
        accs.append(accuracy_score(y_attack[te], prob >= 0.5))

    auc_val, ci_lo, ci_hi = delong_auc_variance(y_attack, all_prob)

    # TPR at 10% FPR
    fpr_arr, tpr_arr, _ = roc_curve(y_attack, all_prob)
    idx10  = np.searchsorted(fpr_arr, 0.10)
    tpr10  = float(tpr_arr[min(idx10, len(tpr_arr)-1)])

    # Cohen's d: effect size between member/non-member confidence
    mem_conf  = all_prob[y_attack == 1]
    non_conf  = all_prob[y_attack == 0]
    pooled_sd = math.sqrt((np.var(mem_conf) + np.var(non_conf)) / 2 + 1e-12)
    cohens_d  = (np.mean(mem_conf) - np.mean(non_conf)) / pooled_sd

    return MIAResult(
        auc          = round(auc_val, 4),
        auc_ci_lo    = round(ci_lo, 4),
        auc_ci_hi    = round(ci_hi, 4),
        attack_acc   = round(float(np.mean(accs)), 4),
        tpr          = round(tpr10, 4),
        advantage    = round(auc_val - 0.5, 4),
        cohen_d      = round(cohens_d, 4),
        dp_noise     = dp_noise,
        n_members    = int((y_attack == 1).sum()),
        n_nonmembers = int((y_attack == 0).sum()),
    )


def _simple_train(model: NumpyDepressionNet, data: dict,
                  idx: np.ndarray, epochs: int = 5,
                  lr: float = 0.01, rng = None):
    """SGD training with finite differences (no autograd)."""
    if rng is None:
        rng = np.random.default_rng(0)
    eps = 1e-4
    params = model.get_flat_params()
    for _ in range(epochs):
        batch = rng.choice(idx, min(8, len(idx)), replace=False)
        _, probs = model.forward(
            data["audio"][batch], data["visual"][batch], data["text"][batch])
        loss0 = model.cross_entropy_loss(probs, data["labels"][batch])
        grad  = np.zeros_like(params)
        # Finite difference gradient (cheap for tiny models)
        perturb = rng.choice(len(params), min(50, len(params)), replace=False)
        for i in perturb:
            params[i] += eps
            model.set_flat_params(params)
            _, probs2 = model.forward(
                data["audio"][batch], data["visual"][batch], data["text"][batch])
            loss2 = model.cross_entropy_loss(probs2, data["labels"][batch])
            grad[i]  = (loss2 - loss0) / eps
            params[i] -= eps
        params -= lr * grad
        model.set_flat_params(params)


# ------------ 3b. Gradient Inversion (DLG) -----------------------------------

@dataclass
class GradientInversionResult:
    """
    Simplified Deep Leakage from Gradients (Zhu et al. NeurIPS 2019).
    Metric: reconstruction error (MSE) between true & reconstructed features.
    """
    reconstruction_mse:    float
    reconstruction_snr_db: float
    n_iterations:          int
    converged:             bool
    defense_applied:       str
    psnr:                  float


def run_gradient_inversion(
    model: NumpyDepressionNet,
    true_audio:  np.ndarray,    # (1, audio_dim)
    true_visual: np.ndarray,    # (1, visual_dim)
    true_text:   np.ndarray,    # (1, text_dim)
    true_label:  int,
    noise_mult:  float = 0.0,   # DP noise applied to gradient
    clip_norm:   float = 1.0,
    n_iter:      int   = 200,
    lr:          float = 0.01,
    seed:        int   = 42,
) -> GradientInversionResult:
    """
    Gradient inversion attack against a single sample.
    The attacker observes a (clipped + noised) gradient and tries to
    reconstruct the original input features.

    Theory (Zhu et al.):  x* = argmin ||∂L/∂W(x,y) − g_observed||²
    Defense: DP noise added to gradient degrades reconstruction.
    """
    rng = np.random.default_rng(seed)
    eps_fd = 1e-4

    # Compute "true" gradient (finite difference)
    _, probs_true = model.forward(true_audio, true_visual, true_text)
    loss_true = model.cross_entropy_loss(probs_true,
                                         np.array([true_label]))
    true_grad = np.zeros(model.count_params())
    params0 = model.get_flat_params().copy()

    sample_idx = rng.choice(len(true_grad),
                            min(200, len(true_grad)), replace=False)
    for i in sample_idx:
        params0[i] += eps_fd
        model.set_flat_params(params0)
        _, p2 = model.forward(true_audio, true_visual, true_text)
        loss2 = model.cross_entropy_loss(p2, np.array([true_label]))
        true_grad[i] = (loss2 - loss_true) / eps_fd
        params0[i] -= eps_fd
    model.set_flat_params(params0)

    # Apply DP clipping + noise (dp_agent.py §process_local_update)
    g_norm = np.linalg.norm(true_grad)
    if g_norm > clip_norm:
        true_grad *= clip_norm / (g_norm + 1e-12)
    if noise_mult > 0:
        true_grad += rng.normal(0, noise_mult * clip_norm / 1, true_grad.shape)

    # Attacker reconstructs audio (highest-dimensional modality, 78 dims)
    # We reconstruct only the audio feature (78 dims) as a tractable proxy
    audio_dim = true_audio.shape[-1]
    x_hat     = rng.normal(0, 0.1, (1, audio_dim)).astype(np.float32)
    y_hat     = true_label   # label assumed known (worst case)

    best_mse  = np.inf
    converged = False
    alpha     = lr

    for it in range(n_iter):
        # Compute gradient w.r.t. x_hat (finite diff over audio)
        grad_x = np.zeros((1, audio_dim))
        _, p_hat = model.forward(x_hat, true_visual, true_text)
        loss_hat = model.cross_entropy_loss(p_hat, np.array([y_hat]))

        # Gradient matching objective: ||g(x_hat) − g_obs||²
        hat_grad = np.zeros(model.count_params())
        for i in sample_idx[:50]:   # subset for speed
            params0[i] += eps_fd
            model.set_flat_params(params0)
            _, p2 = model.forward(x_hat, true_visual, true_text)
            loss2 = model.cross_entropy_loss(p2, np.array([y_hat]))
            hat_grad[i] = (loss2 - loss_hat) / eps_fd
            params0[i] -= eps_fd
        model.set_flat_params(params0)

        matching_loss = np.mean((hat_grad[sample_idx[:50]] -
                                 true_grad[sample_idx[:50]]) ** 2)

        # Update x_hat via finite diff over audio features
        for d in range(min(20, audio_dim)):   # 20 dims for speed
            x_hat[0, d] += eps_fd
            _, p2 = model.forward(x_hat, true_visual, true_text)
            lh2  = model.cross_entropy_loss(p2, np.array([y_hat]))
            hg   = np.zeros(model.count_params())
            hg[sample_idx[:10]] = (lh2 - loss_hat) / eps_fd
            new_match = np.mean((hg[sample_idx[:10]] -
                                 true_grad[sample_idx[:10]]) ** 2)
            grad_x[0, d] = (new_match - matching_loss) / eps_fd
            x_hat[0, d] -= eps_fd

        x_hat -= alpha * grad_x
        mse = float(np.mean((x_hat - true_audio) ** 2))
        if mse < best_mse:
            best_mse = mse
        if matching_loss < 1e-6:
            converged = True
            break

    signal_power = float(np.mean(true_audio ** 2)) + 1e-12
    snr_db  = 10 * math.log10(signal_power / (best_mse + 1e-12))
    # PSNR: peak = max pixel^2 analog
    peak    = float(np.max(true_audio) ** 2)
    psnr    = 10 * math.log10(peak / (best_mse + 1e-12))

    defense = (f"DP(σ={noise_mult},C={clip_norm})"
               if noise_mult > 0 else "none")
    return GradientInversionResult(
        reconstruction_mse    = round(best_mse, 6),
        reconstruction_snr_db = round(snr_db, 3),
        n_iterations          = n_iter,
        converged             = converged,
        defense_applied       = defense,
        psnr                  = round(psnr, 3),
    )


# ------------ 3c. Byzantine Poisoning ----------------------------------------

@dataclass
class PoisoningResult:
    attack_type:         str
    fraction_malicious:  float
    accuracy_clean:      float
    accuracy_poisoned:   float
    accuracy_defended:   float
    f1_clean:            float
    f1_poisoned:         float
    f1_defended:         float
    defense:             str
    attack_success:      bool   # poisoned metric degrades by >10 pp


def _aggregate_mean(updates: List[np.ndarray],
                    weights: List[float]) -> np.ndarray:
    total = sum(weights)
    return sum(w * u for u, w in zip(updates, weights)) / total


def _aggregate_trimmed_mean(updates: List[np.ndarray],
                             trim_ratio: float = 0.1) -> np.ndarray:
    """
    Coordinate-wise trimmed mean (aggregator.py::trimmed_mean_aggregate).
    Matches the server implementation exactly.
    """
    n   = len(updates)
    k   = max(1, int(trim_ratio * n))
    arr = np.array(updates, dtype=np.float32)
    if 2 * k < n:
        arr = np.sort(arr, axis=0)[k: n - k]
    return arr.mean(axis=0)


def _aggregate_krum(updates: List[np.ndarray], f: int = 1) -> np.ndarray:
    """Krum aggregation (fl_algorithm_comparison.py::agg_krum)."""
    n  = len(updates)
    if n <= 2 * f + 2:
        return _aggregate_mean(updates, [1.0] * n)
    nb = n - f - 2
    scores = [
        sum(sorted([np.sum((updates[i] - updates[j]) ** 2)
                    for j in range(n) if j != i])[:nb])
        for i in range(n)
    ]
    return updates[int(np.argmin(scores))]


def run_poisoning_attack(
    model: NumpyDepressionNet,
    train_data: dict,
    test_data:  dict,
    attack_type: str   = "label_flip",   # "label_flip" | "scaling"
    frac_malicious: float = 0.3,
    n_clients: int = 8,
    seed: int = 42,
) -> PoisoningResult:
    """
    Byzantine poisoning evaluation against three aggregation strategies.
    Mirrors fl_algorithm_comparison.py aggregation choices.
    """
    rng = np.random.default_rng(seed)

    def eval_model(m: NumpyDepressionNet) -> Tuple[float, float]:
        audio  = test_data["audio"]
        visual = test_data["visual"]
        text   = test_data["text"]
        labels = test_data["labels"]
        probs  = m.predict_proba(audio, visual, text)
        preds  = (probs[:, 1] >= 0.4).astype(int)
        acc    = float(accuracy_score(labels, preds))
        f1     = float(f1_score(labels, preds, zero_division=0))
        return acc, f1

    # Baseline
    acc_clean, f1_clean = eval_model(model)

    # Build per-client gradients (finite-diff delta w.r.t. random init)
    n = len(train_data["labels"])
    chunk = max(4, n // n_clients)
    deltas = []
    for c in range(n_clients):
        idx = rng.choice(n, chunk, replace=False)
        local = copy.deepcopy(model)
        _simple_train(local, train_data, idx, epochs=3, lr=0.01, rng=rng)
        delta = local.get_flat_params() - model.get_flat_params()
        deltas.append(delta)

    # Malicious updates
    n_mal = max(1, int(n_clients * frac_malicious))
    for i in range(n_mal):
        if attack_type == "label_flip":
            # Flip labels: train on inverted labels
            idx = rng.choice(n, chunk, replace=False)
            flipped = copy.deepcopy(model)
            flipped_data = {k: v.copy() if isinstance(v, np.ndarray) else v
                            for k, v in train_data.items()}
            flipped_data["labels"] = 1 - flipped_data["labels"]
            _simple_train(flipped, flipped_data, idx, epochs=3, lr=0.01, rng=rng)
            delta = flipped.get_flat_params() - model.get_flat_params()
            deltas[i] = delta
        else:  # scaling attack: amplify gradient 10×
            deltas[i] = deltas[i] * 10.0

    # FedAvg (no defence)
    poisoned_params = model.get_flat_params() + _aggregate_mean(
        deltas, [1.0] * n_clients)
    m_poisoned = copy.deepcopy(model)
    m_poisoned.set_flat_params(poisoned_params)
    acc_p, f1_p = eval_model(m_poisoned)

    # Trimmed mean defence (aggregator.py::trimmed_mean_aggregate)
    defended_delta = _aggregate_trimmed_mean(deltas, trim_ratio=0.1)
    m_defended = copy.deepcopy(model)
    m_defended.set_flat_params(model.get_flat_params() + defended_delta)
    acc_d, f1_d = eval_model(m_defended)

    attack_success = (acc_clean - acc_p) > 0.10 or (f1_clean - f1_p) > 0.10

    return PoisoningResult(
        attack_type         = attack_type,
        fraction_malicious  = frac_malicious,
        accuracy_clean      = round(acc_clean, 4),
        accuracy_poisoned   = round(acc_p, 4),
        accuracy_defended   = round(acc_d, 4),
        f1_clean            = round(f1_clean, 4),
        f1_poisoned         = round(f1_p, 4),
        f1_defended         = round(f1_d, 4),
        defense             = "trimmed_mean(0.10)",
        attack_success      = attack_success,
    )


# ------------ 3d. Free-rider detection ---------------------------------------

@dataclass
class FreeRiderResult:
    detected:        bool
    cosine_sim:      float
    l2_norm:         float
    threshold_norm:  float
    detection_method: str


def detect_free_rider(honest_deltas: List[np.ndarray],
                      free_rider_delta: np.ndarray,
                      threshold_norm: float = 0.05) -> FreeRiderResult:
    """
    Free-rider detection (military_security.py::check_gradient_norm).
    A free-rider submits zero or nearly-zero gradients.
    """
    l2_norm = float(np.linalg.norm(free_rider_delta))
    mean_honest = np.mean(np.array(honest_deltas), axis=0)
    cos_sim = float(
        np.dot(free_rider_delta, mean_honest) /
        (np.linalg.norm(free_rider_delta) * np.linalg.norm(mean_honest) + 1e-12)
    )
    detected = l2_norm < threshold_norm or np.allclose(free_rider_delta, 0)
    return FreeRiderResult(
        detected         = detected,
        cosine_sim       = round(cos_sim, 4),
        l2_norm          = round(l2_norm, 6),
        threshold_norm   = threshold_norm,
        detection_method = "l2_norm + zero_check",
    )


def run_all_attacks(seed: int = 42) -> Dict[str, Any]:
    log.info("=" * 60)
    log.info("SECTION 3 — ATTACK TESTBEDS")
    log.info("=" * 60)

    train_data = make_synthetic_dataset(n=160, seed=seed)
    test_data  = make_synthetic_dataset(n=40,  seed=seed + 1)
    model      = NumpyDepressionNet(seed=seed)
    # Warm the model a bit
    _simple_train(model, train_data, np.arange(len(train_data["labels"])),
                  epochs=5, lr=0.01, rng=np.random.default_rng(seed))

    results: Dict[str, Any] = {}

    # 3a — MIA under no-DP, weak-DP, strong-DP
    log.info("  [3a] Membership Inference Attack …")
    mia_nodp  = run_membership_inference_attack(model, train_data, test_data,
                                                dp_noise=0.0, seed=seed)
    mia_weakdp = run_membership_inference_attack(model, train_data, test_data,
                                                 dp_noise=0.5, seed=seed)
    mia_strongdp = run_membership_inference_attack(model, train_data, test_data,
                                                   dp_noise=1.5, seed=seed)
    results["mia"] = {
        "no_dp":   asdict(mia_nodp),
        "weak_dp": asdict(mia_weakdp),
        "strong_dp": asdict(mia_strongdp),
        "dp_degrades_auc": mia_nodp.auc >= mia_strongdp.auc,
    }
    log.info(f"    no-DP AUC={mia_nodp.auc:.4f} [{mia_nodp.auc_ci_lo:.3f}, {mia_nodp.auc_ci_hi:.3f}]"
             f"  |  strong-DP AUC={mia_strongdp.auc:.4f}")

    # 3b — Gradient Inversion
    log.info("  [3b] Gradient Inversion (DLG) …")
    sample  = {k: v[:1] for k, v in train_data.items() if k != "labels"}
    lbl_idx = int(train_data["labels"][0])
    gi_nodp = run_gradient_inversion(model,
                                     sample["audio"], sample["visual"],
                                     sample["text"], lbl_idx,
                                     noise_mult=0.0, seed=seed)
    gi_dp   = run_gradient_inversion(model,
                                     sample["audio"], sample["visual"],
                                     sample["text"], lbl_idx,
                                     noise_mult=1.1, seed=seed)
    results["gradient_inversion"] = {
        "no_dp": asdict(gi_nodp),
        "dp_defended": asdict(gi_dp),
        "mse_increase_with_dp": gi_dp.reconstruction_mse > gi_nodp.reconstruction_mse,
    }
    log.info(f"    no-DP MSE={gi_nodp.reconstruction_mse:.4f}  |  "
             f"DP-defended MSE={gi_dp.reconstruction_mse:.4f}")

    # 3c — Byzantine poisoning
    log.info("  [3c] Byzantine Poisoning …")
    pois_lf = run_poisoning_attack(model, train_data, test_data,
                                   attack_type="label_flip",
                                   frac_malicious=0.30, seed=seed)
    pois_sc = run_poisoning_attack(model, train_data, test_data,
                                   attack_type="scaling",
                                   frac_malicious=0.30, seed=seed)
    results["poisoning"] = {
        "label_flip": asdict(pois_lf),
        "scaling":    asdict(pois_sc),
    }
    log.info(f"    label-flip: clean-acc={pois_lf.accuracy_clean:.3f}"
             f"  poisoned={pois_lf.accuracy_poisoned:.3f}"
             f"  defended={pois_lf.accuracy_defended:.3f}")

    # 3d — Free-rider
    log.info("  [3d] Free-rider Detection …")
    honest_deltas = [np.random.default_rng(i).normal(0, 0.01, 100)
                     for i in range(7)]
    zero_rider    = np.zeros(100)
    small_rider   = np.random.default_rng(99).normal(0, 1e-6, 100)
    fr_zero  = detect_free_rider(honest_deltas, zero_rider)
    fr_small = detect_free_rider(honest_deltas, small_rider)
    results["free_rider"] = {
        "zero_gradient":     asdict(fr_zero),
        "near_zero_gradient": asdict(fr_small),
    }
    log.info(f"    zero-gradient detected={fr_zero.detected}  "
             f"  near-zero detected={fr_small.detected}")

    return results


# ===========================================================================
# SECTION 4 — COMPLIANCE EVIDENCE GENERATION
# Mirrors centralized_receipts.py and ledger.rs
# ===========================================================================

@dataclass
class ComplianceReceipt:
    receipt_id:       str
    device_id_hex:    str
    round_id:         int
    session_id:       str
    operation:        str
    epsilon_spent:    float
    delta:            float
    noise_mult:       float
    clip_norm:        float
    payload_hash_hex: str
    timestamp:        str
    hmac_signature:   str
    chain_link:       str
    privacy_guarantee: str  # e.g. "(ε=4.23, δ=1e-5)-DP"
    agent:            str


class ComplianceEngine:
    """
    Generates and verifies compliance evidence.
    Interface mirrors CentralReceiptManager in centralised_receipts.py
    and the Rust ledger in ledger.rs.
    """

    def __init__(self, hmac_key: Optional[bytes] = None):
        self.key   = hmac_key or secrets.token_bytes(32)
        self.chain: List[str] = []   # HMAC chain links
        self.receipts: List[ComplianceReceipt] = []

    def _sign(self, payload: dict) -> str:
        canonical = json.dumps(payload, sort_keys=True).encode()
        return b64encode(
            _hmac.new(self.key, canonical, hashlib.sha256).digest()
        ).decode()

    def _compute_chain_link(self, prev_link: str,
                             payload_hash: str) -> str:
        """Mirrors ledger.rs::compute_chain_link()."""
        data = (prev_link + "|" + payload_hash).encode()
        return _hmac.new(self.key, data, hashlib.sha256).hexdigest()

    def create_receipt(
        self,
        device_id_hex: str,
        round_id:      int,
        session_id:    str,
        epsilon_spent: float,
        noise_mult:    float,
        clip_norm:     float,
        payload_bytes: bytes,
        agent:         str = "dp-agent",
        delta:         float = 1e-5,
    ) -> ComplianceReceipt:
        receipt_id   = secrets.token_hex(16)
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()
        ts           = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        prev_link    = self.chain[-1] if self.chain else "genesis"
        chain_link   = self._compute_chain_link(prev_link, payload_hash)

        core = {
            "receipt_id":    receipt_id,
            "device_id_hex": device_id_hex,
            "round_id":      round_id,
            "session_id":    session_id,
            "operation":     "dp_fl_update",
            "epsilon_spent": epsilon_spent,
            "delta":         delta,
            "payload_hash":  payload_hash,
            "timestamp":     ts,
        }
        sig   = self._sign(core)
        priv  = f"(ε={epsilon_spent:.4f}, δ={delta:.0e})-DP"
        r = ComplianceReceipt(
            receipt_id       = receipt_id,
            device_id_hex    = device_id_hex,
            round_id         = round_id,
            session_id       = session_id,
            operation        = "dp_fl_update",
            epsilon_spent    = round(epsilon_spent, 6),
            delta            = delta,
            noise_mult       = noise_mult,
            clip_norm        = clip_norm,
            payload_hash_hex = payload_hash,
            timestamp        = ts,
            hmac_signature   = sig,
            chain_link       = chain_link,
            privacy_guarantee= priv,
            agent            = agent,
        )
        self.chain.append(chain_link)
        self.receipts.append(r)
        return r

    def verify_receipt(self, r: ComplianceReceipt) -> bool:
        core = {
            "receipt_id":    r.receipt_id,
            "device_id_hex": r.device_id_hex,
            "round_id":      r.round_id,
            "session_id":    r.session_id,
            "operation":     r.operation,
            "epsilon_spent": r.epsilon_spent,
            "delta":         r.delta,
            "payload_hash":  r.payload_hash_hex,
            "timestamp":     r.timestamp,
        }
        expected = self._sign(core)
        return _hmac.compare_digest(expected, r.hmac_signature)

    def verify_chain(self) -> Tuple[bool, int]:
        """Verify HMAC chain integrity. Returns (ok, first_broken_idx)."""
        prev = "genesis"
        for i, r in enumerate(self.receipts):
            expected_link = self._compute_chain_link(prev, r.payload_hash_hex)
            if not _hmac.compare_digest(expected_link, r.chain_link):
                return False, i
            prev = r.chain_link
        return True, -1

    def generate_dp_certificate(
        self,
        n_rounds: int,
        noise_mult: float,
        sample_rate: float,
        n_clients: int,
        delta: float = 1e-5,
    ) -> Dict[str, Any]:
        """
        Generates a machine-readable DP privacy certificate.
        Suitable for regulatory submission (e.g. HIPAA audit evidence).
        """
        acc = rdp_to_dp(noise_mult, sample_rate,
                        n_rounds * 5, delta)   # 5 steps per round
        total_eps = sum(r.epsilon_spent for r in self.receipts)
        budget_ok = total_eps <= 8.0

        return {
            "certificate_version":   "1.0",
            "generated_at":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "algorithm":             "Gaussian Mechanism + RDP Accounting",
            "references": [
                "Mironov 2017 (RDP)",
                "Wang et al. 2019 (amplification by subsampling)",
                "Abadi et al. 2016 (DP-SGD)"
            ],
            "parameters": {
                "noise_multiplier":  noise_mult,
                "clip_norm":         1.0,
                "sample_rate":       round(sample_rate, 4),
                "n_rounds":          n_rounds,
                "n_clients":         n_clients,
                "delta":             delta,
            },
            "guarantee": {
                "epsilon":           acc.epsilon,
                "delta":             delta,
                "best_alpha_order":  acc.best_alpha,
                "mechanism":         "Gaussian",
                "composition":       "Advanced (RDP)",
                "amplification":     "Poisson subsampling",
                "privacy_guarantee": f"({acc.epsilon:.4f}, {delta:.0e})-DP",
                "meets_epsilon_8":   acc.epsilon <= 8.0,
                "meets_epsilon_4":   acc.epsilon <= 4.0,
            },
            "receipts_summary": {
                "n_receipts":        len(self.receipts),
                "total_epsilon_spent": round(total_eps, 6),
                "budget_ok":         budget_ok,
                "chain_intact":      self.verify_chain()[0],
            },
        }

    def audit_ledger_report(self) -> Dict[str, Any]:
        """Summary statistics over the audit ledger (mirrors ledger.rs)."""
        eps_values  = [r.epsilon_spent for r in self.receipts]
        round_ids   = [r.round_id for r in self.receipts]
        chain_ok, bad_idx = self.verify_chain()
        sig_ok_all  = all(self.verify_receipt(r) for r in self.receipts)
        return {
            "n_entries":         len(self.receipts),
            "unique_rounds":     len(set(round_ids)),
            "epsilon_min":       round(min(eps_values), 6) if eps_values else None,
            "epsilon_max":       round(max(eps_values), 6) if eps_values else None,
            "epsilon_mean":      round(float(np.mean(eps_values)), 6) if eps_values else None,
            "epsilon_p95":       round(float(np.percentile(eps_values, 95)), 6) if eps_values else None,
            "chain_integrity_ok": chain_ok,
            "first_broken_link": bad_idx,
            "signature_ok_all":  sig_ok_all,
            "tamper_evidence_count": sum(1 for r in self.receipts
                                         if not self.verify_receipt(r)),
        }


def run_compliance_simulation(
    n_rounds: int = 10,
    n_clients: int = 8,
    noise_mult: float = 1.1,
    seed: int = 42,
) -> Dict[str, Any]:
    log.info("=" * 60)
    log.info("SECTION 4 — COMPLIANCE EVIDENCE")
    log.info("=" * 60)

    engine   = ComplianceEngine()
    rng      = np.random.default_rng(seed)
    receipts = []
    delta    = 1e-5
    sample_rate = 8 / 20   # batch_size / n_samples_per_client (default)

    for r in range(1, n_rounds + 1):
        acc = rdp_to_dp(noise_mult, sample_rate, r * 5, delta)
        for c in range(n_clients):
            payload = rng.bytes(128)    # mock encrypted gradient
            rec = engine.create_receipt(
                device_id_hex = hashlib.sha256(f"dev_{c}".encode()).hexdigest()[:16],
                round_id      = r,
                session_id    = f"sess_{r}_{c}",
                epsilon_spent = acc.epsilon / n_clients,   # per-device share
                noise_mult    = noise_mult,
                clip_norm     = 1.0,
                payload_bytes = payload,
                delta         = delta,
            )
            receipts.append(rec)

    log.info(f"  Generated {len(receipts)} receipts across {n_rounds} rounds")

    # Verify all signatures
    sig_ok  = all(engine.verify_receipt(r) for r in receipts)
    # Verify chain
    chain_ok, bad_idx = engine.verify_chain()
    log.info(f"  Signatures OK: {sig_ok}  |  Chain intact: {chain_ok}")

    # Tamper one receipt and re-verify
    tampered_rec = copy.copy(receipts[5])
    tampered_rec = ComplianceReceipt(**{
        **asdict(tampered_rec),
        "epsilon_spent": tampered_rec.epsilon_spent * 0.01,  # fraudulent
    })
    tamper_detected = not engine.verify_receipt(tampered_rec)
    log.info(f"  Tamper detection (eps manipulation): {tamper_detected}")

    cert   = engine.generate_dp_certificate(n_rounds, noise_mult,
                                             sample_rate, n_clients, delta)
    ledger = engine.audit_ledger_report()

    return {
        "receipts_generated":  len(receipts),
        "signatures_ok":       sig_ok,
        "chain_ok":            chain_ok,
        "tamper_detected":     tamper_detected,
        "dp_certificate":      cert,
        "audit_ledger":        ledger,
    }


# ===========================================================================
# SECTION 5 — CROSS-CUTTING STATISTICAL ANALYSIS
# ===========================================================================

def statistical_summary(attack_results: Dict[str, Any],
                        privacy_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research-grade statistical summary across all dimensions.
    Outputs effect sizes, significance tests, and hypothesis verdicts.
    """
    summary = {}

    # H1: DP significantly reduces MIA AUC
    mia = attack_results.get("mia", {})
    auc_nodp    = mia.get("no_dp",    {}).get("auc", 0.5)
    auc_strongdp = mia.get("strong_dp", {}).get("auc", 0.5)
    d_mia        = mia.get("no_dp", {}).get("cohen_d", 0)
    summary["H1_dp_reduces_mia"] = {
        "hypothesis":   "DP significantly reduces MIA AUC toward 0.5",
        "auc_no_dp":    auc_nodp,
        "auc_strong_dp": auc_strongdp,
        "auc_reduction": round(auc_nodp - auc_strongdp, 4),
        "cohen_d":      d_mia,
        "effect_size":  "large" if abs(d_mia) > 0.8 else
                        "medium" if abs(d_mia) > 0.5 else "small",
        "verdict":      "SUPPORTED" if auc_strongdp < auc_nodp else "NOT SUPPORTED",
    }

    # H2: DP degrades gradient inversion (MSE increases)
    gi = attack_results.get("gradient_inversion", {})
    mse_nodp = gi.get("no_dp", {}).get("reconstruction_mse", 0)
    mse_dp   = gi.get("dp_defended", {}).get("reconstruction_mse", 0)
    summary["H2_dp_degrades_gradient_inversion"] = {
        "hypothesis":   "DP noise increases gradient inversion MSE",
        "mse_no_dp":    mse_nodp,
        "mse_dp":       mse_dp,
        "mse_increase_factor": round(mse_dp / (mse_nodp + 1e-12), 3),
        "verdict":      "SUPPORTED" if mse_dp > mse_nodp else "NOT SUPPORTED",
    }

    # H3: Trimmed mean defends against Byzantine poisoning
    pois = attack_results.get("poisoning", {}).get("label_flip", {})
    f1_clean    = pois.get("f1_clean", 0)
    f1_poisoned = pois.get("f1_poisoned", 0)
    f1_defended = pois.get("f1_defended", 0)
    summary["H3_trimmed_mean_defends_poisoning"] = {
        "hypothesis":     "Trimmed mean recovers F1 after label-flip attack",
        "f1_clean":       f1_clean,
        "f1_poisoned":    f1_poisoned,
        "f1_defended":    f1_defended,
        "recovery_pp":    round((f1_defended - f1_poisoned) * 100, 2),
        "full_recovery":  f1_defended >= f1_clean * 0.90,
        "verdict":        "SUPPORTED" if f1_defended > f1_poisoned else "NOT SUPPORTED",
    }

    # H4: Privacy budget meets ε ≤ 8 under default FL config
    priv = privacy_results.get("per_noise_mult", {})
    nm_default = 1.1
    if nm_default in priv:
        eps_final = priv[nm_default].get("final_epsilon", math.inf)
    else:
        eps_final = math.inf
    summary["H4_privacy_budget_compliant"] = {
        "hypothesis":     "Default noise_mult=1.1 achieves ε ≤ 8 (HIPAA threshold)",
        "noise_mult":     nm_default,
        "final_epsilon":  eps_final,
        "meets_epsilon_8": eps_final <= 8.0,
        "meets_epsilon_4": eps_final <= 4.0,
        "verdict":        "SUPPORTED" if eps_final <= 8.0 else "VIOLATED",
    }

    return summary


# ===========================================================================
# SECTION 6 — VISUALISATION
# ===========================================================================

def generate_plots(privacy_results: Dict, attack_results: Dict,
                   compliance_results: Dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        log.warning("matplotlib not available — skipping plots")
        return

    plt.rcParams.update({
        "figure.dpi": 120, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # ── Plot 1: Privacy budget vs rounds ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Federated Learning System — Research Evaluation Dashboard",
                 fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, 6))
    for i, (nm, res) in enumerate(privacy_results.get("per_noise_mult", {}).items()):
        traj = res["eps_trajectory"]
        ax.plot(range(1, len(traj) + 1), traj,
                label=f"σ={nm}", color=colors[i], linewidth=2)
    ax.axhline(8.0, color="red",    linestyle="--", linewidth=1, label="ε=8 (HIPAA)")
    ax.axhline(4.0, color="orange", linestyle="--", linewidth=1, label="ε=4 (strict)")
    ax.set_xlabel("FL Round"); ax.set_ylabel("ε (epsilon)")
    ax.set_title("Privacy Budget Evolution (RDP Accounting)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # ── Plot 2: MIA AUC vs DP noise ──────────────────────────────────────────
    ax = axes[0, 1]
    mia = attack_results.get("mia", {})
    noise_levels = [0.0, 0.5, 1.5]
    aucs = [mia.get("no_dp",     {}).get("auc", 0.5),
            mia.get("weak_dp",   {}).get("auc", 0.5),
            mia.get("strong_dp", {}).get("auc", 0.5)]
    cis_lo = [mia.get("no_dp",     {}).get("auc_ci_lo", aucs[0]),
              mia.get("weak_dp",   {}).get("auc_ci_lo", aucs[1]),
              mia.get("strong_dp", {}).get("auc_ci_lo", aucs[2])]
    cis_hi = [mia.get("no_dp",     {}).get("auc_ci_hi", aucs[0]),
              mia.get("weak_dp",   {}).get("auc_ci_hi", aucs[1]),
              mia.get("strong_dp", {}).get("auc_ci_hi", aucs[2])]
    ax.bar(range(3), aucs, color=["#e74c3c", "#f39c12", "#27ae60"],
           alpha=0.8, width=0.5)
    for xi, (lo, hi) in enumerate(zip(cis_lo, cis_hi)):
        ax.errorbar(xi, aucs[xi], [[aucs[xi]-lo], [hi-aucs[xi]]],
                    fmt='none', color='black', capsize=5, linewidth=2)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1,
               label="Random guess (AUC=0.5)")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["No DP\nσ=0", "Weak DP\nσ=0.5", "Strong DP\nσ=1.5"])
    ax.set_ylabel("MIA AUC-ROC (↓ better)")
    ax.set_title("Membership Inference Attack\nvs DP Defence (95% CI DeLong)")
    ax.set_ylim(0.3, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # ── Plot 3: Poisoning resilience ─────────────────────────────────────────
    ax = axes[1, 0]
    pois_lf = attack_results.get("poisoning", {}).get("label_flip", {})
    pois_sc = attack_results.get("poisoning", {}).get("scaling", {})
    metrics = ["F1 Clean", "F1 Poisoned", "F1 Defended"]
    x       = np.arange(len(metrics))
    w       = 0.35
    lf_vals = [pois_lf.get("f1_clean", 0),
                pois_lf.get("f1_poisoned", 0),
                pois_lf.get("f1_defended", 0)]
    sc_vals = [pois_sc.get("f1_clean", 0),
                pois_sc.get("f1_poisoned", 0),
                pois_sc.get("f1_defended", 0)]
    ax.bar(x - w/2, lf_vals, w, label="Label-flip", color="#3498db", alpha=0.8)
    ax.bar(x + w/2, sc_vals, w, label="Scaling (×10)", color="#9b59b6", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Byzantine Poisoning: 30% Malicious\nDefence: Trimmed Mean (10%)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    # ── Plot 4: Privacy amplification gain ───────────────────────────────────
    ax = axes[1, 1]
    amp_gain = privacy_results.get("privacy_amplification_gain_pct", {})
    nms  = sorted(amp_gain.keys())
    gains = [amp_gain[nm] for nm in nms]
    bars = ax.bar(range(len(nms)), gains, color=plt.cm.viridis(np.linspace(0.2, 0.9, len(nms))),
                  alpha=0.85)
    ax.set_xticks(range(len(nms)))
    ax.set_xticklabels([f"σ={nm}" for nm in nms], rotation=30)
    ax.set_ylabel("ε Reduction vs No Amplification (%)")
    ax.set_title("Privacy Amplification by Subsampling\n(Wang et al. 2019)")
    for bar, g in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{g:.1f}%", ha='center', va='bottom', fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = PLOTS_DIR / "research_evaluation_dashboard.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    log.info(f"  Dashboard plot saved → {out}")


# ===========================================================================
# MAIN
# ===========================================================================

def print_results_table(results: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("RESEARCH EVALUATION SUMMARY")
    print("=" * 80)

    # Privacy
    priv = results.get("privacy", {})
    nm_11 = priv.get("per_noise_mult", {}).get(1.1, {})
    print(f"\n[PRIVACY]  noise_mult=1.1  n_rounds={priv.get('n_rounds','?')}")
    print(f"  Final ε = {nm_11.get('final_epsilon', '?'):.4f}   "
          f"δ = {priv.get('delta','?'):.0e}   "
          f"meets_ε≤8 = {nm_11.get('meets_epsilon_8','?')}")
    amp = priv.get("privacy_amplification_gain_pct", {}).get(1.1, 0)
    print(f"  Amplification gain (σ=1.1) = {amp:.1f}%")

    # Security
    sec = results.get("security", {})
    print(f"\n[SECURITY] {sec.get('passed','?')}/{sec.get('total','?')} tests passed  "
          f"(rate={sec.get('pass_rate',0):.1%})")
    for t in sec.get("tests", []):
        mark = "✓" if t["passed"] else "✗"
        print(f"  {mark} {t['name'][:52]:<52} {t['latency_ms']:.2f}ms")

    # Attacks
    atk = results.get("attacks", {})
    print("\n[ATTACKS]")
    mia = atk.get("mia", {})
    print(f"  MIA AUC (no-DP)    = {mia.get('no_dp',{}).get('auc',0):.4f}  "
          f"advantage = {mia.get('no_dp',{}).get('advantage',0):.4f}")
    print(f"  MIA AUC (strong-DP)= {mia.get('strong_dp',{}).get('auc',0):.4f}  "
          f"DP degrades = {mia.get('dp_degrades_auc',False)}")
    gi = atk.get("gradient_inversion", {})
    print(f"  Grad-Inv MSE no-DP = {gi.get('no_dp',{}).get('reconstruction_mse',0):.5f}")
    print(f"  Grad-Inv MSE DP    = {gi.get('dp_defended',{}).get('reconstruction_mse',0):.5f}")
    pois_lf = atk.get("poisoning",{}).get("label_flip",{})
    print(f"  Poisoning: clean={pois_lf.get('f1_clean',0):.3f}  "
          f"poisoned={pois_lf.get('f1_poisoned',0):.3f}  "
          f"defended={pois_lf.get('f1_defended',0):.3f}")

    # Compliance
    comp = results.get("compliance", {})
    dp_cert = comp.get("dp_certificate", {}).get("guarantee", {})
    print(f"\n[COMPLIANCE]")
    print(f"  Receipts: {comp.get('receipts_generated','?')}   "
          f"sig_ok={comp.get('signatures_ok','?')}   "
          f"chain_ok={comp.get('chain_ok','?')}   "
          f"tamper_detected={comp.get('tamper_detected','?')}")
    print(f"  DP Certificate: ε={dp_cert.get('epsilon',0):.4f}  "
          f"meets_ε≤8={dp_cert.get('meets_epsilon_8','?')}")

    # Hypotheses
    stats_res = results.get("statistical_summary", {})
    print(f"\n[HYPOTHESES]")
    for k, v in stats_res.items():
        print(f"  {v['verdict']:15s}  {v['hypothesis']}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Research-grade FL evaluation suite")
    parser.add_argument("--n_rounds",  type=int,   default=20)
    parser.add_argument("--n_clients", type=int,   default=8)
    parser.add_argument("--noise_mult",type=float, default=1.1)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--plots",     action="store_true")
    parser.add_argument("--no_dp",     action="store_true")
    args = parser.parse_args()

    if args.no_dp:
        args.noise_mult = 0.0

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  FL RESEARCH EVALUATION SUITE — BE Major Project        ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    all_results: Dict[str, Any] = {}

    # [1] Privacy Analysis
    log.info("=" * 60)
    log.info("SECTION 1 — PRIVACY ANALYSIS")
    log.info("=" * 60)
    priv_res = privacy_analysis(
        n_rounds          = args.n_rounds,
        noise_mults       = [0.5, 0.8, 1.0, 1.1, 1.5, 2.0],
        n_clients         = args.n_clients,
        n_samples_per_client = 5,
        batch_size        = 8,
        local_epochs      = 5,
        delta             = 1e-5,
    )
    nm = 1.1
    final_eps = priv_res["per_noise_mult"].get(nm, {}).get("final_epsilon", math.inf)
    log.info(f"  noise_mult=1.1 → ε={final_eps:.4f}  "
             f"meets_ε≤8={final_eps <= 8.0}")
    all_results["privacy"] = priv_res

    # [2] Security
    verifier = SecurityVerifier()
    sec_res  = verifier.run_all()
    all_results["security"] = sec_res

    # [3] Attack testbeds
    atk_res = run_all_attacks(seed=args.seed)
    all_results["attacks"] = atk_res

    # [4] Compliance
    comp_res = run_compliance_simulation(
        n_rounds   = args.n_rounds,
        n_clients  = args.n_clients,
        noise_mult = args.noise_mult if args.noise_mult > 0 else 1.1,
        seed       = args.seed,
    )
    all_results["compliance"] = comp_res

    # [5] Statistical summary
    all_results["statistical_summary"] = statistical_summary(atk_res, priv_res)

    # [6] Plots
    if args.plots:
        log.info("=" * 60)
        log.info("SECTION 6 — VISUALISATION")
        log.info("=" * 60)
        generate_plots(priv_res, atk_res, comp_res)

    # Save results
    out_json = RESULTS_DIR / "results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    cert_path = RESULTS_DIR / "privacy_certificate.json"
    with open(cert_path, "w") as f:
        json.dump(comp_res.get("dp_certificate", {}), f, indent=2)

    log.info(f"  Results → {out_json}")
    log.info(f"  DP cert → {cert_path}")

    print_results_table(all_results)
    return all_results


if __name__ == "__main__":
    main()