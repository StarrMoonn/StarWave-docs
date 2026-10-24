#!/usr/bin/env python3
"""Reproduce seven 2-D CPU surface comparisons using ordinary StarWave imports.

Install the source commit recorded below, Torch, NumPy and Matplotlib first.
No native library, CUDA device, source-path injection, or kernel edit is used.
Example: python run_free_surface_acoustic.py --output results
"""
from __future__ import annotations
import argparse
import datetime as dt_clock
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import torch
import starwave
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE_COMMIT = "200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb"
CASES = ("scalar", "born", "vrz", "vrr", "vti", "tti", "gsls")
NX, NZ, DX, DT, NT, ORDER, PML = 80, 64, 10.0, 0.001, 750, 4, 12
FREQUENCY, PEAK_TIME = 20.0, 0.075
DESCRIPTIONS = {
    "scalar": ("Scalar acoustic", "pressure, normalized forcing", "Fixed-zero external halo; physical z=0 remains active. Order 4 truncation is not an odd-mirror high-order closure."),
    "born": ("Scalar Born", "linearized scattered pressure", "Zero-PML reflective scalar boundary in the vendored Born solver; physical edge remains active with external zero halo. Not elastic traction-free."),
    "vrz": ("Acoustic VRZ", "pressure, normalized forcing", "Top PML and buffer removed; M=order/2 zero halo cells remain. Physical z=0 remains active."),
    "vrr": ("Acoustic VRR", "pressure, normalized forcing", "Top PML and buffer removed; M=order/2 zero halo cells remain. Physical z=0 remains active."),
    "vti": ("Duveneck acoustic VTI", "vertical particle velocity vz (m/s)", "Physical z=0 satisfies sV=0. Surface stiffness is locally projected. vz is sampled on its native vertical half-cell grid."),
    "tti": ("Local qP TTI", "single qP scalar field u", "Physical z=0 is projected to u=0. Normal derivatives use order-2 closure on the first M interior rows; not elastic traction-free."),
    "gsls": ("Viscoacoustic GSLS", "pressure (Pa)", "Physical z=0 is projected to p=0 with the supported normal-derivative closure. Not an elastic P/S free surface."),
}


def geometry():
    vp = torch.full((NX, NZ), 2000.0, dtype=torch.float32)
    rho = torch.full_like(vp, 2000.0)
    vp[:, 30:] = 2600.0
    rho[:, 30:] = 2200.0
    scatter = torch.zeros_like(vp)
    scatter[:, 30:] = 100.0
    src = torch.tensor([[[40, 8]]], dtype=torch.long)
    rec = torch.tensor([[[x, 5] for x in range(10, 71)]], dtype=torch.long)
    t = torch.arange(NT, dtype=vp.dtype) * DT
    a = (math.pi * FREQUENCY * (t - PEAK_TIME)).square()
    wavelet = ((1 - 2 * a) * torch.exp(-a))[None, None, :]
    # VRR r = 1/2 grad(log Z), in inverse metres. This explicit model mapping
    # is not a claim of discrete equivalence between the VRZ and VRR operators.
    logz = torch.log(vp * rho)
    rz = torch.zeros_like(vp)
    rz[:, 1:-1] = (logz[:, 2:] - logz[:, :-2]) / (4 * DX)
    rx = torch.zeros_like(vp)
    return vp, rho, scatter, src, rec, wavelet, rx, rz


def time_plan_metadata(case, inputs):
    """Re-evaluate the same pinned setup helpers, without changing propagation."""
    vp, rho, *_ = inputs
    if case == "scalar":
        from starwave.time_sampling import make_time_sampling_plan
        plan = make_time_sampling_plan(vp, DX, DT, NT)
    elif case == "vrz":
        from starwave.time_sampling import make_vrz_time_sampling_plan
        plan = make_vrz_time_sampling_plan(vp, vp*rho, DX, DT, NT, accuracy=ORDER)
    elif case == "vti":
        from starwave._vti_time import make_vti_time_sampling_plan
        plan = make_vti_time_sampling_plan(vp, torch.full_like(vp,.1),
            torch.full_like(vp,.05), rho, DX, DT, NT, accuracy=ORDER)
    elif case == "tti":
        from starwave._tti_setup import plan_time
        plan = plan_time(vp, torch.full_like(vp,.1), torch.full_like(vp,.05),
            torch.full_like(vp,math.radians(20)), (DX,DX), DT, NT, ORDER, None)
    elif case == "born":
        from starwave._deepwave.common import cfl_condition_n
        inner_dt, ratio = cfl_condition_n((DX,DX), DT, float(vp.max()))
        return dict(internal_dt_s=inner_dt, internal_nt=NT*ratio, step_ratio=ratio,
                    provenance="Vendored Born cfl_condition_n at pinned source")
    elif case == "vrr":
        ratio = max(1, math.ceil(DT / (.6*DX/(float(vp.max())*math.sqrt(2)))))
        return dict(internal_dt_s=DT/ratio, internal_nt=NT*ratio, step_ratio=ratio,
                    provenance="Explicit VRR _prepare principal-speed formula at pinned source")
    else:
        return dict(internal_dt_s=DT, internal_nt=NT, step_ratio=1,
                    provenance="GSLS does not resample; uses supplied dt directly")
    return dict(internal_dt_s=plan.internal_dt, internal_nt=plan.internal_nt,
        step_ratio=plan.step_ratio, max_vel_m_s=plan.max_vel,
        cfl_max_dt_s=plan.max_dt, provenance="Same operator-specific setup helper at pinned source")


def gsls_quality_metadata():
    from starwave.visco_gsls_coefficients import (
        coefficients, hao1_relaxation_times, quality_spectrum, fit_diagnostics)
    v = torch.tensor([[2000.0]], dtype=torch.float32)
    q = torch.full_like(v,60.0)
    _, strength, _, _ = coefficients(v,q,DT,FREQUENCY,mode="hao1",
        frequency_band=(1.,200.),n_mechanisms=5)
    times = hao1_relaxation_times((1.,200.),dtype=torch.float32)
    report = fit_diagnostics(q,strength,times,(1.,200.),
                            target_convention="hao1_nominal_Q0")
    return dict(actual_modulus_q_at_f_ref=float(quality_spectrum(strength,times,[FREQUENCY]).item()),
        design_band_max_relative_inverse_q_error=float(report["max_relative_inverse_q_error"]),
        default_fit_tolerance=.05,
        warning="Hao finite-Q approximation exceeds the 5% default inverse-Q tolerance over the 1:200 Hz design band; warning retained, no fit or PDE altered.")


def run_one(case, free, inputs):
    vp, rho, scatter, src, rec, wavelet, rx, rz = inputs
    width = [PML, PML, 0 if free else PML, PML]
    common = dict(source_amplitudes=wavelet, source_locations=src,
                  receiver_locations=rec, accuracy=ORDER, pml_width=width,
                  memory="full", backend="torch", execution="eager")
    standard = dict(common, pml_freq=FREQUENCY)
    extras = {}
    if case == "scalar":
        out = starwave.scalar(vp, DX, DT, **standard)
    elif case == "born":
        # Born follows the input axes directly; this example explicitly uses
        # [x,z] and hence the top is pml_width[2], as in scalar.
        out = starwave.scalar_born(vp, scatter, DX, DT,
            bg_receiver_locations=rec, **standard)
        extras["background"] = out[-2][0].detach().cpu().numpy()
    elif case == "vrz":
        out = starwave.vrz(vp, DX, DT, density=rho, **standard)
    elif case == "vrr":
        out = starwave.vrr(vp, DX, DT, rx=rx, rz=rz, **standard)
    elif case == "vti":
        out = starwave.vti(vp, torch.full_like(vp, .1),
            torch.full_like(vp, .05), rho, DX, DT,
            source_fields=("sH", "sV"), receiver_fields=("vz",), **standard)
    elif case == "tti":
        out = starwave.tti(vp, torch.full_like(vp, .1),
            torch.full_like(vp, .05), torch.full_like(vp, math.radians(20)),
            DX, DT, **standard)
    elif case == "gsls":
        # GSLS explicitly uses [z,x] instead of the other six input layouts.
        common.update(source_locations=src.flip(-1), receiver_locations=rec.flip(-1),
                      pml_width=[0, PML, PML, PML] if free else PML)
        out = starwave.visco_gsls(vp.T.contiguous(), torch.full_like(vp.T, 60.0),
            rho.T.contiguous(), DX, DT, f_ref=FREQUENCY, mode="hao1", frequency_band=(1.0, 200.0), n_mechanisms=5, **common)
    else:
        raise ValueError(case)
    record = out[-1][0].detach().cpu().numpy()
    if record.shape != (61, NT):
        raise RuntimeError(f"Unexpected receiver array shape: {record.shape}")
    if not np.isfinite(record).all():
        raise FloatingPointError(f"{case} {'free' if free else 'absorbing'} produced nonfinite records")
    return record.copy(), extras


def stats(a):
    return {"finite": bool(np.isfinite(a).all()), "max_abs": float(np.max(np.abs(a))),
            "rms": float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))}


def plot_case(case, output, inputs, absorbing, surface, difference, background=False):
    vp, rho, scatter, src, rec, wavelet, rx, rz = inputs
    title, observable, convention = DESCRIPTIONS[case]
    if background:
        title, observable = "Scalar Born background", "background pressure"
    t = np.arange(NT) * DT
    receiver_x = rec[0, :, 0].numpy() * DX
    pair = np.concatenate((absorbing.ravel(), surface.ravel()))
    # One explicitly disclosed symmetric limit for all three panels; no
    # per-gather normalization, trace AGC, muting, or time gain is applied.
    limit = float(np.quantile(np.abs(pair), .995))
    if not limit > 0:
        limit = 1.0
    fig = plt.figure(figsize=(13.2, 9.3), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 2.15))
    model_ax, src_ax, trace_ax = [fig.add_subplot(grid[0, i]) for i in range(3)]
    im = model_ax.imshow(vp.T.numpy(), extent=(0, (NX-1)*DX, (NZ-1)*DX, 0),
        cmap="viridis", aspect="auto", vmin=2000, vmax=2600)
    model_ax.scatter([400], [80], marker="*", s=95, color="#f44336", edgecolor="white", label="source")
    model_ax.plot(receiver_x, np.full_like(receiver_x, 55 if case == "vti" else 50), "v", ms=2.6, color="#ffdc45", label="receivers")
    model_ax.axhline(0, color="#34b8cc", linewidth=2)
    model_ax.set(xlabel="x (m)", ylabel="z (m)", title="Same layered model and acquisition")
    model_ax.legend(loc="lower left", fontsize=8)
    fig.colorbar(im, ax=model_ax, shrink=.75, label="Vp (m/s)")
    src_ax.plot(t, wavelet[0, 0].numpy(), color="#7d4cb0")
    src_ax.set(xlabel="Time (s)", ylabel="Source amplitude", title="20 Hz Ricker, peak at 0.075 s", xlim=(0, .2))
    src_ax.grid(alpha=.2)
    idx = int(np.argmin(np.abs(receiver_x - 500)))
    trace_ax.plot(t, absorbing[idx], label="Top PML", color="#596573", linewidth=1.05)
    trace_ax.plot(t, surface[idx], label="Free top", color="#d55c25", linewidth=.95, alpha=.85)
    trace_ax.set(xlabel="Time (s)", title=f"Unscaled trace at x={receiver_x[idx]:.0f} m", xlim=(0, t[-1]))
    trace_ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    trace_ax.legend(fontsize=8)
    trace_ax.grid(alpha=.2)
    for i, (data, subtitle) in enumerate(((absorbing, "Top PML"), (surface, "Free top"), (difference, "Difference: free top - top PML"))):
        ax = fig.add_subplot(grid[1, i])
        image = ax.imshow(data.T, cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit,
            extent=(receiver_x[0]-DX/2, receiver_x[-1]+DX/2, t[-1]+DT/2, -DT/2))
        ax.set(xlabel="Receiver x (m)", ylabel="Time (s)", title=subtitle)
        cb = fig.colorbar(image, ax=ax, shrink=.76, pad=.02)
        cb.ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    fig.suptitle(f"{title} | 2-D CPU Torch eager | {observable}\n"
        f"Shared colour limits ±{limit:.3g} (99.5% of paired absolute amplitudes); no AGC or time gain",
        fontsize=13)
    suffix = "_background" if background else ""
    fig.savefig(output / f"{case}{suffix}.png", dpi=155, facecolor="white")
    plt.close(fig)
    return limit


def run_case(case, output):
    inputs = geometry()
    vp, rho, scatter, src, rec, wavelet, rx, rz = inputs
    started = time.perf_counter()
    runtime = {}
    records, extras = {}, {}
    with torch.no_grad():
        for label, free in (("absorbing", False), ("free_surface", True)):
            t0 = time.perf_counter()
            print(f"{case}: starting {label}", flush=True)
            records[label], extras[label] = run_one(case, free, inputs)
            runtime[label] = time.perf_counter() - t0
            print(f"{case}: {label} done in {runtime[label]:.3f}s; {stats(records[label])}", flush=True)
    records["difference"] = records["free_surface"] - records["absorbing"]
    metadata = dict(source_commit=SOURCE_COMMIT, dimension=2, device="cpu", backend="torch",
        execution="eager", memory="full", autograd_enabled=False, dtype="float32", threads=2,
        torch_version=torch.__version__, python_version=platform.python_version(),
        starwave_import_path=starwave.__file__, generated_at_utc=dt_clock.datetime.now(dt_clock.timezone.utc).isoformat(),
        case=case, observable=DESCRIPTIONS[case][1], surface_convention=DESCRIPTIONS[case][2],
        grid_shape_xz=[NX, NZ], input_axes="z,x" if case == "gsls" else "x,z",
        dx_m=DX, dz_m=DX, dt_s=DT, nt=NT, spatial_order=ORDER,
        pml_width_absorbing=[PML]*4,
        pml_width_free_surface=[0, PML, PML, PML] if case == "gsls" else [PML, PML, 0, PML],
        pml_face_order="z-low,z-high,x-low,x-high" if case == "gsls" else "x-low,x-high,z-low,z-high",
        boundary_buffer_cells=None if case in ("born", "gsls") else 5,
        model=dict(interface_z_m=300, vp_above_m_s=2000, vp_below_m_s=2600,
            rho_above_kg_m3=2000, rho_below_kg_m3=2200,
            density_note="Not an input to scalar, Born or qP TTI; VRR uses density only in explicit r=0.5 grad(log(vp*rho)) preprocessing."),
        source=dict(type="Ricker", peak_frequency_hz=FREQUENCY, peak_time_s=PEAK_TIME,
            amplitude=1.0, location_xz_m=[400,80], location_xz_indices=[40,8],
            convention="Pa/s rate into both sH and sV" if case == "vti" else "negative second-order forcing Pa/m^2; first-update half weight" if case == "gsls" else "normalized forcing; internal increment -v_source^2*internal_dt^2*upsampled_wavelet"),
        receivers=dict(count=61, x_m_min=100, x_m_max=700, spacing_m=10,
            z_index=5, z_coordinate_m=55 if case == "vti" else 50,
            note="Native vz half-cell grid: integer z=5 lies at z=55m" if case == "vti" else "Off-surface receiver row, z=50m"),
        time_plan=time_plan_metadata(case, inputs), elapsed_s=runtime, stats={k:stats(v) for k,v in records.items()},
        difference_relative_l2=float(np.linalg.norm(records["difference"].astype(np.float64))/max(np.linalg.norm(records["absorbing"].astype(np.float64)),np.finfo(float).tiny)),
        limitations=["Documentation example, not physical acceptance or backend parity validation.",
            "Compare only within each pair: source and observable conventions differ across operators.",
            "Only the top boundary option changes within a pair; no added noise, FWI or 3-D modelling."])
    if case in ("vti", "tti"):
        metadata["model"].update(epsilon=.1, delta=.05)
    if case == "tti":
        metadata["model"]["theta_degrees"] = 20.0
    if case == "gsls":
        metadata["model"].update(q_nominal=60.0, gsls_mode="hao1", f_ref_hz=FREQUENCY,
            frequency_band_hz=[1.0, 200.0], n_mechanisms=5,
            quality_note="Nominal Hao Q0, not exact modulus Q at f_ref; five mechanisms with an explicit 1:200 Hz design band.")
        metadata["quality_diagnostics"] = gsls_quality_metadata()
    if case == "vrr":
        metadata["model"].update(reflectivity_mapping="rx=0; rz=0.5*centered_second_order_derivative(log(vp*rho))", reflectivity_unit="1/m")
    if case == "born":
        metadata["model"].update(scatter_delta_v_below_interface_m_s=100.0, scatter_delta_v_above_interface_m_s=0.0)
        metadata["limitations"].append("Born plot is first-order delta-v scattered pressure, not full nonlinear multiple scattering. Background records are also saved separately.")
    metadata["plot_shared_symmetric_clip"] = plot_case(case, output, inputs, **dict(absorbing=records["absorbing"], surface=records["free_surface"], difference=records["difference"]))
    arrays = dict(time_s=np.arange(NT)*DT, receiver_x_m=rec[0,:,0].numpy()*DX,
        receiver_z_m=np.full(61, metadata["receivers"]["z_coordinate_m"]),
        source_xz_m=np.array([400,80]), vp_xz_m_s=vp.numpy(), rho_xz_kg_m3=rho.numpy(),
        source_wavelet=wavelet[0,0].numpy(), **records)
    if case == "born":
        arrays.update(scatter_delta_v_xz_m_s=scatter.numpy(),
            absorbing_background=extras["absorbing"]["background"],
            free_surface_background=extras["free_surface"]["background"])
        arrays["background_difference"] = arrays["free_surface_background"] - arrays["absorbing_background"]
        metadata["background_stats"]={k:stats(arrays[k]) for k in ("absorbing_background", "free_surface_background", "background_difference")}
        plot_case(case, output, inputs, arrays["absorbing_background"], arrays["free_surface_background"], arrays["background_difference"], background=True)
    if case == "vrr":
        arrays.update(rx_xz_per_m=rx.numpy(),rz_xz_per_m=rz.numpy())
    arrays["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(output / f"{case}.npz", **arrays)
    metadata["total_elapsed_s"] = time.perf_counter() - started
    (output / f"{case}.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(f"{case}: complete; difference relative L2={metadata['difference_relative_l2']:.6g}", flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    print(json.dumps(dict(source_commit=SOURCE_COMMIT, torch=torch.__version__,
        starwave=starwave.__file__, device="cpu", backend="torch", cases=args.cases)), flush=True)
    failures=[]
    for case in args.cases:
        try:
            run_case(case,args.output)
        except Exception:
            failure=traceback.format_exc()
            failure_file=args.output / f"{case}_failure_{time.time_ns()}.log"
            failure_file.write_text(failure,encoding="utf-8")
            print(failure,flush=True)
            failures.append(case)
    if failures:
        raise SystemExit("Failed cases (logs preserved): "+", ".join(failures))

if __name__ == "__main__":
    main()
