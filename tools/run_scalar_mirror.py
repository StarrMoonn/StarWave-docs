#!/usr/bin/env python3
"""Validate StarWave scalar2D zero-halo boundary against a negative image source.

Run with an ordinary installed StarWave import (no path injection/kernel edits):
    python run_scalar_mirror.py --output-dir results
Uses the public Torch eager API on CPU. Numerical pressure is not rescaled to
match the reference. This is a homogeneous, planar, 2D forward test, not a
native/GPU/3D/gradient/PML-wide validation.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import starwave
from starwave.time_sampling import make_time_sampling_plan

COMMIT = "200621cc31c0ebeebaa01ffaa3591c0ebd7e0adb"
C = 1500.0
DT = 0.0005
NT = 481
FREQ = 25.0
PEAK_TIME = 0.06
SOURCE_Z = 60.0
RECEIVER_Z = 30.0
RECEIVER_X = np.arange(-120.0, 120.1, 30.0)
HALF_EXTENT = 300.0
PML = 10
BUFFER = 5
GATE = (0.0, 0.24)
LATE_GATE = (0.11, 0.24)


def idx(metres: float, h: float) -> int:
    i = round(metres / h)
    assert abs(i*h-metres) < 1e-10, (metres, h)
    return i


def metric(numerical: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict:
    a, b = numerical[:, mask].astype(np.float64), reference[:, mask].astype(np.float64)
    difference = a-b
    denominator = np.linalg.norm(b)
    peak = np.max(np.abs(b))
    return dict(relative_l2=float(np.linalg.norm(difference)/denominator),
                max_abs_difference=float(np.max(np.abs(difference))),
                max_abs_reference=float(peak),
                max_difference_over_reference_peak=float(np.max(np.abs(difference))/peak),
                sample_count=int(difference.size))


def run_case(h: float, accuracy: int, extent: float = HALF_EXTENT) -> tuple[dict, dict]:
    started = time.perf_counter()
    n = idx(extent, h)
    nx, nz_half, nz_full = 2*n+1, n, 2*n+1
    assert max(nx, nz_full) < 160
    half_model = torch.full((nx, nz_half), C, dtype=torch.float32)
    full_model = torch.full((nx, nz_full), C, dtype=torch.float32)
    # Wall is z=0. Half model index j represents z=(j+1)h.
    # Full model index k represents z=(k-n)h and contains a live wall plane.
    half_sources = torch.tensor([[[n, idx(SOURCE_Z, h)-1]]], dtype=torch.int64)
    full_sources = torch.tensor([[[n, n+idx(SOURCE_Z, h)],
                                  [n, n-idx(SOURCE_Z, h)]]], dtype=torch.int64)
    half_receivers = torch.tensor([[[n+idx(x,h), idx(RECEIVER_Z,h)-1]
                                    for x in RECEIVER_X]], dtype=torch.int64)
    full_receivers = torch.tensor([[[n+idx(x,h), n+idx(RECEIVER_Z,h)]
                                    for x in RECEIVER_X]], dtype=torch.int64)
    wall_receivers = torch.tensor([[[n+idx(x,h), n] for x in RECEIVER_X]], dtype=torch.int64)
    # Q(t) has unit peak, interpreted as a 2D spatially integrated source in Pa.
    # The sampled RHS f=Q/h^2 is Pa/m^2; injection is -c^2 dt_internal^2 f.
    # No empirical amplitude calibration or trace normalization is applied.
    time_s = np.arange(NT, dtype=np.float64)*DT
    phase = np.pi*FREQ*(time_s-PEAK_TIME)
    ricker = (1-2*phase**2)*np.exp(-phase**2)
    forcing = torch.as_tensor(ricker/h**2, dtype=torch.float32)
    half_plan = make_time_sampling_plan(half_model,h,DT,NT,max_vel=C)
    full_plan = make_time_sampling_plan(full_model,h,DT,NT,max_vel=C)
    assert asdict(half_plan) == asdict(full_plan)
    assert half_plan.step_ratio == 1
    options = dict(grid_spacing=h, dt=DT, accuracy=accuracy, pml_freq=FREQ,
                   boundary_buffer=BUFFER, max_vel=C, backend="torch",
                   execution="eager", memory="full")
    with torch.no_grad():
        numerical = starwave.scalar(half_model, source_amplitudes=forcing[None,None],
                source_locations=half_sources, receiver_locations=half_receivers,
                pml_width=[PML,PML,0,PML], **options)[-1][0].numpy()
        full_record = starwave.scalar(full_model,
                source_amplitudes=torch.stack((forcing,-forcing))[None],
                source_locations=full_sources,
                receiver_locations=torch.cat((full_receivers,wall_receivers),dim=1),
                pml_width=PML, **options)[-1][0].numpy()
    reference, wall = full_record[:len(RECEIVER_X)],full_record[len(RECEIVER_X):]
    arrays = dict(numerical=numerical, reference=reference,
                  difference=numerical-reference, wall_pressure=wall,
                  source_forcing_pa_per_m2=forcing.numpy())
    finite = all(bool(np.isfinite(a).all()) for a in arrays.values())
    assert finite
    mask = (time_s >= GATE[0]) & (time_s <= GATE[1])
    late = (time_s >= LATE_GATE[0]) & (time_s <= LATE_GATE[1])
    wall_peak = float(np.max(np.abs(wall)))
    report = dict(accuracy=accuracy, grid_spacing_m=h, outer_extent_m=extent,
                  model_half_shape_xz=[nx,nz_half], model_full_shape_xz=[nx,nz_full],
                  padded_half_shape_xz=[nx+2*(PML+BUFFER+accuracy//2),
                                       nz_half+PML+BUFFER+accuracy],
                  padded_full_shape_xz=[nx+2*(PML+BUFFER+accuracy//2),
                                       nz_full+2*(PML+BUFFER+accuracy//2)],
                  half_source_indices_xz=half_sources.tolist(),
                  full_source_indices_xz=full_sources.tolist(),
                  half_receiver_indices_xz=half_receivers.tolist(),
                  full_receiver_indices_xz=full_receivers.tolist(),
                  clock=asdict(half_plan), gate_s=list(GATE), late_gate_s=list(LATE_GATE),
                  gather=metric(numerical,reference,mask),
                  late_gather=metric(numerical,reference,late),
                  wall_peak_pa=wall_peak,
                  wall_peak_over_reference_peak=wall_peak/float(np.max(np.abs(reference))),
                  finite=finite, elapsed_s=time.perf_counter()-started)
    print(json.dumps({k:report[k] for k in ("accuracy","grid_spacing_m","gather","late_gather","finite","elapsed_s")}),flush=True)
    return report, arrays


def plot_results(path: Path, report: dict, arrays: dict):
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10, "axes.titlesize":11,
                         "axes.spines.top":False, "axes.spines.right":False})
    fig = plt.figure(figsize=(13.2,8.8),constrained_layout=True)
    gs = fig.add_gridspec(2,3,height_ratios=(1,1.06))
    ax = fig.add_subplot(gs[0,0])
    ax.axhline(0,color="black",lw=1.1)
    ax.axhspan(0,145,facecolor="#e5f3fa",alpha=.8)
    ax.plot(0,SOURCE_Z,"*",color="#c43b32",ms=15,label="real +Q(t)")
    ax.plot(0,-SOURCE_Z,"*",color="#2678b2",ms=15,label="image -Q(t)")
    ax.scatter(RECEIVER_X,np.full_like(RECEIVER_X,RECEIVER_Z),marker="v",s=28,color="#333333",label="receivers")
    ax.annotate("wall z = 0",(80,0),xytext=(85,-12),ha="center")
    ax.annotate("first half-model row: z = h",(-145,10),fontsize=9)
    ax.set(xlim=(-150,150),ylim=(145,-100),xlabel="x relative to source (m)",ylabel="z (m)",title="A  Geometry: mirror about the zero plane")
    ax.legend(loc="lower right",fontsize=8,frameon=False)
    ax.grid(alpha=.12)
    ax = fig.add_subplot(gs[0,1])
    for order,color in [(2,"#333333"),(4,"#d46a28"),(8,"#2678b2")]:
        cases = sorted((c for c in report["cases"] if c["accuracy"]==order),key=lambda c:c["grid_spacing_m"])
        y = np.array([c["gather"]["relative_l2"] for c in cases])
        ax.plot([c["grid_spacing_m"] for c in cases],100*y,"o-",color=color,label=f"order {order}")
        if order==2:
            ax.annotate("exact equality in FP32",(7.5,0),xytext=(6,1.1),fontsize=8,
                        arrowprops=dict(arrowstyle="->",lw=.7))
    ax.set(xlabel="grid spacing h (m)",ylabel="gather relative L2 error (%)",title="B  Same physical geometry, refined mesh",ylim=(-.45,10))
    ax.set_xticks([5,7.5,10]);ax.grid(alpha=.2);ax.legend(frameon=False)
    ax = fig.add_subplot(gs[0,2])
    t = arrays["time_s"]
    key = case_key(8,5)
    ax.plot(t,arrays[key+"_reference"][4],"--",color="#333333",lw=1.8,label="negative-image reference")
    ax.plot(t,arrays[key+"_numerical"][4],color="#2678b2",lw=1.1,label="zero-halo boundary")
    ax.plot(t,arrays[key+"_difference"][4],color="#d46a28",lw=1.1,label="difference")
    for arrival,label in [(PEAK_TIME+abs(SOURCE_Z-RECEIVER_Z)/C,"direct"),
                          (PEAK_TIME+(SOURCE_Z+RECEIVER_Z)/C,"reflected")]:
        ax.axvline(arrival,color="#aaaaaa",ls=":",lw=.7)
        ax.text(arrival+.003,.035,label,fontsize=8,rotation=90,va="center")
    ax.set(xlabel="time (s)",ylabel="pressure (Pa; unit integrated source)",title="C  Central receiver: order 8, h=5 m",xlim=GATE)
    ax.legend(fontsize=8,frameon=False,loc="lower right");ax.grid(alpha=.2)
    key=case_key(8,5)
    a,b,d=[arrays[key+"_"+name] for name in ("numerical","reference","difference")]
    common = float(max(np.max(np.abs(a)),np.max(np.abs(b))))
    extent=[RECEIVER_X[0]-15,RECEIVER_X[-1]+15,t[-1],t[0]]
    for k,(name,data) in enumerate([( "D  Order 8, h=5 m: zero-halo gather",a),
                                   ( "E  Negative-image reference",b),
                                   ( "F  Difference (same color limits)",d)]):
        ax=fig.add_subplot(gs[1,k]);im=ax.imshow(data.T,origin="upper",aspect="auto",extent=extent,
                cmap="seismic",vmin=-common,vmax=common,interpolation="nearest")
        ax.set(xlabel="receiver x relative to source (m)",ylabel="time (s)",title=name)
        ax.axhline(LATE_GATE[0],color="#777777",ls=":",lw=.7)
        if k==2:
            err=next(c["gather"]["relative_l2"] for c in report["cases"] if c["accuracy"]==8 and c["grid_spacing_m"]==5)
            ax.text(.03,.93,f"relative L2 = {err:.3%}",transform=ax.transAxes,fontsize=9,
                    bbox=dict(facecolor="white",alpha=.8,edgecolor="none"))
    fig.colorbar(im,ax=fig.axes[-3:],label="pressure (Pa)",shrink=.8)
    fig.suptitle("Scalar2D zero halo vs. negative image source\nOrder 2 has the matching discrete wall; higher-order zero halos are not odd-mirror closure",fontsize=14)
    fig.savefig(path,dpi=180)
    plt.close(fig)


def case_key(order: int,h: float) -> str:
    return f"order{order}_h{h:g}".replace(".","p")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent/"results")
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    reports=[]; arrays={"time_s":np.arange(NT,dtype=np.float64)*DT,"receiver_x_m":RECEIVER_X,
                        "receiver_z_m":np.full_like(RECEIVER_X,RECEIVER_Z)}
    for order in (2,4,8):
        for h in (10.,7.5,5.):
            report,data=run_case(h,order);reports.append(report)
            arrays.update({case_key(order,h)+"_"+k:v for k,v in data.items()})
    # Move all outer physical faces out by 60 m while keeping source, receivers,
    # wall and time unchanged. This independently checks the selected time gate.
    extended_report,extended=run_case(7.5,8,extent=360.)
    base=case_key(8,7.5)
    mask=arrays["time_s"] <= GATE[1]
    control={name:metric(arrays[base+"_"+name],extended[name],mask)
             for name in ("numerical","reference")}
    control.update(grid_spacing_m=7.5,accuracy=8,base_extent_m=300.,extended_extent_m=360.,
                   extended_case=extended_report)
    arrays.update({"boundary_control_"+k:v for k,v in extended.items()})
    selected=case_key(8,5)
    for name in ("numerical","reference","difference"):
        arrays[name]=arrays[selected+"_"+name]
    report=dict(source_commit=COMMIT,starwave_file=str(Path(starwave.__file__).resolve()),
        torch_version=torch.__version__,backend="torch",execution="eager",device="cpu",dtype="float32",threads=2,
        model=dict(velocity_m_s=C,x_extent_m=[-HALF_EXTENT,HALF_EXTENT],half_z_extent_m="[h,300]",full_z_extent_m=[-HALF_EXTENT,HALF_EXTENT],wall_z_m=0),
        source=dict(real_xz_m=[0,SOURCE_Z],image_xz_m=[0,-SOURCE_Z],wavelet="Q(t)=(1-2*a^2)*exp(-a^2), a=pi*f*(t-t_peak)",
                    frequency_hz=FREQ,peak_time_s=PEAK_TIME,integrated_peak_pa=1.,
                    sample_forcing="f_real=+Q(t)/h^2; f_image=-Q(t)/h^2, in Pa/m^2",
                    discrete_injection="-v_at_source^2 * internal_dt^2 * f",normalization="none; raw pressure comparison"),
        receivers=dict(x_m=RECEIVER_X.tolist(),z_m=RECEIVER_Z),
        boundary=dict(public_face_order=["x-low","x-high","z-low","z-high"],half_pml_width=[PML,PML,0,PML],full_pml_width=[PML]*4,boundary_buffer=BUFFER,
            half_index_to_coordinate="x=-300+i*h; z=(j+1)*h",full_index_to_coordinate="x=-300+i*h; z=-300+k*h",
            interpretation="order 2 is Dirichlet on z=0; orders >2 clamp every ghost cell to zero, not an odd extension"),
        gate_s=list(GATE),late_gate_s=list(LATE_GATE),
        outer_boundary_gate_justification="Shortest geometric source-to-outer-face-to-receiver distance is >= 480 m (side) or 510 m (bottom), at least 0.32 s at 1500 m/s even from source onset. Gate ends at 0.24 s; a +60 m extent control checks dispersive/pre-arrival leakage.",
        cases=reports,boundary_distance_control=control,
        selected_npz_alias=selected,
        limitations=["Homogeneous horizontal planar scalar2D forward validation only.",
            "Reference is the same interior finite-difference scheme in doubled full space, not an independent continuum solver.",
            "Order-2 exact matching validates geometry/sign/discrete symmetry, not continuum accuracy.",
            "No native CPU, CUDA, 3D, heterogeneous media, arbitrary faces, gradients, or general PML acceptance claim."])
    assert all(c["finite"] for c in reports)
    assert all(c["gather"]["relative_l2"] < 2e-5 for c in reports if c["accuracy"]==2)
    assert all(control[k]["relative_l2"] < 2e-5 for k in ("numerical","reference"))
    for order in (4,8):
        errs=[c["gather"]["relative_l2"] for c in reports if c["accuracy"]==order]
        assert errs[2] < errs[1] < errs[0], (order,errs)
    np.savez_compressed(args.output_dir/"scalar_mirror.npz",**arrays)
    (args.output_dir/"scalar_mirror.json").write_text(json.dumps(report,indent=2)+"\n")
    plot_results(args.output_dir/"scalar_mirror.png",report,arrays)
    print("Saved",args.output_dir/"scalar_mirror.png",flush=True)
    print("Boundary distance control",json.dumps(control),flush=True)

if __name__=="__main__":
    main()
