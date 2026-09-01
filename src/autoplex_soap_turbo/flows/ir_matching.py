"""Equilibrate a liquid, predict its infrared spectrum, then bias it toward an
experimental one -- by each of the four estimators, so they can be compared.

This is not a training workflow. Nothing is fitted: the potential and the dipole
model are given, and what the flow produces is a spectrum and a measurement of
how far four different biasing schemes move it.

    potential -> equilibrate -> check -> predict
              -> [reference arms] -> [biased arms] -> compare -> analyse

WHY EACH STAGE EXISTS

``potential``
    Compose the energy model and the dipole model into the single ``.gap``
    turboGAP reads, then evaluate one frame with it and read the result back.

    THE ENERGY MODEL OF A LIQUID HAS NO DIPOLE IN IT. They are separate GAPs
    fitted to different targets; a block flagged ``dipole_model = .true.``
    contributes to the dipole and to nothing else, so concatenating them leaves
    the energy surface exactly as the energy model had it. Each model keeps its
    own subdirectory under ``gap_files/``, because both converters name their
    outputs after the descriptor type and flattening them has one silently
    overwrite the other.

    The one-frame check is here rather than further down because all three ways
    this flow can measure the wrong thing are visible in a single frame and in
    none of thermo.log: a potential that did not load (every energy term zero),
    a composed file with no dipole model in it (no ``dipole=`` tag, and every
    spectrum downstream is the autocorrelation of a constant), and a dispersion
    term that was configured and then not applied (``vdw energy: 0.00000000``).

``equilibrate``
    MD from a starting configuration to a liquid. Optionally NPT first, so the
    density is the potential's own rather than the one the box was built at.

    Then ONE FURTHER STAGE PER ARM THERMOSTAT, under that thermostat, whose last
    frame is what those arms start from. A bath only equilibrates the modes it
    couples to, and a GLE built for an IR run couples to almost nothing inside
    the fitted band -- so a GLE arm handed a Bussi frame spends picoseconds
    draining vibrational energy the GLE will not touch. Measured on the
    64-molecule trial: both GLE arms sat at 375-380 K against a 300 K target for
    their whole 3 ps while the Bussi arms held 301.6 K. Paid once per
    thermostat here rather than once per arm.

``check``
    DoSPT (github.com/mcaroba/dospt), and a molecular-composition census.

    THIS STAGE IS THE POINT OF THE FLOW HAVING A CHECK AT ALL. A thermostatted
    run reaches its target temperature in a few hundred femtoseconds and holds
    it whatever the structure is doing, so temperature is nearly useless as an
    equilibration test. Two things that are not:

    - the CENSUS. Count how many oxygens carry exactly two hydrogens. A water
      potential that is subtly wrong produces a proton-disordered network --
      stretched O-H, protons shared between oxygens, hydronium and hydroxide in
      quantity -- and every downstream number is then about that object rather
      than about water. Measured on one real model: median O-H 1.085 A against
      water's 0.97, 43% of protons within 0.15 A of equidistant between two
      oxygens, and only 44 of 64 oxygens in intact molecules. Nothing in
      thermo.log showed it.
    - TRANSPORT. The diffusion coefficient is the zero-frequency limit of the
      translational density of states and it keeps drifting while the
      hydrogen-bond network relaxes. The test is that it has stopped drifting
      between consecutive windows, not that it equals any particular value.

``predict``
    ``turbogap predict`` with ``do_ir``: the dipole model is evaluated on every
    frame of a trajectory already on disk and the spectrum comes out at the end.
    The frame interval is read from the ``time=`` tags rather than assumed,
    because it is ``write_xyz * md_step`` and getting it wrong rescales the
    whole wavenumber axis while looking entirely plausible.

``reference`` and ``biased`` arms
    One unbiased arm per matching method, and one biased arm per rung of that
    method's ``bias_ladder`` -- all of identical length, all starting from the
    same equilibrated structure. What the bias did is then the difference
    between a biased arm and its own reference and nothing else.

    THE LADDER EXISTS BECAUSE ONE STRENGTH ANSWERS NOTHING. If a single biased
    arm lands lower than its reference, that is one draw of a fluctuating
    quantity; if the response is monotone in the strength across three rungs, it
    is a response. Rungs share their reference -- same estimator, same bath, so
    one measurement sizes them all -- while methods never do.

    The three methods, which are three different answers to "how do you get a
    spectrum out of a running trajectory and a force back into it":

    ``naive``  ``ir_bias_mode = acf``, ``ir_acf_mode = block``. Keep a rolling
               buffer of the dipole, autocorrelate it, window it, transform it,
               compare with experiment, differentiate the whole chain and apply
               the force -- every step.
    ``xl``     ``ir_bias_mode = xl`` (``mad_ir_xl.f90``). No autocorrelation at
               all: a bank of damped resonators driven by the dipole, whose
               standing amplitude IS the spectrum.
    ``gle``    the same block ACF bias, under ``thermostat = gle``. Not a fourth
               estimator -- a different bath. A white-noise thermostat broadens
               every line by 5308.8/tau_t cm^-1, which at tau_t = 100 fs is
               ~53 cm^-1 and wider than the features being fitted; the GLE's
               friction is frequency-dependent and can be kept out of the band.

    THE UNBIASED ARM IS NOT SHARED BETWEEN METHODS. ``exp_energy_scales``
    multiplies a dissimilarity, and each estimator's dissimilarity is in its own
    arbitrary units, so the only way to push them with the same ENERGY is to
    size each from an unbiased arm measured by the same meter. ``gle`` needs its
    own for the further reason that it changes the dynamics, not just the meter.

``compare``
    Every arm also writes ``ir_spectrum.dat``, the ordinary windowed ACF on the
    experimental grid, measured identically in all of them. That is the only
    common observable there is: each estimator's own spectrum is the thing it
    biases, and those are different objects.

``analyse``
    The evidence behind that number, in ``05_analysis``: where in the spectrum
    the mismatch lives, whether the bias changed the SHAPE of it or only its
    size, whether the change cleared the arm's own scatter, and whether every
    thermostat reached its target. Written as columns, as generated gnuplot
    scripts, and as two reports -- ``report.md`` and ``equilibration_report.md``.

WHAT THE FLOW DOES NOT DO. It cannot make a model absorb where it has no mode.
MAD moves intensity between the modes a potential already has; if the model's
O-H stretch is 1300 cm^-1 from the experimental band, no coupling strength
reaches it, and the honest outcome of this flow is that measurement.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jobflow import Flow, job

from autoplex_soap_turbo.config import WorkerSettings
from autoplex_soap_turbo.flows.common import apply_worker
from autoplex_soap_turbo.turbogap.ir import ESTIMATORS, arm_pairs, expand_arms


def _worker(config: dict, key: str) -> WorkerSettings | None:
    """The YAML gives a plain mapping; apply_worker wants the dataclass."""
    spec = config.get(key)
    if not spec:
        return None
    known = {f for f in WorkerSettings.__dataclass_fields__}
    unknown = set(spec) - known
    if unknown:
        raise ValueError(
            f"{key}: unknown key(s) {sorted(unknown)}; "
            f"WorkerSettings takes {sorted(known)}"
        )
    return WorkerSettings(**spec)

logger = logging.getLogger(__name__)


# Re-exported from turbogap.ir, which has no jobflow dependency, so that a
# driver running the stages in-process can ask for the same list without
# importing a scheduler. See ESTIMATORS there.
__all__ = ["ESTIMATORS", "arm_pairs", "expand_arms", "ir_matching_flow"]


@job
def build_potential(config: dict) -> dict:
    """Compose energy + dipole into one .gap, and prove it on one frame."""
    from autoplex_soap_turbo.turbogap.ir import run_build_potential

    return run_build_potential(config)


@job
def equilibrate(config: dict, pot: dict) -> dict:
    """Melt if asked, then equilibrate, then sample finely for the DoS.

    THE MELT IS NOT DECORATION when the starting configuration came from a
    different potential: its structure is that potential's, and carrying it
    into the production run samples someone else's minimum relaxing for as long
    as the run lasts.
    """
    from autoplex_soap_turbo.turbogap.ir import run_equilibration

    return run_equilibration(config, pot)


@job
def check_equilibration(equil: dict, config: dict) -> dict:
    """Composition census, then DoSPT drift across consecutive windows."""
    from autoplex_soap_turbo.turbogap.ir import run_dospt_check

    result = run_dospt_check(equil, config)
    if config.get("ir", {}).get("stop_if_not_equilibrated", True):
        if not result["census"]["ok"]:
            raise RuntimeError(
                "the configuration is not molecular water: "
                + result["census"]["summary"]
                + ". Every spectrum computed from it would be about a different "
                "object. Set ir.stop_if_not_equilibrated = false to continue anyway."
            )
        if not result["settled"]:
            raise RuntimeError(
                "the structural observables are still drifting between windows: "
                + result["summary"]
                + ". Equilibrate for longer."
            )
    return result


@job
def predict_ir(equil: dict, config: dict, pot: dict) -> dict:
    """`turbogap predict` + do_ir over the sampled trajectory."""
    from autoplex_soap_turbo.turbogap.ir import run_ir_prediction

    return run_ir_prediction(equil, config, pot)


@job
def ir_arm(equil: dict, config: dict, arm: dict, pot: dict,
           reference: dict | None = None) -> dict:
    """One MD arm: unbiased reference, or biased and sized from one.

    `reference` is the completed reference arm of the same estimator family;
    None for the reference arms themselves.
    """
    from autoplex_soap_turbo.turbogap.ir import run_ir_arm

    return run_ir_arm(equil, config, arm, reference, pot)


@job
def compare_arms(arms: list[dict], prediction: dict, config: dict) -> dict:
    """Cross-estimator comparison on the one common observable."""
    from autoplex_soap_turbo.turbogap.ir import compare_ir_arms

    return compare_ir_arms(arms, prediction, config)


@job
def analyse(arms: list[dict], equil: dict, check: dict, prediction: dict,
            config: dict) -> dict:
    """Tables, gnuplot scripts and the two written reports.

    Separate from ``compare`` because they answer different questions. That one
    produces the single number the flow's output carries; this one produces the
    evidence -- where the mismatch lives, whether the bias changed the SHAPE of
    it, whether the change cleared the arm's own noise, and whether every
    thermostat actually reached its target. All of it as columns a gnuplot
    script reads, plus the scripts themselves, so the plots come out of the run
    rather than out of whoever happened to analyse it.
    """
    from autoplex_soap_turbo.turbogap.ir_analysis import run_analysis

    return run_analysis(arms, equil, check, prediction, config)


def ir_matching_flow(config: dict) -> Flow:
    """Build the flow. Static: every stage is known before anything runs, so the
    whole thing can be inspected with `jf flow` and restarted per job."""
    ir = config.get("ir", {})

    jobs = []

    pot = build_potential(config)
    pot.name = "compose potential (energy + dipole)"
    apply_worker(pot, _worker(config, "md_worker"))
    jobs.append(pot)

    eq = equilibrate(config, pot.output)
    eq.name = "equilibrate"
    apply_worker(eq, _worker(config, "md_worker"))
    jobs.append(eq)

    chk = check_equilibration(eq.output, config)
    chk.name = "check equilibration (DoSPT + census)"
    apply_worker(chk, _worker(config, "analysis_worker"))
    jobs.append(chk)

    pred = predict_ir(eq.output, config, pot.output)
    pred.name = "predict IR from the trajectory"
    apply_worker(pred, _worker(config, "md_worker"))
    jobs.append(pred)

    if not ir.get("bias", True):
        out = compare_arms([], pred.output, config)
        out.name = "report"
        apply_worker(out, _worker(config, "analysis_worker"))
        jobs.append(out)
        return Flow(jobs, output=out.output, name=config.get("name", "ir_matching"))

    # ONE UNBIASED ARM PER MATCHING METHOD, and one biased arm per strength on
    # that method's ladder -- all of identical length, all from the same
    # equilibrated structure. What the bias did is then the difference between a
    # biased arm and its own reference and nothing else.
    #
    # The unbiased arm is NOT shared between methods. exp_energy_scales
    # multiplies a dissimilarity, each estimator's dissimilarity is in its own
    # arbitrary units, and the only way to push them with the same ENERGY is to
    # size each from an unbiased arm measured by the same meter. A method that
    # also changes the thermostat -- gle -- needs its own for the further reason
    # that it changes the dynamics and not only the meter.
    #
    # It IS shared down a ladder, and that is the point of having one: every
    # strength on a ladder is the same estimator under the same bath, so they
    # are all sized from the same measurement and differ only in how hard they
    # push. Running a separate reference per strength would cost one full arm
    # each to reproduce a number already in hand -- and on the 64-molecule trial
    # two references that differed only in estimator turned out byte-identical.
    ref_specs, bias_specs = expand_arms(ir)

    ref_jobs = {}
    for spec in ref_specs:
        r = ir_arm(eq.output, config, spec, pot.output)
        r.name = f"unbiased arm: {spec['method']} ({spec['thermostat']})"
        apply_worker(r, _worker(config, "md_worker"))
        ref_jobs[spec["name"]] = r
        jobs.append(r)

    bias_jobs = []
    for spec in bias_specs:
        r = ref_jobs[spec["reference"]]
        b = ir_arm(eq.output, config, spec, pot.output, r.output)
        rungs = sum(1 for s in bias_specs if s["reference"] == spec["reference"])
        b.name = (f"biased arm: {spec['method']} at {spec['bias_target_kt']:g} kT"
                  if rungs > 1 else
                  f"biased arm: {spec['method']} ({spec['label']})")
        apply_worker(b, _worker(config, "md_worker"))
        bias_jobs.append(b)
        jobs.append(b)

    arm_outputs = ([r.output for r in ref_jobs.values()]
                   + [b.output for b in bias_jobs])

    cmp_job = compare_arms(arm_outputs, pred.output, config)
    cmp_job.name = "compare estimators"
    apply_worker(cmp_job, _worker(config, "analysis_worker"))
    jobs.append(cmp_job)

    ana = analyse(arm_outputs, eq.output, chk.output, pred.output, config)
    ana.name = "analyse: tables, gnuplot scripts, reports"
    apply_worker(ana, _worker(config, "analysis_worker"))
    jobs.append(ana)

    return Flow(jobs, output=ana.output, name=config.get("name", "ir_matching"))


def load(path) -> dict:
    """Read the IR settings file. Plain YAML -- this workflow has no dataclass."""
    import yaml  # noqa: PLC0415

    return yaml.safe_load(Path(path).read_text())


def describe(flow, config: dict) -> None:
    """Print what the run will do, and whether it can measure anything.

    The arithmetic below is the reason this is a hook rather than a generic job
    listing. A buffered estimator reports nothing until it has filled, so the
    window that actually carries the bias is the arm minus the buffer -- and if
    that is not several times the buffer, the scatter of the dissimilarity
    swamps whatever the bias did. On the first 64-molecule run it was 0.36x and
    none of the three methods could be told apart from its own reference. It is
    printed before submission rather than discovered afterwards.
    """
    from autoplex_soap_turbo.turbogap.ir import (  # noqa: PLC0415
        arm_pairs,
        ensemble_frames,
        thermalise_stages,
    )

    print(f"\nflow: {flow.name}")
    for i, j in enumerate(flow.jobs, 1):
        print(f"  {i:2d}. {j.name}")

    ir = config.get("ir", {})
    dt = config["turbogap"].get("md_step", 0.5)
    print("\n  methods    :")
    for m in arm_pairs(ir):
        ladder = m.get("bias_ladder") or [m.get("bias_target_kt",
                                                ir.get("bias_target_kt", 0.5))]
        print(f"     {m['name']:<8} {m['thermostat']:<6} "
              f"bias_mode={m['bias_mode']}"
              + (f", acf_mode={m['acf_mode']}" if m.get("acf_mode") else "")
              + f"   bias at {', '.join(f'{x:g}' for x in ladder)} kT")

    # THE ARITHMETIC THAT DECIDES WHETHER THE RUN CAN MEASURE ANYTHING AT ALL,
    # printed before submission rather than discovered afterwards. A buffered
    # estimator reports nothing until it has filled, so the biased window is the
    # arm minus the buffer -- and if that is not several times the buffer, the
    # scatter of the dissimilarity swamps whatever the bias did. On the first
    # 64-molecule run it was 0.36x, and none of the three methods could be
    # distinguished from its own reference.
    n = ir.get("arm_steps", 20000)
    buf = ensemble_frames(ir, dt) * max(1, ir.get("stride", 1))
    biased = n - buf
    print(f"\n  arm length : {n} steps of {dt} fs = {n * dt / 1000:.1f} ps, "
          "unbiased and biased alike")
    print(f"  buffer     : {buf:.0f} steps = {buf * dt / 1000:.2f} ps  "
          f"(lag_factor {ir.get('lag_factor', 2)}, resolution "
          f"{ir.get('resolution', 35.0)} cm^-1)")
    print(f"  biased for : {biased:.0f} steps = {biased * dt / 1000:.2f} ps "
          f"= {biased / buf:.1f}x the buffer"
          + ("   <-- THIN: a bias effect would be inside the noise"
             if biased < 2 * buf else ""))
    print("               xl reports from ~ir_xl_tau_mem instead, so its "
          "biased window is longer")
    for key, label in (("melt_steps", "melt"), ("equil_steps", "equilibrate"),
                       ("thermalise_steps", "thermalise"),
                       ("dospt_steps", "DoS sample")):
        if ir.get(key):
            print(f"  {label:<11}: {ir[key]} steps = {ir[key] * dt / 1000:.2f} ps")
    extra = thermalise_stages(ir)
    if extra:
        print(f"  hand-off   : own equilibrated frame for {extra}")
    print(f"  workdir    : {config.get('workdir')}")
    print(f"  experiment : {config.get('experiment')}")

