"""mx2cy's sample run for C5's DV01 reconstruction: MYGA_US_SEC's calls on every point of the SEC table,
on Projection and on Valuation at every close and shock, then the Outer DV01s."""
import sys

sys.setrecursionlimit(10_000)

from DV01_SEC_nomx import mx_model  # noqa: E402

POINTS = 14          # the SEC table's points; the book repeats them
CLOSES = range(3)
SHOCKS = (0, 1, 2)    # down, base, up: Market shifts by (shock - 1) bp
LABELS = ("policy_id", "sex", "tax_status", "mgsv_wd_convention", "renewal_architecture",
          "free_wd_rule", "mva_family", "mva_cap_rule", "scenario_id", "wd_schedule_id")


def sweep(p, kinds, eom, bef_wd):
    p.result_cf()
    p.result_pols()
    p.result_av()
    p.check_av_roll_fwd()
    p.check_pols_roll_fwd()
    p.model_point(), p.point_id()
    for label in LABELS:
        getattr(p, label)()
    p.mgsv_rate_statutory(0.02), p.mgsv_rate_is_compliant(0.02)
    p.mva_ref_term_years(985 / 365.25)
    for t in range(p.proj_len() + 1):
        p.tax_basis_pp(t)
        p.gp_end(t)
        p.av_pp_at(t, eom)
        p.av_at(t, bef_wd)
        for kind in kinds:
            p.claims_over_av(t, kind)


projection = mx_model.Projection
kinds = (projection.DEATH, projection.LAPSE, projection.ANNUITIZATION, projection.MATURITY, projection.ALL)
for idx in range(POINTS):
    sweep(projection[idx], kinds, projection.EOM, projection.BEF_WD)
    for close in CLOSES:
        for shock in SHOCKS:
            v = mx_model.Valuation[idx, close, shock]
            v.pv_net_cf()
            sweep(v, kinds, projection.EOM, projection.BEF_WD)
        mx_model.Outer[idx].dv01(close), mx_model.Outer[idx].reserve(close)
