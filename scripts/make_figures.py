"""Rebuild figures/ from results/ (no GPU, no API). Usage: python scripts/make_figures.py"""
import json, csv, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[1]; RES = ROOT / "results"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

# ---- fig1: core loop, attacker of round r vs defender-so-far
rows = list(csv.DictReader(open(RES / "core/attack_asr_by_round.csv")))
def agg(att):
    out = {}
    for r in rows:
        if r["attacker"] == att: out.setdefault(int(r["round"]), [0, 0]); out[int(r["round"])][0] += int(r["sum"]); out[int(r["round"])][1] += int(r["count"])
    return {k: v[0] / v[1] for k, v in sorted(out.items())}
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for att, lab in (("autodan", "AutoDAN-HGA (n=16)"), ("gcg", "GCG (n=8)")):
    d = agg(att); ax.plot([f"R{k}\nvs D{k-1}" for k in d], list(d.values()), marker="o", label=lab)
ax.set_ylabel("attack success rate (API judge)"); ax.set_ylim(-0.03, 1.05); ax.legend(); ax.set_title("Core loop: ASR per round\n(D2=D3=D1 weights: rounds 3-4 = noise floor)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig1_core_attack_asr_per_round.png", dpi=160); plt.close(fig)

# ---- fig2: held-out evaluation D0 vs D1
rd = list(csv.DictReader(open(RES / "core/summary_by_defender.csv")))
D = {r[list(r.keys())[0]]: r for r in rd}
d0, d1 = D["D_0"], D["D_1"]
keys = [("asr_wei_refusal_suppression", "Wei refusal-supp."), ("asr_wei_style_injection_short", "Wei style-inj."), ("asr_wei_prefix_injection", "Wei prefix-inj."), ("asr_deepinception", "DeepInception"), ("direct_asr", "HarmBench direct")]
fig, ax = plt.subplots(figsize=(6.4, 3.4)); w = 0.38
ax.bar([i - w/2 for i in range(len(keys))], [float(d0[k]) for k, _ in keys], w, label="D0 (base)")
ax.bar([i + w/2 for i in range(len(keys))], [float(d1[k]) for k, _ in keys], w, label="D1 (after RL round 1)")
ax.set_xticks(range(len(keys))); ax.set_xticklabels([l for _, l in keys], fontsize=7); ax.set_ylabel("ASR (n=159 each)"); ax.legend(); ax.set_title("Held-out attack families (never trained on)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig2_heldout_D0_vs_D1.png", dpi=160); plt.close(fig)

# ---- fig3: bonus RL attacker (3 rounds)
def L(p): return list(csv.DictReader(open(RES / p)))
logs = [L(f"bonus/attacker_log_round{b}.csv") for b in (1, 2, 3)]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4)); off = 0
for b, lg in enumerate(logs, 1):
    xs = [off + i for i in range(len(lg))]; ax[0].plot(xs, [float(r["asr"]) for r in lg], color="C0", label="judged ASR (training batch)" if b == 1 else None)
    ax[0].plot(xs, [float(r["valid_rate"]) for r in lg], color="C1", label="valid-prompt rate" if b == 1 else None); off += len(lg)
    if b < 3: ax[0].axvline(off - 0.5, color="grey", ls=":")
ax[0].set_xlabel("attacker GRPO step (round 1 | round 2 | round 3)"); ax[0].legend(fontsize=7); ax[0].set_title("RL attacker training curves", fontsize=9)
c1, c2, c3 = (json.load(open(RES / f"bonus/coevo_round{b}.json")) for b in (1, 2, 3))
lab = ["A0 untrained\nvs D0", "A1\nvs D0", "A2\nvs D1", "A3\nvs D2", "A3\nvs D0 (control)"]
val = [c1["A0_untrained_vs_D0"], c1["A1_trained_vs_D0"], c2["A2_trained_vs_D1"], c3["A3_trained_vs_D2"], c3["A3_trained_vs_D0_control"]]
ax[1].bar(lab, [v["asr"] for v in val], yerr=[[v["asr"] - v["asr_ci"][0] for v in val], [v["asr_ci"][1] - v["asr"] for v in val]], capsize=3)
ax[1].tick_params(axis="x", labelsize=7); ax[1].set_ylabel("ASR (176 prompts, 95% Wilson)"); ax[1].set_title("Bonus: attacker test (D0 = D1 = D2 = D3 weights)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig3_bonus_attacker.png", dpi=160); plt.close(fig)

# ---- fig4: dev-probe snapshots = the safety vs utility trade-off
def probes(p): return json.load(open(RES / p))["probes"]
fig, ax = plt.subplots(figsize=(5.4, 3.6))
for p, name, mk in (("core/rl_probe_round1.json", "core round 1 (accepted at step 40)", "o"), ("bonus/rl_probe_round1.json", "bonus defender round 1 (rejected)", "s"), ("bonus/rl_probe_round2.json", "bonus defender round 2 (rejected)", "^"), ("bonus/rl_probe_round3.json", "bonus defender round 3 (rejected)", "D")):
    pr = probes(p); ax.scatter([v["benign_compliance"] for v in pr.values()], [v["harm_refusal"] for v in pr.values()], marker=mk, label=name)
    for s, v in pr.items(): ax.annotate(f"s{s}", (v["benign_compliance"], v["harm_refusal"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
ax.axvline(0.575 - 0.05, color="grey", ls="--", lw=0.8); ax.text(0.527, 0.955, "benign tolerance", fontsize=6, rotation=90)
ax.set_xlabel("benign compliance on dev probe (n=40, higher = better utility)"); ax.set_ylabel("harmful refusal on dev probe (n=40)"); ax.legend(fontsize=6); ax.set_title("Making the model refuse more costs benign compliance", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig4_dev_probe_safety_vs_utility.png", dpi=160); plt.close(fig)
print("figures written to", FIG)
