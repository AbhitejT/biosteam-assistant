"""Turn a chat session into a downloadable, self-contained report.

A report bundles the conversation, any built-process results (streams,
economics, direct carbon, verification), and the provenance metadata so a
non-programmer can save and share a reproducible record of an analysis.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def _built_section(built: dict[str, Any]) -> list[str]:
    results = built.get("results") or {}
    lines: list[str] = []
    lines.append(f"### Built process: {results.get('name', 'custom process')}")
    lines.append("")

    def stream_table(title: str, rows: list[dict]) -> None:
        if not rows:
            return
        lines.append(f"**{title}**")
        lines.append("")
        lines.append("| Stream | Phase | T (K) | kg/hr | kmol/hr |")
        lines.append("|--------|-------|-------|-------|---------|")
        for s in rows:
            lines.append(
                f"| {s['name']} | {s['phase']} | {s['T_K']} | "
                f"{s['F_mass_kg_hr']:.2f} | {s['F_mol_kmol_hr']:.3f} |"
            )
        lines.append("")

    stream_table("Feeds", results.get("feeds", []))
    stream_table("Products", results.get("products", []))
    stream_table("Intermediates", results.get("intermediates", []))

    units = results.get("units", [])
    if units:
        lines.append("**Unit operations**")
        lines.append("")
        lines.append("| Unit | Type | Installed cost (USD) |")
        lines.append("|------|------|----------------------|")
        for u in units:
            cost = u.get("installed_cost_usd")
            lines.append(
                f"| {u['id']} | {u['type']} | "
                f"{_fmt_money(cost) if cost is not None else 'n/a'} |"
            )
        lines.append("")
        total = results.get("total_installed_equipment_cost_usd")
        if total is not None:
            lines.append(f"Total installed equipment cost: **{_fmt_money(total)}**")
            lines.append("")

    econ = results.get("economics")
    if econ:
        a = econ["assumptions"]
        lines.append("**Economics (simplified TEA estimate)**")
        lines.append("")
        lines.append(
            f"- Minimum selling price ({econ['product']}): "
            f"**${econ['min_selling_price_usd_per_kg']:.3f}/kg**"
        )
        lines.append(
            f"- Total capital investment: "
            f"{_fmt_money(econ['total_capital_investment_usd'])}"
        )
        lines.append(
            f"- Material cost: {_fmt_money(econ['material_cost_usd_per_yr'])}/yr; "
            f"fixed operating cost: "
            f"{_fmt_money(econ['fixed_operating_cost_usd_per_yr'])}/yr"
        )
        lines.append(
            f"- Assumptions: IRR {a['IRR']:.0%}, {int(a['plant_years'])}-yr life, "
            f"{int(a['operating_days'])} operating days/yr, Lang factor "
            f"{a['lang_factor']}, FOC {a['FOC_over_FCI']:.0%} of FCI"
        )
        lines.append("")

    carbon = results.get("carbon")
    if carbon:
        lines.append("**Direct greenhouse-gas emissions**")
        lines.append("")
        ghg = carbon.get("direct_ghg_kg_per_hr") or {}
        if ghg:
            for gas, kg in ghg.items():
                lines.append(f"- {gas}: {kg:.2f} kg/hr")
            lines.append(f"- Total: **{carbon['co2e_kg_per_hr']:.2f} kg CO2e/hr**")
            if "co2e_kg_per_kg_product" in carbon:
                lines.append(
                    f"- Intensity: {carbon['co2e_kg_per_kg_product']:.3f} "
                    f"kg CO2e per kg {carbon.get('product', 'product')}"
                )
        else:
            lines.append("- No direct greenhouse gases in outlet streams.")
        lines.append(f"- Scope: {carbon['scope']} ({carbon['gwp_basis']}).")
        lines.append("")

    verification = built.get("verification")
    if verification:
        lines.extend(_verification_lines(verification))
    return lines


def _verification_lines(report: dict[str, Any]) -> list[str]:
    label = {"pass": "PASS", "warn": "REVIEW RECOMMENDED", "fail": "FAIL"}
    lines = [f"**Validation: {label.get(report['overall'], report['overall'])}**", ""]
    for c in report.get("checks", []):
        mark = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}.get(c["status"], "?")
        lines.append(f"- [{mark}] {c['name']}: {c['detail']}")
    lines.append("")
    return lines


def build_markdown_report(
    history: list[dict[str, Any]],
    model: str,
    log_name: str | None = None,
    biosteam_version: str | None = None,
) -> str:
    """Render a full session report as Markdown."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# BioSTEAM Assistant — Session Report",
        "",
        f"- Generated: {now}",
        f"- Assistant model: `{model}`",
    ]
    if biosteam_version:
        lines.append(f"- BioSTEAM version: `{biosteam_version}`")
    if log_name:
        lines.append(f"- Provenance log: `{log_name}`")
    lines.append("")
    lines.append(
        "> Quantitative results come from BioSTEAM simulations via an "
        "allowlisted tool layer; every tool call is recorded in the provenance "
        "log. Built-process selling prices are simplified-TEA estimates and "
        "carbon figures are direct gate-to-gate emissions only."
    )
    lines.append("")
    lines.append("## Conversation")
    lines.append("")

    for msg in history:
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"### {role}")
        lines.append("")
        lines.append(msg.get("content", "").strip() or "_(no text)_")
        lines.append("")
        if msg.get("built"):
            lines.extend(_built_section(msg["built"]))
        if msg.get("verification") and not msg.get("built"):
            lines.extend(_verification_lines(msg["verification"]))
        if msg.get("sources"):
            seen = set()
            uniq = []
            for s in msg["sources"]:
                key = (s["title"], s["source"])
                if key not in seen:
                    seen.add(key)
                    uniq.append(s)
            if uniq:
                lines.append("**Sources**")
                lines.append("")
                for s in uniq:
                    lines.append(f"- {s['title']} — _{s['source']}_")
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_json_report(
    history: list[dict[str, Any]],
    model: str,
    log_name: str | None = None,
    biosteam_version: str | None = None,
) -> str:
    """Render the full session (including structured results) as JSON."""
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "assistant_model": model,
        "biosteam_version": biosteam_version,
        "provenance_log": log_name,
        "messages": history,
    }
    return json.dumps(payload, indent=2, default=str)
