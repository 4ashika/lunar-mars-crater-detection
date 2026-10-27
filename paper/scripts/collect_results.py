"""Generate IEEE table bodies and a missing-data ledger from experiment CSVs."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
EXPORTS = PAPER / "exports"
ROW_END = r" \\"


def read_csv(name: str) -> list[dict]:
    path = EXPORTS / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def tbd(label: str) -> str:
    return rf"\textcolor{{red}}{{TBD}}"


def val(value, digits=2) -> str:
    if value in (None, "", "nan", "NaN"):
        return tbd("missing result")
    try:
        return f"{float(value):.{digits}f}"
    except (ValueError, TypeError):
        return str(value).replace("_", r"\_")


def latex_escape(value: str) -> str:
    return (str(value).replace("&", r"\&").replace("_", r"\_")
            .replace("%", r"\%").replace("#", r"\#"))


def make_dataset_table(tiles: list[dict]) -> str:
    rows = []
    for planet in ("Moon", "Mars"):
        subset = [r for r in tiles if r["planet"] == planet]
        if not subset:
            rows.append(f"{planet} & {tbd('tiles')} & {tbd('resolution')} & {tbd('area')} & {tbd('source')}" + ROW_END)
            continue
        area = sum(int(r["width_px"]) * int(r["height_px"]) * float(r["m_per_px"])**2 for r in subset) / 1e6
        scales = [float(r["m_per_px"]) for r in subset]
        source = "Chandrayaan-2 OHRC / Roboflow" if planet == "Moon" else "Roboflow mixed-source Mars set"
        resolution = f"{min(scales):.3g}--{max(scales):.3g}"
        if planet == "Mars":
            assumed = sum(r["scale_calibrated"].lower() not in ("true", "1") for r in subset)
            resolution += f" ({assumed}/{len(subset)} assumed)"
        rows.append(f"{planet} & {len(subset)} & {resolution} & {area:.5g} & {source}" + ROW_END)
    return "\n".join(rows)


def make_detection_table(detections: list[dict]) -> str:
    rows=[]
    for planet, mode in (("Moon","v1 Hough"),("Mars","v2 Ellipse")):
        group=[r for r in detections if r["planet"]==planet and r["mode"]==mode]
        size=Counter(r["size_class"] for r in group)
        deg=Counter(r["degradation"] for r in group)
        rows.append(f"{planet} ({mode}) & {len(group)} & {size['Small']} / {size['Medium']} / {size['Large']} & "
                    f"{deg['Fresh']} / {deg['Slightly Degraded']} / {deg['Moderately Degraded']} / {deg['Heavily Degraded']}" + ROW_END)
    return "\n".join(rows)


def make_validation_table(validation: list[dict]) -> tuple[str, list[dict]]:
    rows=[]; summaries=[]
    cases=(("Moon","v1 Hough"),("Mars","v1 Hough"),("Mars","v2 Ellipse"))
    for planet, mode in cases:
        group=[r for r in validation if r["planet"]==planet and r["mode"]==mode]
        tp=sum(int(r["tp"]) for r in group); fp=sum(int(r["fp"]) for r in group); fn=sum(int(r["fn"]) for r in group)
        p=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0
        f1=2*p*rec/(p+rec) if p+rec else 0.0
        iou_num=sum(float(r["mean_iou"])*int(r["tp"]) for r in group)
        miou=iou_num/tp if tp else 0.0
        label=f"{planet} {mode}"
        if not group:
            rows.append(f"{label} & {tbd('precision')} & {tbd('recall')} & {tbd('F1')} & {tbd('mean IoU')}" + ROW_END)
        else:
            rows.append(f"{label} & {p:.3f} & {rec:.3f} & {f1:.3f} & {miou:.3f}" + ROW_END)
        summaries.append({"planet":planet,"mode":mode,"tiles":len(group),"tp":tp,"fp":fp,"fn":fn,"precision":p,"recall":rec,"f1":f1,"mean_iou":miou})
    return "\n".join(rows), summaries


def make_depth_table(detections: list[dict]) -> str:
    rows=[]
    for planet, mode in (("Moon","v1 Hough"),("Mars","v2 Ellipse")):
        group=[r for r in detections if r["planet"]==planet and r["mode"]==mode]
        if not group:
            rows.append(f"{planet} & {tbd('count')} & {tbd('depth')} & {tbd('d/D')} & {tbd('low-confidence share')}" + ROW_END)
            continue
        depths=[float(r["depth_m"]) for r in group if r["depth_m"] not in (None, "")]
        ratios=[float(r["d_D"]) for r in group if r["d_D"] not in (None, "")]
        low=sum(r["depth_low_confidence"].lower() in ("true","1") for r in group)/len(group)
        assumed=any(r["scale_calibrated"].lower() not in ("true","1") for r in group)
        mean_depth=val(sum(depths)/len(depths) if depths else None)
        if planet == "Mars" and assumed:
            mean_depth += r"\textsuperscript{*}"
        rows.append(f"{planet} & {len(group)} & {mean_depth} & "
                    f"{val(sum(ratios)/len(ratios) if ratios else None,3)} & {100*low:.1f}\\%" + ROW_END)
    return "\n".join(rows)


def make_mission_table(missions: list[dict]) -> str:
    rows=[]
    for planet in ("Moon","Mars"):
        group=next((r for r in missions if r["planet"]==planet),None)
        if not group:
            rows.append(f"{planet} & {tbd('landing clearance')} & {tbd('path')} & {tbd('straight')} & {tbd('detour')} & {tbd('crossings')} & {tbd('minimum clearance')}" + ROW_END)
            continue
        scale_note=r"\textsuperscript{*}" if group.get("scale_assumed", "").lower() in ("true","1") else ""
        crossing=f"{group.get('straight_craters_crossed') or tbd('line crossings')} / {group.get('route_craters_crossed') or tbd('route crossings')}"
        rows.append(f"{planet}{scale_note} & {val(group.get('landing_clearance_m'))} & {val(group.get('path_length_m'))} & "
                    f"{val(group.get('straight_length_m'))} & {val(group.get('detour_percent'),1)}\\% & {crossing} & {val(group.get('minimum_rim_clearance_m'))}" + ROW_END)
    return "\n".join(rows)


def write_tables() -> list[str]:
    tiles=read_csv("tile_summary.csv"); detections=read_csv("detections.csv")
    validation=read_csv("validation_tile_metrics.csv"); missions=read_csv("mission_results.csv")
    validation_tex, summaries=make_validation_table(validation)
    tables={"datasetrows":make_dataset_table(tiles),"detectionrows":make_detection_table(detections),
            "validationrows":validation_tex,"depthrows":make_depth_table(detections),
            "missionrows":make_mission_table(missions)}
    (PAPER/"generated_tables.tex").write_text("\n".join(f"\\newcommand{{\\{name}}}{{%\n{body}\n}}\n" for name,body in tables.items()),encoding="utf-8")
    (EXPORTS/"validation_summary.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    missing=[]
    required=(("tile_summary.csv","dataset and detection summaries"),("detections.csv","crater-level catalog"),
              ("validation_tile_metrics.csv","held-out validation"),("mission_results.csv","mission planner outputs"))
    for filename,description in required:
        if not (EXPORTS/filename).exists() or not read_csv(filename): missing.append(f"{description}: experiment export {filename} is absent or empty")
    if not missions or any(not r.get("path_length_m") for r in missions):
        missing.append("mission route metrics: a selected tile has no feasible route to a detected crater rim")
    return missing


if __name__=="__main__":
    missing=write_tables()
    print(f"Generated paper/generated_tables.tex from CSV exports.")
    print("Missing values requiring visible TBDs:")
    for item in missing: print(f"- {item}")
