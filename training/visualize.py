"""Prediction map visualisation -- TP / FP / FN coloring + confusion matrix.

Reuses the Leaflet template from ``src.visualization`` and adds
prediction-specific layers and a metrics summary.

Usage::

    python -m training.visualize --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd

from src.visualization import _HTML_TEMPLATE, _build_layer_js, _legend_item, _safe_val, save_map

from .config import PipelineConfig, VizConfig, load_config, resolve_paths

log = logging.getLogger(__name__)

PRED_COLORS = {
    "TP": "#2ecc71",
    "FP": "#e74c3c",
    "FN": "#f39c12",
    "TN": "#95a5a6",
    # Unlabeled (no ground truth): UP = predicted farm, UN = predicted not-farm
    "UP": "#3498db",
    "UN": "#d0d3d4",
}

SPLIT_COLORS = {
    "train": "#3498db",
    "val": "#9b59b6",
    "test": "#e67e22",
}

# Human-readable legend labels for prediction classes.
CLASS_LABELS = {
    "TP": "True positive",
    "FP": "False positive",
    "FN": "False negative",
    "TN": "True negative",
    "UP": "Predicted farm (no label)",
    "UN": "Predicted not-farm (no label)",
}

# Multi-class palette + names. Indices match training/rachel_to_candidates.py
# multiclass mode (0..6). Add classes here if the taxonomy grows.
MULTICLASS_COLORS = {
    0: "#95a5a6",   # NotFarm — gray
    1: "#e67e22",   # Poultry: Meat — orange
    2: "#f1c40f",   # Poultry: Eggs — yellow
    3: "#e74c3c",   # Poultry: Unspecified — red
    4: "#3498db",   # Pigs — blue
    5: "#27ae60",   # Cattle — green
    6: "#8e44ad",   # Other — purple
}
MULTICLASS_NAMES = {
    0: "NotFarm",
    1: "Poultry: Meat",
    2: "Poultry: Eggs",
    3: "Poultry: Unspecified",
    4: "Pigs",
    5: "Cattle",
    6: "Other",
}


def _classify_predictions(scored: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a ``pred_class`` column: TP, FP, FN, TN."""
    scored = scored.copy()

    def _row_class(row):
        pred = int(row.get("predicted_label", 0))
        true = int(row.get("true_label", 0))
        if true not in (0, 1):
            # No ground truth (unlabeled) — not part of any confusion cell.
            return "UP" if pred == 1 else "UN"
        if pred == 1 and true == 1:
            return "TP"
        if pred == 1 and true == 0:
            return "FP"
        if pred == 0 and true == 1:
            return "FN"
        return "TN"

    scored["pred_class"] = scored.apply(_row_class, axis=1)
    return scored


def _confusion_counts(scored: gpd.GeoDataFrame) -> dict[str, int]:
    counts = scored["pred_class"].value_counts().to_dict()
    return {k: counts.get(k, 0) for k in ("TP", "FP", "FN", "TN")}


def _split_metrics_html(scored: gpd.GeoDataFrame) -> str:
    """Build per-split (train/val/test/inspected/eval/generalization) metrics rows."""
    if "split" not in scored.columns:
        return ""
    rows = ""
    for split in ("train", "val", "test", "inspected", "eval", "generalization"):
        subset = scored[scored["split"] == split]
        if len(subset) == 0:
            continue
        counts = subset["pred_class"].value_counts().to_dict()
        tp, fp, fn, tn = counts.get("TP", 0), counts.get("FP", 0), counts.get("FN", 0), counts.get("TN", 0)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        color = SPLIT_COLORS.get(split, "#333")
        rows += (
            f'<tr><td style="color:{color};font-weight:bold">{split}</td>'
            f"<td>{prec:.2f}</td><td>{rec:.2f}</td><td>{f1:.2f}</td>"
            f"<td>{tp}</td><td>{fp}</td><td>{fn}</td><td>{tn}</td>"
            f"<td>{len(subset)}</td></tr>"
        )
    if not rows:
        return ""
    return (
        "<br><b>Per-Split</b>"
        '<table style="width:100%;border-collapse:collapse;margin-top:5px;">'
        "<tr><th>Split</th><th>Prec</th><th>Rec</th><th>F1</th>"
        "<th>TP</th><th>FP</th><th>FN</th><th>TN</th><th>N</th></tr>"
        f"{rows}</table>"
    )


def _metrics_html(counts: dict[str, int], per_country: dict, scored: gpd.GeoDataFrame | None = None) -> str:
    """Build an HTML table summarising prediction quality."""
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

    rows = (
        f"<tr><td>Accuracy</td><td>{accuracy:.3f}</td></tr>"
        f"<tr><td>Precision</td><td>{precision:.3f}</td></tr>"
        f"<tr><td>Recall</td><td>{recall:.3f}</td></tr>"
        f"<tr><td>F1</td><td>{f1:.3f}</td></tr>"
        f"<tr><td>TP</td><td>{tp}</td></tr>"
        f"<tr><td>FP</td><td>{fp}</td></tr>"
        f"<tr><td>FN</td><td>{fn}</td></tr>"
        f"<tr><td>TN</td><td>{tn}</td></tr>"
    )
    country_rows = ""
    for country, cm in sorted(per_country.items()):
        c_tp, c_fp, c_fn = cm.get("TP", 0), cm.get("FP", 0), cm.get("FN", 0)
        c_up, c_un = cm.get("UP", 0), cm.get("UN", 0)
        c_prec = c_tp / max(c_tp + c_fp, 1)
        c_rec = c_tp / max(c_tp + c_fn, 1)
        country_rows += (
            f"<tr><td>{country}</td>"
            f"<td>{c_prec:.2f}</td><td>{c_rec:.2f}</td>"
            f"<td>{c_tp}</td><td>{c_fp}</td><td>{c_fn}</td>"
            f"<td>{c_up}</td><td>{c_un}</td></tr>"
        )

    split_html = _split_metrics_html(scored) if scored is not None else ""

    return (
        '<div style="background:white;padding:10px;margin:5px;border-radius:5px;'
        'border:1px solid #999;font:12px/1.4 sans-serif;max-width:450px;max-height:80vh;overflow-y:auto;">'
        "<b>Overall Metrics</b>"
        '<table style="width:100%;border-collapse:collapse;margin-top:5px;">'
        f"{rows}</table>"
        f"{split_html}"
        "<br><b>Per-Country</b>"
        '<table style="width:100%;border-collapse:collapse;margin-top:5px;">'
        "<tr><th>Country</th><th>Prec</th><th>Rec</th>"
        "<th>TP</th><th>FP</th><th>FN</th><th>UP</th><th>UN</th></tr>"
        f"{country_rows}</table></div>"
    )


def _gdf_to_features(subset: gpd.GeoDataFrame) -> list[dict]:
    """Convert a GeoDataFrame subset to a list of GeoJSON feature dicts."""
    return [
        {
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {k: _safe_val(v) for k, v in row.items() if k != "geometry"},
        }
        for _, row in subset.iterrows()
    ]


def _build_pred_layers(scored: gpd.GeoDataFrame, viz_cfg: VizConfig):
    """Build layer JS + legend HTML for each visible prediction class."""
    show_map = {
        "TP": viz_cfg.show_true_positives, "FP": viz_cfg.show_false_positives,
        "FN": viz_cfg.show_false_negatives, "TN": viz_cfg.show_true_negatives,
        # Unlabeled classes have no viz flags — always show when present.
        "UP": True, "UN": True,
    }
    layers_js = ""
    legend_html = ""
    for pred_class in ("TP", "FP", "FN", "TN", "UP", "UN"):
        if not show_map.get(pred_class, False):
            continue
        subset = scored[scored["pred_class"] == pred_class]
        if len(subset) == 0:
            continue
        color = PRED_COLORS[pred_class]
        features = _gdf_to_features(subset)
        layers_js += _build_layer_js(pred_class, features, color)
        legend_html += _legend_item(CLASS_LABELS.get(pred_class, pred_class), color, len(features))
    return layers_js, legend_html


def _per_country_counts(scored: gpd.GeoDataFrame) -> dict[str, dict]:
    if "country" not in scored.columns:
        return {}
    return {
        country: scored[scored["country"] == country]["pred_class"].value_counts().to_dict()
        for country in scored["country"].unique()
    }


def _metrics_panel_js(scored: gpd.GeoDataFrame) -> str:
    counts = _confusion_counts(scored)
    per_country = _per_country_counts(scored)
    metrics_div = _metrics_html(counts, per_country, scored)
    return (
        "var metricsCtrl = L.control({position: 'bottomleft'});\n"
        "metricsCtrl.onAdd = function() {\n"
        "    var div = L.DomUtil.create('div', 'metrics-panel');\n"
        f"    div.innerHTML = {json.dumps(metrics_div)};\n"
        "    return div;\n"
        "};\n"
        "metricsCtrl.addTo(map);\n"
    )


def _build_split_layer_js(split_name: str, subset: gpd.GeoDataFrame, n: int) -> str:
    """Build a single split layer where each point is colored by its pred_class."""
    features = _gdf_to_features(subset)
    fc = {"type": "FeatureCollection", "features": features}
    fc_json = json.dumps(fc)
    colors_json = json.dumps(PRED_COLORS)
    safe = split_name.replace("'", "\\'")
    return f"""
    (function() {{
        var predColors = {colors_json};
        var layer = L.geoJSON({fc_json}, {{
            pointToLayer: function(f, ll) {{
                var c = predColors[f.properties.pred_class] || '#999';
                return L.circleMarker(ll, {{radius: 6, fillColor: c, color: c,
                                            weight: 2, fillOpacity: 0.7}});
            }},
            onEachFeature: function(f, layer) {{
                var p = f.properties, c = f.geometry.coordinates;
                var html = '<b>' + (p.pred_class||'') + '</b> ({safe})<br>';
                if(p.predicted_score != null) html += 'Score: ' + p.predicted_score.toFixed(3) + '<br>';
                if(p.confidence_tier) html += 'Confidence: ' + p.confidence_tier + '<br>';
                if(p.candidate_id) html += 'ID: ' + p.candidate_id + '<br>';
                if(p.country) html += 'Country: ' + p.country + '<br>';
                if(p.source) html += 'Source: ' + p.source + '<br>';
                html += '<a href="https://www.google.com/maps/@' + c[1] + ',' + c[0] + ',500m/data=!3m1!1e3" target="_blank">Open in Google Maps</a><br>';
                layer.bindPopup(html);
            }}
        }}).addTo(map);
        overlays['{safe} ({n})'] = layer;
    }})();
"""


def _build_split_layers(scored: gpd.GeoDataFrame):
    """Build toggleable layers for each split present, points colored by pred_class."""
    if "split" not in scored.columns:
        return ""
    layers_js = ""
    known = ("train", "val", "test", "inspected", "eval", "generalization")
    seen: set[str] = set()
    for split in known:
        subset = scored[scored["split"] == split]
        if len(subset) == 0:
            continue
        layers_js += _build_split_layer_js(split, subset, len(subset))
        seen.add(split)
    # Also surface any other split values present (e.g. "unknown" for inference-only runs).
    for split in scored["split"].dropna().unique():
        if split in seen:
            continue
        subset = scored[scored["split"] == split]
        if len(subset) == 0:
            continue
        layers_js += _build_split_layer_js(str(split), subset, len(subset))
    return layers_js


def _is_multiclass(scored: gpd.GeoDataFrame) -> bool:
    """Detect multi-class run: predicted_label has values outside {0, 1}."""
    if "predicted_label" not in scored.columns:
        return False
    uniq = set(int(x) for x in scored["predicted_label"].dropna().unique())
    return any(v > 1 for v in uniq)


def _name_for_class(cls: int, names_override: list[str] | None = None) -> str:
    if names_override and 0 <= cls < len(names_override):
        return names_override[cls]
    return MULTICLASS_NAMES.get(cls, f"class {cls}")


def _color_for_class(cls: int, colors_override: list[str] | None = None) -> str:
    if colors_override and 0 <= cls < len(colors_override):
        return colors_override[cls]
    return MULTICLASS_COLORS.get(cls, "#999")


def _build_confusion_layer_js(label: str, features: list[dict],
                                fill_color: str, stroke_color: str,
                                radius: int = 6, weight: int = 3,
                                dash: str | None = None) -> str:
    """Render a layer where fill = predicted-class color and stroke = true-class color.

    For correct predictions (fill == stroke) the marker is uniform; for confusions
    the contrasting outline makes the true class instantly readable.
    """
    import json
    fc = {"type": "FeatureCollection", "features": features}
    fc_json = json.dumps(fc)
    safe = label.replace("'", "\\'")
    n = len(features)
    dash_js = f"dashArray: '{dash}'," if dash else ""
    return f"""
    (function() {{
        var layer = L.geoJSON({fc_json}, {{
            pointToLayer: function(f, ll) {{
                return L.circleMarker(ll, {{
                    radius: {radius},
                    fillColor: '{fill_color}',
                    color: '{stroke_color}',
                    weight: {weight},
                    {dash_js}
                    fillOpacity: 0.75,
                    opacity: 1.0
                }});
            }},
            onEachFeature: function(f, layer) {{
                var p = f.properties;
                var html = '<b>{safe}</b><br>';
                if(p.country) html += 'Country: ' + p.country + '<br>';
                if(p.category) html += 'Label: ' + p.category + '<br>';
                if(p.confidence !== undefined) html += 'Confidence: ' + Number(p.confidence).toFixed(3) + '<br>';
                layer.bindPopup(html);
            }}
        }}).addTo(map);
        overlays['{safe} ({n})'] = layer;
    }})();
"""


def _build_multiclass_layers(scored: gpd.GeoDataFrame,
                              names: list[str] | None = None,
                              colors: list[str] | None = None):
    """Confusion-aware layers:

      - One layer per (true_class, predicted_class) cell — fill = pred color,
        stroke = true color. Correct predictions get a thin border; confusions
        get a thick contrasting border so they pop on the map.
      - Unlabeled points get a dashed, lighter layer per predicted class so
        they're visually separable from ground-truth points.
    """
    import numpy as np
    layers_js = ""

    has_true = "true_label" in scored.columns
    if has_true:
        labeled = scored[scored["true_label"].notna() & (scored["true_label"] != -1)]
        unlabeled = scored[scored["true_label"].isna() | (scored["true_label"] == -1)]
    else:
        labeled = scored.iloc[0:0]
        unlabeled = scored

    # Build a layer per (true, pred) cell, ordered so correct cells come first
    # (then confusions, sorted by count descending) — improves layer-control UX.
    if len(labeled) > 0:
        y_true = labeled["true_label"].astype(int)
        y_pred = labeled["predicted_label"].astype(int)
        classes = sorted(set(np.concatenate([y_true.values, y_pred.values]).tolist()))
        cells = []
        for t in classes:
            for p in classes:
                subset = labeled[(y_true == t) & (y_pred == p)]
                if len(subset) == 0:
                    continue
                cells.append((t, p, subset))
        # Correct first (t==p), then confusions by descending size
        cells.sort(key=lambda x: (x[0] != x[1], -len(x[2])))
        for t, p, subset in cells:
            true_name = _name_for_class(t, names)
            pred_name = _name_for_class(p, names)
            true_color = _color_for_class(t, colors)
            pred_color = _color_for_class(p, colors)
            if t == p:
                label = f"✓ {true_name} (correct)"
                layers_js += _build_confusion_layer_js(
                    label, _gdf_to_features(subset),
                    fill_color=pred_color, stroke_color=pred_color,
                    radius=5, weight=1,
                )
            else:
                label = f"✗ true:{true_name} → pred:{pred_name}"
                layers_js += _build_confusion_layer_js(
                    label, _gdf_to_features(subset),
                    fill_color=pred_color, stroke_color=true_color,
                    radius=7, weight=3,
                )

    # Unlabeled — show by predicted class, dashed border to mark "no ground truth"
    if len(unlabeled) > 0:
        for cls in sorted(set(int(x) for x in unlabeled["predicted_label"].dropna().unique())):
            subset = unlabeled[unlabeled["predicted_label"].astype(int) == cls]
            if len(subset) == 0:
                continue
            name = _name_for_class(cls, names)
            color = _color_for_class(cls, colors)
            layers_js += _build_confusion_layer_js(
                f"(unlabeled) pred:{name}", _gdf_to_features(subset),
                fill_color=color, stroke_color="#000",
                radius=4, weight=1, dash="2,3",
            )

    return layers_js


def _multiclass_legend(scored: gpd.GeoDataFrame,
                        names: list[str] | None = None,
                        colors: list[str] | None = None) -> str:
    html = ""
    # Class color swatches (predicted-class counts)
    for cls in sorted(set(int(x) for x in scored["predicted_label"].dropna().unique())):
        n = int((scored["predicted_label"] == cls).sum())
        if n == 0:
            continue
        name = _name_for_class(cls, names)
        color = _color_for_class(cls, colors)
        html += _legend_item(name, color, n)
    # Styling key (use double-quoted HTML attrs — the legend is embedded in a
    # single-quoted JS string by the template, so single quotes would break it).
    html += (
        '<div style="margin-top:6px;padding-top:6px;border-top:1px solid #ddd;'
        'font:11px sans-serif;color:#333">'
        "<b>Marker style:</b><br>"
        '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        'background:#e74c3c;border:1px solid #e74c3c;margin-right:4px"></span>'
        "correct (fill = pred class, thin border)<br>"
        '<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
        'background:#e74c3c;border:3px solid #3498db;margin-right:4px"></span>'
        "<b>confused</b> (fill = pred, thick border = true class)<br>"
        '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        'background:#95a5a6;border:1px dashed #000;margin-right:4px"></span>'
        "unlabeled (dashed border)"
        "</div>"
    )
    return html


def _confusion_matrix_html(y_true, y_pred, classes: list[int],
                            names: list[str] | None = None,
                            colors: list[str] | None = None) -> str:
    """Build a colored K x K confusion matrix table (rows=true, cols=predicted).

    Cells are shaded by row-normalized share: diagonals green, off-diagonals red.
    Each cell shows the raw count and the row percentage.
    """
    import numpy as np
    K = len(classes)
    cm = np.zeros((K, K), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    row_totals = cm.sum(axis=1)
    col_totals = cm.sum(axis=0)

    def _cell(i: int, j: int) -> str:
        n = int(cm[i, j])
        rt = int(row_totals[i]) or 1
        share = n / rt
        if i == j:
            # green diagonal, opacity scales with row recall
            bg = f"rgba(46,204,113,{share:.2f})"
        elif n == 0:
            bg = "rgba(255,255,255,0)"
        else:
            # red off-diagonal, opacity scales with row share of confusion
            bg = f"rgba(231,76,60,{min(share * 1.2, 1.0):.2f})"
        pct = (share * 100) if rt else 0.0
        return (
            f"<td style='background:{bg};text-align:center;padding:3px 6px;"
            f"border:1px solid #ddd;min-width:46px'>"
            f"<div style='font-weight:bold'>{n}</div>"
            f"<div style='font-size:10px;color:#444'>{pct:.0f}%</div></td>"
        )

    # Header row: predicted class names
    header = "<tr><th style='padding:3px 6px;border:1px solid #ddd;background:#f6f6f6'>true \\ pred</th>"
    for c in classes:
        name = _name_for_class(c, names)
        color = _color_for_class(c, colors)
        header += (
            f"<th style='padding:3px 6px;border:1px solid #ddd;background:#f6f6f6;"
            f"color:{color};font-weight:bold'>{name}</th>"
        )
    header += "<th style='padding:3px 6px;border:1px solid #ddd;background:#f6f6f6'>total</th></tr>"

    body = ""
    for i, c in enumerate(classes):
        name = _name_for_class(c, names)
        color = _color_for_class(c, colors)
        body += (
            f"<tr><th style='padding:3px 6px;border:1px solid #ddd;background:#f6f6f6;"
            f"color:{color};font-weight:bold;text-align:left'>{name}</th>"
        )
        for j in range(K):
            body += _cell(i, j)
        body += (
            f"<td style='text-align:center;padding:3px 6px;border:1px solid #ddd;"
            f"background:#f6f6f6'>{int(row_totals[i])}</td></tr>"
        )
    # Column-total footer
    footer = "<tr><th style='padding:3px 6px;border:1px solid #ddd;background:#f6f6f6;text-align:left'>pred total</th>"
    for j in range(K):
        footer += (
            f"<td style='text-align:center;padding:3px 6px;border:1px solid #ddd;"
            f"background:#f6f6f6'>{int(col_totals[j])}</td>"
        )
    footer += (
        f"<td style='text-align:center;padding:3px 6px;border:1px solid #ddd;"
        f"background:#f6f6f6'><b>{int(cm.sum())}</b></td></tr>"
    )

    return (
        f"<div style='margin-top:8px'>"
        f"<b>Confusion matrix</b> "
        f"<span style='color:#666'>(rows = true label, cols = predicted; cells colored by row share)</span>"
        f"<table style='border-collapse:collapse;margin-top:4px;font-size:11px'>"
        + header + body + footer
        + f"</table></div>"
    )


def _multiclass_metrics_panel_js(scored: gpd.GeoDataFrame,
                                   names: list[str] | None = None,
                                   colors: list[str] | None = None) -> str:
    """Per-class precision/recall/F1, macro avg, and confusion matrix.

    Computed on the **val split** only — held-out data the model was scored
    against for early stopping. Avoids the inflated numbers that would result
    from including the training set the model already memorized. Falls back to
    all labeled rows if no val split is available (legacy parquets).
    """
    labeled = scored[scored["true_label"].notna() & (scored["true_label"] != -1)]
    panel_split = "val"
    if "split" in labeled.columns and (labeled["split"] == "val").any():
        labeled = labeled[labeled["split"] == "val"]
    else:
        panel_split = "all labeled (no val split available)"
    if len(labeled) == 0:
        return ""
    import numpy as np
    y_true = labeled["true_label"].astype(int).values
    y_pred = labeled["predicted_label"].astype(int).values
    classes = sorted(set(np.concatenate([y_true, y_pred]).tolist()))
    rows = []
    macro_p = macro_r = macro_f = 0.0
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-8)
        macro_p += p; macro_r += r; macro_f += f
        name = _name_for_class(c, names)
        color = _color_for_class(c, colors)
        rows.append(
            f"<tr><td style='color:{color};font-weight:bold'>{name}</td>"
            f"<td>{p:.3f}</td><td>{r:.3f}</td><td>{f:.3f}</td>"
            f"<td>{tp}</td><td>{fp}</td><td>{fn}</td></tr>"
        )
    k = len(classes)
    cm_html = _confusion_matrix_html(y_true, y_pred, classes, names, colors)
    table = (
        f"<div style='font:12px sans-serif;background:white;padding:8px;border:1px solid #ccc;"
        f"max-height:80vh;overflow:auto'>"
        f"<b>Per-class metrics</b> "
        f"<span style='color:#666'>(split: {panel_split}, n={len(labeled):,})</span>"
        f"<table style='border-collapse:collapse;margin-top:4px'>"
        f"<tr><th>Class</th><th>Prec</th><th>Rec</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th></tr>"
        + "".join(rows)
        + f"<tr style='border-top:1px solid #999'><td><b>Macro</b></td>"
        f"<td>{macro_p/k:.3f}</td><td>{macro_r/k:.3f}</td><td>{macro_f/k:.3f}</td>"
        f"<td colspan=3></td></tr>"
        f"</table>"
        + cm_html
        + f"</div>"
    )
    # Inject as a Leaflet info control
    safe = table.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "")
    return (
        f"\n    var metricsCtrl = L.control({{position:'topright'}});\n"
        f"    metricsCtrl.onAdd = function() {{ var d = L.DomUtil.create('div'); d.innerHTML='{safe}'; return d; }};\n"
        f"    metricsCtrl.addTo(map);\n"
    )


def generate_prediction_map(
    scored: gpd.GeoDataFrame,
    viz_cfg: VizConfig,
    title: str = "Farm Detection Predictions",
    class_names: list[str] | None = None,
    class_colors: list[str] | None = None,
) -> str:
    """Generate an interactive Leaflet HTML map coloured by prediction outcome."""
    multi = _is_multiclass(scored)
    if multi:
        # Multi-class: one layer per predicted class, per-class metrics panel.
        layers_js = _build_multiclass_layers(scored, class_names, class_colors)
        legend_html = _multiclass_legend(scored, class_names, class_colors)
        layers_js += _multiclass_metrics_panel_js(scored, class_names, class_colors)
    else:
        scored = _classify_predictions(scored)
        # Split layers: train/val/test toggleable, each point colored by TP/FP/FN/TN
        layers_js = _build_split_layers(scored)
        # Legend shows prediction class colors (not split colors)
        legend_html = ""
        for pred_class in ("TP", "FP", "FN", "TN", "UP", "UN"):
            n = int((scored["pred_class"] == pred_class).sum())
            if n > 0:
                legend_html += _legend_item(
                    CLASS_LABELS.get(pred_class, pred_class), PRED_COLORS[pred_class], n
                )
        layers_js += _metrics_panel_js(scored)

    center_lat = float(scored["lat"].mean()) if len(scored) > 0 else 15.0
    center_lon = float(scored["lng"].mean()) if len(scored) > 0 else 100.0

    return _HTML_TEMPLATE.format(
        title=title, center_lat=center_lat, center_lon=center_lon,
        zoom=6, layers_js=layers_js, legend_html=legend_html,
    )


def visualize(cfg: PipelineConfig) -> Path:
    """Load scored candidates and generate the prediction map."""
    # Config-specific output directory
    scored_dir = Path(cfg.patches.output_dir).parent / "output" / cfg._config_stem
    scored_path = scored_dir / "scored_candidates.parquet"

    if not scored_path.exists():
        # Legacy fallback
        scored_path = Path(cfg.patches.output_dir) / "scored_candidates.parquet"
    if not scored_path.exists():
        log.error("No scored_candidates.parquet found -- run inference first")
        raise FileNotFoundError(scored_path)

    scored = gpd.read_parquet(scored_path)
    log.info("Loaded %d scored candidates", len(scored))

    html = generate_prediction_map(
        scored, cfg.visualization,
        class_names=getattr(cfg.model, "class_names", None),
        class_colors=getattr(cfg.model, "class_colors", None),
    )

    output_dir = Path(cfg.visualization.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / "prediction_map.html"
    save_map(html, map_path)

    # Dated sibling: preserves previous runs and makes the newest one obvious.
    import shutil
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    dated_path = output_dir / f"prediction_map_{stamp}.html"
    shutil.copy2(map_path, dated_path)
    log.info("Wrote dated copy: %s", dated_path.name)

    return map_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise predictions")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = resolve_paths(load_config(args.config))
    path = visualize(cfg)
    print(f"Map saved to: {path}")


if __name__ == "__main__":
    main()
