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
}

SPLIT_COLORS = {
    "train": "#3498db",
    "val": "#9b59b6",
    "test": "#e67e22",
}


def _classify_predictions(scored: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a ``pred_class`` column: TP, FP, FN, TN."""
    scored = scored.copy()

    def _row_class(row):
        pred = int(row.get("predicted_label", 0))
        true = int(row.get("true_label", 0))
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
    """Build per-split (train/val/test) metrics rows."""
    if "split" not in scored.columns:
        return ""
    rows = ""
    for split in ("train", "val", "test"):
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
        c_prec = c_tp / max(c_tp + c_fp, 1)
        c_rec = c_tp / max(c_tp + c_fn, 1)
        country_rows += (
            f"<tr><td>{country}</td>"
            f"<td>{c_prec:.2f}</td><td>{c_rec:.2f}</td>"
            f"<td>{c_tp}</td><td>{c_fp}</td><td>{c_fn}</td></tr>"
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
        "<tr><th>Country</th><th>Prec</th><th>Rec</th><th>TP</th><th>FP</th><th>FN</th></tr>"
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
    }
    layers_js = ""
    legend_html = ""
    for pred_class in ("TP", "FP", "FN", "TN"):
        if not show_map.get(pred_class, False):
            continue
        subset = scored[scored["pred_class"] == pred_class]
        if len(subset) == 0:
            continue
        color = PRED_COLORS[pred_class]
        features = _gdf_to_features(subset)
        layers_js += _build_layer_js(pred_class, features, color)
        legend_html += _legend_item(pred_class, color, len(features))
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
    """Build toggleable layers for train/val/test, points colored by pred_class."""
    if "split" not in scored.columns:
        return ""
    layers_js = ""
    for split in ("train", "val", "test"):
        subset = scored[scored["split"] == split]
        if len(subset) == 0:
            continue
        layers_js += _build_split_layer_js(split, subset, len(subset))
    return layers_js


def generate_prediction_map(
    scored: gpd.GeoDataFrame,
    viz_cfg: VizConfig,
    title: str = "Farm Detection Predictions",
) -> str:
    """Generate an interactive Leaflet HTML map coloured by prediction outcome."""
    scored = _classify_predictions(scored)

    # Split layers: train/val/test toggleable, each point colored by TP/FP/FN/TN
    layers_js = _build_split_layers(scored)

    # Legend shows prediction class colors (not split colors)
    legend_html = ""
    for pred_class in ("TP", "FP", "FN", "TN"):
        n = int((scored["pred_class"] == pred_class).sum())
        if n > 0:
            legend_html += _legend_item(pred_class, PRED_COLORS[pred_class], n)

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

    html = generate_prediction_map(scored, cfg.visualization)

    output_dir = Path(cfg.visualization.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / "prediction_map.html"
    save_map(html, map_path)

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
