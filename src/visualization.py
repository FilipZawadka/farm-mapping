"""Interactive Leaflet map generation -- reusable for any country / method combination."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import geopandas as gpd

log = logging.getLogger(__name__)

# Consistent colour scheme across all maps
METHOD_COLORS: dict[str, str] = {
    "NDBI": "#e41a1c",
    "GoogleOpenBuildings": "#377eb8",
    "MetalRoof": "#00cccc",
    "SAR": "#984ea3",
    "DynamicWorld": "#4daf4a",
    "GLCMTexture": "#ff7f00",
    "EdgeDetection": "#a65628",
    "KnownFarms": "#222222",
}


def _features_by_source(gdf: gpd.GeoDataFrame) -> dict[str, list[dict]]:
    """Group a GeoDataFrame into GeoJSON feature lists keyed by 'source'."""
    groups: dict[str, list[dict]] = {}
    if len(gdf) == 0:
        return groups
    for _, row in gdf.iterrows():
        src = row.get("source", "Unknown")
        feat = {
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {k: _safe_val(v) for k, v in row.items() if k != "geometry"},
        }
        groups.setdefault(src, []).append(feat)
    return groups


def _safe_val(v):
    """Convert numpy / pandas types to JSON-safe Python types."""
    import numpy as np

    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _build_layer_js(method_name: str, features: list[dict], color: str) -> str:
    """Return JS code that adds one GeoJSON layer to ``map``."""
    fc = {"type": "FeatureCollection", "features": features}
    fc_json = json.dumps(fc)
    safe = method_name.replace("'", "\\'")
    n = len(features)
    return f"""
    (function() {{
        var layer = L.geoJSON({fc_json}, {{
            style: function() {{ return {{color: '{color}', weight: 2, fillOpacity: 0.35}}; }},
            pointToLayer: function(f, ll) {{
                return L.circleMarker(ll, {{radius: 6, fillColor: '{color}', color: '{color}',
                                            weight: 2, fillOpacity: 0.7}});
            }},
            onEachFeature: function(f, layer) {{
                var p = f.properties, html = '<b>{safe}</b><br>';
                if(p.area_m2) html += 'Area: ' + Math.round(p.area_m2) + ' m&sup2;<br>';
                if(p.length_m) html += 'Length: ' + Math.round(p.length_m) + ' m<br>';
                if(p.width_m) html += 'Width: ' + Math.round(p.width_m) + ' m<br>';
                if(p.name) html += 'Name: ' + p.name + '<br>';
                if(p.species) html += 'Species: ' + p.species + '<br>';
                layer.bindPopup(html);
            }}
        }}).addTo(map);
        overlays['{safe} ({n})'] = layer;
    }})();
"""


def generate_country_map(
    candidates: gpd.GeoDataFrame,
    known_farms: gpd.GeoDataFrame,
    center: tuple[float, float],
    zoom: int = 8,
    title: str = "Farm Detection",
) -> str:
    """Generate a complete HTML page with an interactive Leaflet map.

    Returns the HTML as a string.
    """
    layers_js = ""
    legend_html = ""

    # Known farms layer
    if len(known_farms) > 0:
        kf_feats = [
            {
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": {k: _safe_val(v) for k, v in row.items() if k != "geometry"},
            }
            for _, row in known_farms.iterrows()
        ]
        color = METHOD_COLORS.get("KnownFarms", "#222222")
        layers_js += _build_layer_js("KnownFarms", kf_feats, color)
        legend_html += _legend_item("Known Farms", color, len(kf_feats))

    # Candidate layers by method
    groups = _features_by_source(candidates)
    for method_name, feats in sorted(groups.items()):
        color = METHOD_COLORS.get(method_name, "#888888")
        layers_js += _build_layer_js(method_name, feats, color)
        legend_html += _legend_item(method_name, color, len(feats))

    return _HTML_TEMPLATE.format(
        title=title,
        center_lat=center[0],
        center_lon=center[1],
        zoom=zoom,
        layers_js=layers_js,
        legend_html=legend_html,
    )


def generate_global_map(
    country_results: dict[str, tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]],
    title: str = "Global Farm Detection",
) -> str:
    """Combine multiple countries into a single map.

    *country_results* maps country name to ``(candidates_gdf, known_farms_gdf)``.
    """
    all_candidates = []
    all_known = []
    for country, (cands, known) in country_results.items():
        if len(cands) > 0:
            cands = cands.copy()
            cands["country"] = country
            all_candidates.append(cands)
        if len(known) > 0:
            known = known.copy()
            all_known.append(known)

    merged_cands = gpd.GeoDataFrame(
        __import__("pandas").concat(all_candidates, ignore_index=True), crs="EPSG:4326",
    ) if all_candidates else gpd.GeoDataFrame()
    merged_known = gpd.GeoDataFrame(
        __import__("pandas").concat(all_known, ignore_index=True), crs="EPSG:4326",
    ) if all_known else gpd.GeoDataFrame()

    return generate_country_map(
        merged_cands, merged_known, center=(20, 0), zoom=3, title=title,
    )


def save_map(html: str, output_path: str | Path) -> Path:
    """Write HTML map string to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    log.info("Saved map: %s", path)
    return path


def _legend_item(label: str, color: str, count: int) -> str:
    return (
        f'<div style="margin:2px 0">'
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{color};margin-right:6px;border:1px solid #333;'
        f'vertical-align:middle;"></span>{label} ({count})</div>'
    )


_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{{margin:0;padding:0;}} #map{{height:100vh;width:100vw;}}</style>
</head>
<body>
<div id="map"></div>
<script>
var map = L.map('map').setView([{center_lat}, {center_lon}], {zoom});
L.tileLayer('https://mt1.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}', {{
    maxZoom: 20, attribution: 'Google Satellite'
}}).addTo(map);

var overlays = {{}};

{layers_js}

L.control.layers(null, overlays, {{collapsed: false}}).addTo(map);

var legend = L.control({{position: 'bottomright'}});
legend.onAdd = function() {{
    var div = L.DomUtil.create('div', 'legend');
    div.style.cssText = 'background:white;padding:10px;border-radius:5px;border:1px solid #999;font:13px/1.5 sans-serif;max-height:300px;overflow:auto;';
    div.innerHTML = '<b>{title}</b><br>{legend_html}';
    return div;
}};
legend.addTo(map);
</script>
</body>
</html>"""
