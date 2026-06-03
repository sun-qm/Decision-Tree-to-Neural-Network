"""SVG visualization for decision trees and generated neural networks."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Sequence

from dt_to_nn.network import ComputedNeuron, Connection, NeuralDecisionNetwork
from dt_to_nn.tree import DecisionNode, Leaf, TreeNode, predict_one, required_n_features


@dataclass(frozen=True)
class SvgConfig:
    tree_width: int = 620
    network_width: int = 900
    margin: int = 56
    tree_level_gap: int = 120
    network_level_gap: int = 180
    neuron_gap: int = 46
    node_radius: int = 28
    font_family: str = "Arial, Helvetica, sans-serif"


@dataclass(frozen=True)
class _TreeDrawNode:
    node: TreeNode
    key: str
    depth: int
    x: float
    y: float


def save_tree_svg(
    tree: TreeNode,
    path: str | Path,
    *,
    sample: Sequence[float] | None = None,
    title: str = "Decision Tree Structure",
    config: SvgConfig | None = None,
) -> Path:
    """Render a decision tree to an SVG file."""

    config = config or SvgConfig()
    svg = render_tree_svg(tree, sample=sample, title=title, config=config)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output


def save_network_svg(
    network: NeuralDecisionNetwork,
    path: str | Path,
    *,
    n_features: int | None = None,
    sample: Sequence[float] | None = None,
    title: str = "Neural Network Architecture",
    config: SvgConfig | None = None,
) -> Path:
    """Render a generated neural network to an SVG file."""

    config = config or SvgConfig()
    svg = render_network_svg(
        network,
        n_features=n_features,
        sample=sample,
        title=title,
        config=config,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output


def save_tree_and_network_svg(
    tree: TreeNode,
    network: NeuralDecisionNetwork,
    path: str | Path,
    *,
    sample: Sequence[float] | None = None,
    title: str = "Decision Tree to Neural Network",
    config: SvgConfig | None = None,
) -> Path:
    """Render tree and network side by side to an SVG file."""

    config = config or SvgConfig()
    tree_svg = _render_tree_body(tree, sample=sample, config=config, x_offset=0, y_offset=0)
    n_features = max(required_n_features(tree), len(sample or []))
    network_svg, network_height = _render_network_body(
        network,
        n_features=n_features,
        sample=sample,
        config=config,
        x_offset=config.tree_width,
        y_offset=0,
    )
    tree_height = _tree_canvas_height(tree, config)
    width = config.tree_width + config.network_width
    height = max(tree_height, network_height) + 40
    body = [
        _svg_header(width, height, config),
        _text(width / 2, 30, title, 22, "middle", weight="700"),
        _text(config.tree_width / 2, 64, "Decision Tree Structure", 18, "middle", weight="700"),
        _text(
            config.tree_width + config.network_width / 2,
            64,
            "Neural Network Architecture",
            18,
            "middle",
            weight="700",
        ),
        tree_svg,
        network_svg,
        "</svg>",
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(body), encoding="utf-8")
    return output


def render_tree_svg(
    tree: TreeNode,
    *,
    sample: Sequence[float] | None = None,
    title: str = "Decision Tree Structure",
    config: SvgConfig | None = None,
) -> str:
    config = config or SvgConfig()
    height = _tree_canvas_height(tree, config) + 40
    return "\n".join(
        [
            _svg_header(config.tree_width, height, config),
            _text(config.tree_width / 2, 34, title, 18, "middle", weight="700"),
            _render_tree_body(tree, sample=sample, config=config, x_offset=0, y_offset=0),
            "</svg>",
        ]
    )


def render_network_svg(
    network: NeuralDecisionNetwork,
    *,
    n_features: int | None = None,
    sample: Sequence[float] | None = None,
    title: str = "Neural Network Architecture",
    config: SvgConfig | None = None,
) -> str:
    config = config or SvgConfig()
    if n_features is None:
        n_features = _network_required_features(network)
    body, height = _render_network_body(
        network,
        n_features=n_features,
        sample=sample,
        config=config,
        x_offset=0,
        y_offset=0,
    )
    return "\n".join(
        [
            _svg_header(config.network_width, height + 40, config),
            _text(config.network_width / 2, 34, title, 18, "middle", weight="700"),
            body,
            "</svg>",
        ]
    )


def _render_tree_body(
    tree: TreeNode,
    *,
    sample: Sequence[float] | None,
    config: SvgConfig,
    x_offset: float,
    y_offset: float,
) -> str:
    leaves: list[str] = []
    positions: dict[str, _TreeDrawNode] = {}
    edges: list[tuple[str, str, str]] = []
    counter = 0

    def assign(node: TreeNode, depth: int, parent: str | None = None, branch: str = "") -> str:
        nonlocal counter
        key = f"t{counter}"
        counter += 1
        if parent is not None:
            edges.append((parent, key, branch))
        if isinstance(node, Leaf):
            leaves.append(key)
            x = len(leaves) * (config.tree_width - 2 * config.margin) / (_leaf_count(tree) + 1)
        else:
            left = assign(node.true_child, depth + 1, key, "True")
            right = assign(node.false_child, depth + 1, key, "False")
            x = (positions[left].x + positions[right].x) / 2.0
        y = 95 + depth * config.tree_level_gap
        positions[key] = _TreeDrawNode(node=node, key=key, depth=depth, x=x, y=y)
        return key

    assign(tree, 0)
    active_path = set(_active_tree_keys(tree, positions, sample)) if sample is not None else set()

    parts: list[str] = []
    for source, target, branch in edges:
        src = positions[source]
        dst = positions[target]
        active = source in active_path and target in active_path
        parts.append(
            _line(
                x_offset + src.x,
                y_offset + src.y + config.node_radius,
                x_offset + dst.x,
                y_offset + dst.y - config.node_radius,
                "#111827" if active else "#222222",
                3 if active else 2,
            )
        )
        label_x = x_offset + (src.x + dst.x) / 2
        label_y = y_offset + (src.y + dst.y) / 2 - 8
        color = "#159447" if branch == "True" else "#d12d35"
        parts.append(_text(label_x, label_y, branch, 12, "middle", color=color, weight="700"))

    for item in sorted(positions.values(), key=lambda node: node.depth):
        fill = "#9de688" if isinstance(item.node, Leaf) else "#bde8f2"
        stroke = "#111827" if item.key in active_path else "#111111"
        parts.append(_circle(x_offset + item.x, y_offset + item.y, config.node_radius, fill, stroke, 2.5))
        if isinstance(item.node, Leaf):
            lines = [f"Class", str(item.node.label)]
        else:
            op = ">="
            lines = [f"x_{item.node.feature_index}", f"{op} {item.node.threshold:.2f}"]
        parts.extend(_multiline_text(x_offset + item.x, y_offset + item.y - 5, lines, 11))
    return "\n".join(parts)


def _render_network_body(
    network: NeuralDecisionNetwork,
    *,
    n_features: int,
    sample: Sequence[float] | None,
    config: SvgConfig,
    x_offset: float,
    y_offset: float,
) -> tuple[str, int]:
    layers = _network_layers(network, n_features)
    max_nodes = max(len(items) for items in layers.values())
    width = config.network_width
    height = max(380, 125 + max_nodes * config.neuron_gap)
    layer_ids = sorted(layers)
    x_positions = {
        layer: x_offset + config.margin + i * (width - 2 * config.margin) / max(1, len(layer_ids) - 1)
        for i, layer in enumerate(layer_ids)
    }
    y_positions: dict[str, tuple[float, float]] = {}
    for layer in layer_ids:
        names = layers[layer]
        start_y = y_offset + 112 + (max_nodes - len(names)) * config.neuron_gap / 2
        for idx, name in enumerate(names):
            y_positions[name] = (x_positions[layer], start_y + idx * config.neuron_gap)

    values = _network_values(network, sample)
    parts: list[str] = []
    for layer in layer_ids:
        title = "Input\nLayer" if layer == 0 else ("Output\nLayer" if layer == max(layer_ids) else f"Hidden\nLayer {layer}")
        x = x_positions[layer]
        parts.extend(_multiline_text(x, y_offset + 78, title.split("\n"), 13, weight="700"))

    for source, target, weight in _network_edges(network, n_features):
        if source not in y_positions or target not in y_positions:
            continue
        x1, y1 = y_positions[source]
        x2, y2 = y_positions[target]
        color = "#2d8cd3" if weight >= 0 else "#d9534f"
        stroke_width = max(1.4, min(5.0, abs(weight) * 2.0))
        parts.append(_line(x1 + 17, y1, x2 - 17, y2, color, stroke_width, opacity=0.78))
        if abs(weight) > 0:
            parts.append(
                _edge_label((x1 + x2) / 2, (y1 + y2) / 2, f"{weight:.2f}")
            )

    for layer in layer_ids:
        for name in layers[layer]:
            x, y = y_positions[name]
            fill = _network_node_color(name, layer, max(layer_ids))
            parts.append(_circle(x, y, 18, fill, "#1f2937", 1.6))
            label = _short_neuron_name(name)
            val = values.get(name)
            lines = [label]
            if val is not None:
                lines.append(f"{val:.2f}")
            elif layer == 0 and sample is not None:
                index = int(name[1:])
                if index < len(sample):
                    lines.append(f"{sample[index]:.2f}")
            parts.extend(_multiline_text(x, y - 4, lines, 9))

    parts.append(_legend(x_offset + width - 190, y_offset + 100))
    return "\n".join(parts), int(height)


def _network_layers(network: NeuralDecisionNetwork, n_features: int) -> dict[int, list[str]]:
    layers = {0: [f"x{i}" for i in range(n_features)]}
    for layer, names in network.layer_map().items():
        layers[layer] = names
    return layers


def _network_edges(network: NeuralDecisionNetwork, n_features: int) -> list[tuple[str, str, float]]:
    edges: list[tuple[str, str, float]] = []
    for neuron in network.condition_neurons.values():
        sign = 1.0 if neuron.branch == "true" else -1.0
        edges.append((f"x{neuron.feature_index}", neuron.name, sign))
    for neuron in network.computed_neurons.values():
        for conn in neuron.incoming:
            edges.append((conn.source, neuron.name, conn.weight))
    for output in network.output_neurons.values():
        for conn in output.incoming:
            edges.append((conn.source, output.name, conn.weight))
    return edges


def _network_values(network: NeuralDecisionNetwork, sample: Sequence[float] | None) -> dict[str, float]:
    if sample is None:
        values = {}
    else:
        values = network.values(sample)
        for label, value in network.outputs(sample).items():
            values[network.output_neurons[label].name] = value
    return values


def _active_tree_keys(
    tree: TreeNode,
    positions: dict[str, _TreeDrawNode],
    sample: Sequence[float] | None,
) -> list[str]:
    if sample is None:
        return []
    # The layout keys are assigned recursively before parent positions are stored,
    # so recover the active path by matching object identity.
    nodes = []
    current = tree
    while True:
        for key, item in positions.items():
            if item.node is current:
                nodes.append(key)
                break
        if isinstance(current, Leaf):
            break
        current = current.true_child if sample[current.feature_index] >= current.threshold else current.false_child
    return nodes


def _leaf_count(node: TreeNode) -> int:
    if isinstance(node, Leaf):
        return 1
    return _leaf_count(node.true_child) + _leaf_count(node.false_child)


def _tree_depth(node: TreeNode) -> int:
    if isinstance(node, Leaf):
        return 0
    return 1 + max(_tree_depth(node.true_child), _tree_depth(node.false_child))


def _tree_canvas_height(tree: TreeNode, config: SvgConfig) -> int:
    return 135 + (_tree_depth(tree) + 1) * config.tree_level_gap


def _network_required_features(network: NeuralDecisionNetwork) -> int:
    max_index = -1
    for neuron in network.condition_neurons.values():
        max_index = max(max_index, neuron.feature_index)
    return max_index + 1


def _network_node_color(name: str, layer: int, max_layer: int) -> str:
    if layer == 0:
        return "#ff6978"
    if layer == max_layer:
        return "#68a8ff"
    if name.startswith("g_"):
        return "#f6a23a"
    if name.startswith("path"):
        return "#67c587"
    return {1: "#f6a23a", 2: "#f4df5d", 3: "#63c37d", 4: "#4ba3e3"}.get(layer, "#f5a85a")


def _short_neuron_name(name: str) -> str:
    if name.startswith("g_true_"):
        return "T"
    if name.startswith("g_false_"):
        return "F"
    if name.startswith("out_"):
        return name.replace("out_", "out")
    if name.startswith("pass_"):
        return "pass"
    if name.startswith("and_"):
        return "and"
    return name[:8]


def _svg_header(width: int, height: int, config: SvgConfig) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{escape(config.font_family)}">'
        "\n<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>"
    )


def _circle(x: float, y: float, r: float, fill: str, stroke: str, stroke_width: float) -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    width: float,
    *,
    opacity: float = 1.0,
) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width:.1f}" opacity="{opacity:.2f}" '
        'stroke-linecap="round"/>'
    )


def _text(
    x: float,
    y: float,
    text: str,
    size: int,
    anchor: str,
    *,
    color: str = "#111827",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">{escape(text)}</text>'
    )


def _multiline_text(
    x: float,
    y: float,
    lines: Sequence[str],
    size: int,
    *,
    weight: str = "600",
) -> list[str]:
    output = [
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" font-size="{size}" '
        f'font-weight="{weight}" fill="#111827">'
    ]
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else size + 2
        output.append(f'<tspan x="{x:.1f}" dy="{dy}">{escape(str(line))}</tspan>')
    output.append("</text>")
    return output


def _edge_label(x: float, y: float, text: str) -> str:
    return (
        f'<g><rect x="{x - 14:.1f}" y="{y - 8:.1f}" width="28" height="14" rx="3" '
        'fill="#ffffff" stroke="#9ca3af" stroke-width="0.8"/>'
        f'<text x="{x:.1f}" y="{y + 3:.1f}" text-anchor="middle" font-size="8" '
        f'font-weight="700" fill="#111827">{escape(text)}</text></g>'
    )


def _legend(x: float, y: float) -> str:
    return (
        f'<g opacity="0.92"><rect x="{x:.1f}" y="{y:.1f}" width="165" height="86" rx="6" '
        'fill="#f3f4f6" stroke="#6b7280"/>'
        f'<text x="{x + 10:.1f}" y="{y + 18:.1f}" font-size="11" font-weight="700">Connections:</text>'
        f'<text x="{x + 10:.1f}" y="{y + 35:.1f}" font-size="10">Blue = positive weight</text>'
        f'<text x="{x + 10:.1f}" y="{y + 50:.1f}" font-size="10">Red = negative weight</text>'
        f'<text x="{x + 10:.1f}" y="{y + 68:.1f}" font-size="10">Node lower text = value</text>'
        "</g>"
    )
