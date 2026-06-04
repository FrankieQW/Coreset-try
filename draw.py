from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RESULTS = [
    (
        Path("baseline/baseline_random10_resnet18/baseline_results.json"),
        "Random-10% Baseline",
    ),
    (
        Path("coreset/coreset_pd10_resnet18/coreset_results.json"),
        "PD-Coreset-10%",
    ),
]

COLORS = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#be123c",
]


@dataclass
class Series:
    label: str
    path: Path
    epochs: list[int]
    train_mse: list[float]
    val_mse: list[float]
    best_val_mse: float
    best_epoch: int
    final_val_mse: float
    config: dict
    dataset: dict


def load_series(path: Path, label: str | None = None) -> Series:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("metrics", {}).get("history", [])
    if not history:
        raise ValueError(f"No metrics.history found in {path}")

    epochs = [int(row["epoch"]) for row in history]
    train_mse = [float(row["train_mse"]) for row in history]
    val_mse = [float(row["val_mse"]) for row in history]
    best_index = min(range(len(val_mse)), key=lambda i: val_mse[i])
    metrics = data.get("metrics", {})

    return Series(
        label=label or infer_label(path, data),
        path=path,
        epochs=epochs,
        train_mse=train_mse,
        val_mse=val_mse,
        best_val_mse=float(metrics.get("best_val_mse", val_mse[best_index])),
        best_epoch=epochs[best_index],
        final_val_mse=float(metrics.get("final_val_mse", val_mse[-1])),
        config=data.get("config", {}),
        dataset=data.get("dataset", {}),
    )


def infer_label(path: Path, data: dict) -> str:
    if "coreset" in data:
        method = data["coreset"].get("method", "Coreset")
        fraction = data.get("config", {}).get("coreset_fraction")
        if fraction is not None:
            return f"{method}-{fraction:g}"
        return method
    if "sample_fraction" in data.get("config", {}):
        fraction = data["config"]["sample_fraction"]
        return f"Random-{fraction:g}"
    return path.parent.name


def parse_result_arg(value: str) -> tuple[Path, str | None]:
    if "=" in value:
        raw_path, label = value.split("=", 1)
        return Path(raw_path), label
    return Path(value), None


def make_points_path(xs: list[int], ys: list[float], sx, sy) -> str:
    return " ".join(
        ("M" if i == 0 else "L") + f"{sx(x):.2f},{sy(y):.2f}"
        for i, (x, y) in enumerate(zip(xs, ys))
    )


def nice_ticks(max_value: float, count: int = 6) -> list[float]:
    if max_value <= 0:
        return [0.0]
    return [max_value * i / (count - 1) for i in range(count)]


def draw_svg(series_list: list[Series], output_path: Path, title: str) -> None:
    width, height = 1100, 680
    left, right, top, bottom = 95, 40, 68, 105
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_min = min(min(series.epochs) for series in series_list)
    x_max = max(max(series.epochs) for series in series_list)
    if x_min == x_max:
        x_max = x_min + 1

    y_min = 0.0
    y_max = max(
        max(max(series.train_mse), max(series.val_mse)) for series in series_list
    )
    y_max *= 1.10

    def sx(x: int) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    svg: list[str] = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg.append(
        f'<text x="{width/2}" y="36" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{escape_xml(title)}</text>'
    )

    for y in nice_ticks(y_max):
        yy = sy(y)
        svg.append(
            f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{left-12}" y="{yy+4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">{y:.3f}</text>'
        )

    tick_count = min(9, max(2, x_max - x_min + 1))
    for i in range(tick_count):
        x = round(x_min + (x_max - x_min) * i / (tick_count - 1))
        xx = sx(x)
        svg.append(
            f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{height-bottom}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{xx:.2f}" y="{height-bottom+28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">{x}</text>'
        )

    svg.append(
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827" stroke-width="1.4"/>'
    )
    svg.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827" stroke-width="1.4"/>'
    )
    svg.append(
        f'<text x="{left + plot_w/2:.2f}" y="{height-36}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#111827">Epoch</text>'
    )
    svg.append(
        f'<text x="26" y="{top + plot_h/2:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#111827" transform="rotate(-90 26 {top + plot_h/2:.2f})">MSE</text>'
    )

    legend_item_width = 245
    legend_rows = (len(series_list) + 1) // 2
    legend_w = min(plot_w, legend_item_width * 2)
    legend_h = 28 + legend_rows * 24
    legend_x = width - right - legend_w + 14
    legend_y = top + 32
    svg.append(
        f'<rect x="{legend_x-14}" y="{legend_y-24}" width="{legend_w}" height="{legend_h}" rx="4" fill="#ffffff" stroke="#d1d5db"/>'
    )

    for index, series in enumerate(series_list):
        color = COLORS[index % len(COLORS)]
        train_path = make_points_path(series.epochs, series.train_mse, sx, sy)
        val_path = make_points_path(series.epochs, series.val_mse, sx, sy)
        svg.append(
            f'<path d="{train_path}" fill="none" stroke="{color}" stroke-width="2.2" stroke-dasharray="7 5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        svg.append(
            f'<path d="{val_path}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )

        best_x = sx(series.best_epoch)
        best_y = sy(series.best_val_mse)
        svg.append(f'<circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="4.5" fill="{color}"/>')

        lx = legend_x + (index % 2) * legend_item_width
        ly = legend_y + (index // 2) * 24
        svg.append(
            f'<line x1="{lx}" y1="{ly}" x2="{lx+34}" y2="{ly}" stroke="{color}" stroke-width="3"/>'
        )
        svg.append(
            f'<line x1="{lx}" y1="{ly+8}" x2="{lx+34}" y2="{ly+8}" stroke="{color}" stroke-width="2.2" stroke-dasharray="7 5"/>'
        )
        text = f"{series.label} val, train dashed"
        svg.append(
            f'<text x="{lx+44}" y="{ly+5}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{escape_xml(text)}</text>'
        )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def write_summary_csv(series_list: list[Series], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "path",
                "epochs",
                "best_val_mse",
                "best_epoch",
                "final_val_mse",
                "arm",
                "instruction",
                "sample_fraction",
                "coreset_fraction",
            ]
        )
        for series in series_list:
            writer.writerow(
                [
                    series.label,
                    series.path,
                    len(series.epochs),
                    series.best_val_mse,
                    series.best_epoch,
                    series.final_val_mse,
                    series.config.get("arm", ""),
                    series.config.get("instruction", ""),
                    series.config.get("sample_fraction", ""),
                    series.config.get("coreset_fraction", ""),
                ]
            )


def escape_xml(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw train/validation MSE comparison curves from baseline_results.json and coreset_results.json."
    )
    parser.add_argument(
        "--result",
        action="append",
        help="Result JSON path, optionally path=label. Can be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("comparison_mse_curve.svg"),
        help="Output SVG path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("comparison_mse_summary.csv"),
        help="Output summary CSV path.",
    )
    parser.add_argument(
        "--title",
        default="Baseline vs Coreset MLP Action Prediction MSE",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_specs = (
        [parse_result_arg(value) for value in args.result]
        if args.result
        else DEFAULT_RESULTS
    )
    series_list = [load_series(path, label) for path, label in result_specs]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    draw_svg(series_list, args.output, args.title)
    write_summary_csv(series_list, args.summary_csv)

    print(f"saved svg: {args.output}")
    print(f"saved summary: {args.summary_csv}")
    for series in series_list:
        print(
            f"{series.label}: best_val_mse={series.best_val_mse:.8f}, epoch={series.best_epoch}, final_val_mse={series.final_val_mse:.8f}"
        )


if __name__ == "__main__":
    main()
