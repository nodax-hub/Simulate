import bisect
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap

from dto import Point
from PumpController import PumpController
from utils import polyline_lengths, point_on_path, Polygon

FIGSIZE = (38, 26)


def plot_polygon(polygon: Polygon):
    points = polygon.vertices
    pts = list(points)
    if pts[0] != pts[-1]:
        pts = pts + [pts[0]]  # замыкаем для отображения

    xs, ys = zip(*pts)
    plt.figure()
    plt.fill(xs, ys, alpha=0.3, edgecolor="black", linewidth=1.5)
    plt.plot(xs, ys, "o-", color="black")

    # координаты центра тяжести (среднее по вершинам для подписи)
    cx, cy = sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)

    ha = 100 * 100  # 1Га = 100м x 100 м = 10'000 м^2
    area = polygon.area
    plt.text(cx, cy, f"Area = {area / ha :.2f} Га", ha="center", va="center", fontsize=12, weight="bold")
    plt.axis("equal")
    plt.show()


def plot_density_profile(points, v_pump, v_motion, density,
                         n_lower=4, n_upper=4):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    l = PumpController.instantaneous_densities(v_motion, v_pump)
    summary = pd.Series(l).describe()
    print(summary)
    print(f"diff = {l.mean() - density}")
    print(f"СКО = {np.mean([pow(v - density, 2) for v in l])}")

    xs, ys = zip(*points)

    # гарантируем, что density попадает в диапазон
    vmin = min(l.min(), density)
    vmax = max(l.max(), density)

    # нормировка
    if np.isclose(density, vmin):
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = ["green", "red"]
    elif np.isclose(density, vmax):
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = ["blue", "green"]
    else:
        colors = ["blue", "green", "red"]
        norm = TwoSlopeNorm(vmin=vmin, vcenter=density, vmax=vmax)

    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    sc = ax.scatter(xs, ys, c=l, cmap=cmap, norm=norm, label='траектория')
    ax.plot(xs, ys, '--', color='gray')

    # колорбар
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)  # fraction уменьшает ширину

    # два диапазона тиков для colorbar
    lower = np.linspace(vmin, density, n_lower + 1, endpoint=True)[:-1]
    upper = np.linspace(density, vmax, n_upper + 1, endpoint=True)
    ticks = np.concatenate([lower, upper])
    ticks = np.unique(np.round(ticks, 12))

    cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4g'))

    # аннотация уровня density
    y = norm(density)
    cbar.ax.annotate(
        f'excepted {density = }',
        xy=(1, y),
        xytext=(5, 0.5),
        xycoords='axes fraction',
        ha='left', va='center',
        arrowprops=dict(arrowstyle="->", color="red")
    )

    # === равномерная сетка на scatter ===
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=8))  # примерно 8 делений по X
    # ax.yaxis.set_major_locator(MaxNLocator(nbins=8))  # примерно 8 делений по Y
    # ax.grid(True, which='major', linestyle='--', alpha=0.7)

    # === равный шаг по осям ===
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='major', linestyle='--', alpha=0.7)

    ax.set_title('Плотность внесения на маршруте')
    plt.tight_layout()
    plt.show()


def scatter_with_color(p: list[Point], points_values: list[float], label="Скорость, м/с", cmap='viridis') -> None:
    xs, ys = zip(*p)
    sc = plt.scatter(xs, ys, c=points_values, cmap=cmap)
    plt.plot(xs, ys, '--')
    cbar = plt.colorbar(sc, fraction=0.1, pad=0.02)
    cbar.set_label(label)


def scatter_with_color_and_profile(p: list[Point], points_values: list[float], cmap='viridis', scatter_size=10) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE)

    path_x, path_y = zip(*p)
    sc = ax1.scatter(path_x, path_y, c=points_values, cmap=cmap, s=scatter_size)

    ax1.plot(path_x, path_y, linestyle="--", color="gray", alpha=0.3)
    ax1.set_xlabel("X, м")
    ax1.set_ylabel("Y, м")
    ax1.set_title("Траектория полёта (цвет = скорость, м/с)")
    fig.colorbar(sc, ax=ax1, label="скорость, м/с")

    ax2.plot(polyline_lengths(p), points_values, label="Скорость, м/с")
    ax2.set_xlabel("длина пути, м")
    ax2.set_ylabel("скорость, м/с")
    ax2.set_title("Профиль")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_at_ax_scatter_with_color_and_empty(
        ax,
        p: list[Point],
        points_values: list[float],
        empty_s_list: list[float],
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
) -> None:
    xs, ys = zip(*p)

    sc = ax.scatter(xs, ys, c=points_values, cmap='viridis', label='траектория')
    ax.plot(xs, ys, '--', color="gray")

    # колорбар
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("values")

    # отмечаем empty
    if empty_s_list:
        empty_points = [point_on_path(p, e) for e in empty_s_list]
        xe, ye = zip(*empty_points)
        ax.scatter(xe, ye, marker="x", s=40, color="red", label="empty")

    # подписи, если заданы
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # ax.legend()


def plot_at_ax_profile_speeds(ax, s_list, v_pump, v_motion):
    ax.plot(s_list, v_pump, label="v_pump", linewidth=1)
    ax.set_ylabel("v_pump")

    # правая ось
    axr = ax.twinx()
    axr.plot(s_list, v_motion, label="v_motion", linewidth=1, linestyle="--")
    axr.set_ylabel("v_motion")

    # объединённая легенда
    lines, labels = [], []
    for a in (ax, axr):
        l, lab = a.get_legend_handles_labels()
        lines += l
        labels += lab

    ax.legend(lines, labels, loc="upper right")

    ax.set_xlabel("s (distance)")
    ax.grid(True, which="both", axis="both", linestyle=":", linewidth=0.5)


def plot_speeds_profile(points, s_list, v_pump, v_motion, empty_s_list):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE)
    plot_at_ax_scatter_with_color_and_empty(ax1, points, v_motion, empty_s_list, title='v_motion')
    # plot_at_ax_scatter_with_color_and_empty(ax2, points, v_motion, empty_s_list)
    plot_at_ax_profile_speeds(ax2, s_list, v_pump, v_motion)
    # plot_at_ax_profile_speeds(ax4, s_list, v_pump, v_motion)
    plt.show()


@dataclass(frozen=True)
class PolylineSampler:
    waypoints: list[Point]
    s_nodes: list[float]  # накопленные длины для waypoints

    @classmethod
    def from_waypoints(cls, waypoints: list[Point]) -> "PolylineSampler":
        return cls(waypoints=waypoints, s_nodes=polyline_lengths(waypoints))

    @property
    def total_length(self) -> float:
        return self.s_nodes[-1]

    def position_at_s(self, s: float) -> Point:
        """
        Интерполяция точки на ломаной по дуговой длине s.
        s должен быть в диапазоне [0, total_length].
        """
        if s <= 0.0:
            return self.waypoints[0]
        if s >= self.total_length:
            return self.waypoints[-1]

        # индекс правого узла сегмента: s_nodes[i-1] <= s < s_nodes[i]
        i = bisect.bisect_right(self.s_nodes, s)
        # защитный случай (на границе)
        i = min(max(i, 1), len(self.waypoints) - 1)

        s0, s1 = self.s_nodes[i - 1], self.s_nodes[i]
        (x0, y0), (x1, y1) = self.waypoints[i - 1], self.waypoints[i]
        seg_len = s1 - s0
        # На случай нулевой длины сегмента (совпадающие точки)
        if seg_len == 0.0:
            return Point(x0, y0)
        t = (s - s0) / seg_len
        return Point(x0 + t * (x1 - x0), y0 + t * (y1 - y0))

    def sample_points(self, s_list: list[float]) -> list[Point]:
        return [self.position_at_s(s) for s in s_list]

    @classmethod
    def plot_trajectory_with_samples(cls,
                                     waypoints: list[Point],
                                     s_list: list[float],
                                     v_list: list[float],
                                     s_nodes: list[float] | None = None,
                                     figsize: tuple[int, int] = (12, 8),  # Увеличил картинку
                                     grid_step: float = 20.0,  # Шаг сетки
                                     ):
        if len(s_list) != len(v_list):
            raise ValueError("Длины списков s_list и v_list должны совпадать.")

        sampler = (
            cls(waypoints=waypoints, s_nodes=s_nodes)
            if s_nodes is not None
            else cls.from_waypoints(waypoints)
        )

        pts = sampler.sample_points(s_list)
        xs, ys = zip(*waypoints)
        xs_s, ys_s = zip(*pts) if pts else ([], [])

        fig, ax = plt.subplots(figsize=figsize)

        # Ломаная
        ax.plot(xs, ys, "--", linewidth=1.5)

        # Точки по s_list с раскраской по v_list
        sc = ax.scatter(xs_s, ys_s, c=v_list, s=40)

        # Colorbar меньшего размера
        cbar = fig.colorbar(sc, ax=ax, fraction=0.1, pad=0.02)
        cbar.set_label("Скорость, м/с")

        # Сетка равномерная
        ax.set_xticks(range(0, int(max(xs)) + 1, int(grid_step)))
        ax.set_yticks(range(0, int(max(ys)) + 1, int(grid_step)))
        ax.grid(True)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x, м")
        ax.set_ylabel("y, м")
        ax.set_title("Скорость по траектории")
        plt.tight_layout()
        plt.show()
