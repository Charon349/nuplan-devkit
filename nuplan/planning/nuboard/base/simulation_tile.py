import concurrent.futures
import json
import logging
import lzma
import math
import os
import pathlib
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import msgpack
import numpy as np
from bokeh.document import without_document_lock
from bokeh.document.document import Document
from bokeh.events import PointEvent
from bokeh.io.export import get_screenshot_as_png
from bokeh.layouts import column, gridplot, row
from bokeh.models import Button, ColumnDataSource, Slider, Title
from bokeh.plotting.figure import Figure
from bokeh.server.callbacks import PeriodicCallback
from bokeh.util.callback_manager import EventCallback
from selenium import webdriver
from tornado import gen
from tqdm import tqdm

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.abstract_map_factory import AbstractMapFactory
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
from nuplan.planning.nuboard.base.data_class import SimulationScenarioKey
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.base.plot_data import MapPoint, SimulationData, SimulationFigure
from nuplan.planning.nuboard.style import (
    simulation_map_layer_color,
    simulation_tile_style,
    simulation_tile_trajectory_style,
)
from nuplan.planning.nuboard.tabs.config.scenario_tab_config import (
    ScenarioTabFrameButtonConfig,
    first_button_config,
    last_button_config,
    next_button_config,
    play_button_config,
    prev_button_config,
)
from nuplan.planning.simulation.simulation_log import SimulationLog

try:
    import chromedriver_binary
except ImportError:
    chromedriver_binary = None


logger = logging.getLogger(__name__)


def extract_source_from_states(states: List[EgoState]) -> ColumnDataSource:
    """Helper function to get the xy coordinates into ColumnDataSource format from a list of states.
    :param states: List of states (containing the pose)
    :return: A ColumnDataSource object containing the xy coordinates.
    """
    x_coords = []
    y_coords = []
    for state in states:
        x_coords.append(state.center.x)
        y_coords.append(state.center.y)
    source = ColumnDataSource(dict(xs=x_coords, ys=y_coords))
    return source


def _transform_local_xy_to_global(local_xy: List[List[float]], ego: Dict[str, float]) -> Tuple[List[float], List[float]]:
    """
    Transform local ego-frame xy coordinates into global coordinates.
    :param local_xy: Local [x, y] coordinates.
    :param ego: Ego pose with x, y, heading.
    :return: Global x/y lists.
    """
    origin = StateSE2(float(ego["x"]), float(ego["y"]), float(ego["heading"]))
    relative_poses = [StateSE2(float(xy[0]), float(xy[1]), 0.0) for xy in local_xy]
    absolute_poses = relative_to_absolute_poses(origin, relative_poses)
    xs = [pose.x for pose in absolute_poses]
    ys = [pose.y for pose in absolute_poses]
    return xs, ys


def _load_anchor_xy_templates(anchor_path: str, state_format: str) -> Optional[np.ndarray]:
    """
    Load all ego anchor templates as local-frame xy trajectories.
    :param anchor_path: Anchor file path.
    :param state_format: "delta" or "absolute".
    :return: Anchor local xy array with shape [K, T, 2].
    """
    if not anchor_path:
        return None

    path = pathlib.Path(anchor_path)
    if not path.exists():
        return None

    if path.suffix in (".pt", ".pth"):
        import torch

        anchor = torch.load(str(path), map_location="cpu")
    else:
        anchor = np.load(str(path))

    if isinstance(anchor, dict):
        for key in ("anchors", "ego_anchors", "plan_anchor"):
            if key in anchor:
                anchor = anchor[key]
                break
        else:
            return None

    if hasattr(anchor, "detach"):
        anchor = anchor.detach().cpu().numpy()

    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.ndim != 3 or anchor.shape[-1] < 2:
        return None

    # Older logs did not record anchor_state_format. A [K, T, 3] anchor file is
    # absolute [x, y, heading]; delta anchors in this project have [K, T, 4].
    if state_format == "absolute" or anchor.shape[-1] == 3:
        return anchor[..., :2]

    return np.cumsum(anchor[..., :2], axis=1)


def _global_xy_to_local_xy(global_xy: List[List[float]], ego: Any) -> np.ndarray:
    """
    Transform global xy coordinates into ego local frame.
    :param global_xy: Global xy coordinates.
    :param ego: Ego rear axle pose.
    :return: Local xy array.
    """
    ego_x = float(ego.x)
    ego_y = float(ego.y)
    ego_heading = float(ego.heading)
    cos_h = math.cos(ego_heading)
    sin_h = math.sin(ego_heading)

    local_xy = []
    for x, y in global_xy:
        dx = float(x) - ego_x
        dy = float(y) - ego_y
        local_xy.append([dx * cos_h + dy * sin_h, -dx * sin_h + dy * cos_h])
    return np.asarray(local_xy, dtype=np.float64)


def _extract_serialization_type(first_file: pathlib.Path) -> str:
    """
    Deduce the serialization type
    :param first_file: serialized file
    :return: one from ["msgpack", "pickle", "json"].
    """
    msg_pack = first_file.suffixes == [".msgpack", ".xz"]
    msg_pickle = first_file.suffixes == [".pkl", ".xz"]
    msg_json = first_file.suffix == ".json"
    number_of_available_types = int(msg_pack) + int(msg_json) + int(msg_pickle)

    # We can handle only conclusive serialization type
    if number_of_available_types != 1:
        raise RuntimeError(f"Inconclusive file type: {first_file}!")

    if msg_pickle:
        return "pickle"
    elif msg_json:
        return "json"
    elif msg_pack:
        return "msgpack"
    else:
        raise RuntimeError("Unknown condition!")


def _load_data(file_name: pathlib.Path, serialization_type: str) -> Any:
    """
    Load data from file_name
    :param file_name: the name of a file which we want to deserialize
    :param serialization_type: type of serialization of the file
    :return: deserialized type
    """
    if serialization_type == "json":
        with open(str(file_name), "r") as f:
            return json.load(f)
    elif serialization_type == "msgpack":
        with lzma.open(str(file_name), "rb") as f:
            return msgpack.unpackb(f.read())
    elif serialization_type == "pickle":
        with lzma.open(
            str(file_name),
            "rb",
        ) as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown serialization type: {serialization_type}!")


class SimulationTile:
    """Scenario simulation tile for visualization."""

    def __init__(
        self,
        doc: Document,
        experiment_file_data: ExperimentFileData,
        vehicle_parameters: VehicleParameters,
        map_factory: AbstractMapFactory,
        period_milliseconds: int = 5000,
        radius: float = 300.0,
        async_rendering: bool = True,
        frame_rate_cap_hz: int = 60,
    ):
        """
        Scenario simulation tile.
        :param doc: Bokeh HTML document.
        :param experiment_file_data: Experiment file data.
        :param vehicle_parameters: Ego pose parameters.
        :param map_factory: Map factory for building maps.
        :param period_milliseconds: Milliseconds to update the tile.
        :param radius: Map radius.
        :param async_rendering: When true, will use threads to render asynchronously.
        :param frame_rate_cap_hz: Maximum frames to render per second. Internally this value is capped at 60.
        """
        self._doc = doc
        self._vehicle_parameters = vehicle_parameters
        self._map_factory = map_factory
        self._experiment_file_data = experiment_file_data
        self._period_milliseconds = period_milliseconds
        self._radius = radius
        self._selected_scenario_keys: List[SimulationScenarioKey] = []
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._maps: Dict[str, AbstractMap] = {}
        self._figures: List[SimulationFigure] = []
        self._nearest_vector_map: Dict[SemanticMapLayer, List[MapObject]] = {}
        self._async_rendering = async_rendering
        self._plot_render_queue: Optional[Tuple[SimulationFigure, int]] = None
        self._doc.add_periodic_callback(self._periodic_callback, period_milliseconds=1000)
        self._last_frame_time = time.time()
        self._current_frame_index = 0
        self._last_frame_index = 0
        self._playback_callback_handle: Optional[PeriodicCallback] = None

        # Recheck that we are passed the correct frame rate cap.
        if frame_rate_cap_hz < 1 or frame_rate_cap_hz > 60:
            raise ValueError("frame_rate_cap_hz should be between 1 and 60")

        self._minimum_frame_time_seconds = 1.0 / float(frame_rate_cap_hz)

        logger.info("Minimum frame time=%4.3f s", self._minimum_frame_time_seconds)

    @property
    def get_figure_data(self) -> List[SimulationFigure]:
        """Return figure data."""
        return self._figures

    @property
    def is_in_playback(self) -> bool:
        """Returns True if we're currently rendering a playback of a figure."""
        return self._playback_callback_handle is not None

    def _on_mouse_move(self, event: PointEvent, figure_index: int) -> None:
        """
        Event when mouse moving in a figure.
        :param event: Point event.
        :param figure_index: Figure index where the mouse is moving.
        """
        main_figure = self._figures[figure_index]
        # Update x and y coordinate values.
        main_figure.x_y_coordinate_title.text = (
            f"x [m]: {np.round(event.x, simulation_tile_style['decimal_points'])}, "
            f"y [m]: {np.round(event.y, simulation_tile_style['decimal_points'])}"
        )

    def _create_frame_control_button(
        self, button_config: ScenarioTabFrameButtonConfig, click_callback: EventCallback, figure_index: int
    ) -> Button:
        """
        Helper function to create a frame control button (prev, play, etc.) based on the provided config.
        :param button_config: Configuration object for the frame control button.
        :param click_callback: Button click event callback that will be registered to the created button.
        :param figure_index: The figure index to be passed to the button's click event callback.
        :return: The created Bokeh Button instance.
        """
        button_instance = Button(
            label=button_config.label,
            margin=button_config.margin,
            css_classes=button_config.css_classes,
            width=button_config.width,
        )
        button_instance.on_click(partial(click_callback, figure_index=figure_index))
        return button_instance

    def _create_initial_figure(
        self, figure_index: int, figure_sizes: List[int], backend: Optional[str] = "webgl"
    ) -> SimulationFigure:
        """
        Create an initial Bokeh figure.
        :param figure_index: Figure index.
        :param figure_sizes: width and height in pixels.
        :param backend: Bokeh figure backend.
        :return: A Bokeh figure.
        """
        selected_scenario_key = self._selected_scenario_keys[figure_index]

        experiment_path = Path(
            self._experiment_file_data.file_paths[selected_scenario_key.nuboard_file_index].metric_main_path
        )
        planner_name = selected_scenario_key.planner_name
        presented_planner_name = planner_name + f' ({experiment_path.stem})'
        simulation_figure = Figure(
            x_range=(-self._radius, self._radius),
            y_range=(-self._radius, self._radius),
            width=figure_sizes[0],
            height=figure_sizes[1],
            title=f"{presented_planner_name}",
            tools=["pan", "wheel_zoom", "save", "reset"],
            match_aspect=True,
            active_scroll="wheel_zoom",
            margin=simulation_tile_style["figure_margins"],
            background_fill_color=simulation_tile_style["background_color"],
            output_backend=backend,
        )
        simulation_figure.on_event("mousemove", partial(self._on_mouse_move, figure_index=figure_index))
        simulation_figure.axis.visible = False
        simulation_figure.xgrid.visible = False
        simulation_figure.ygrid.visible = False
        simulation_figure.title.text_font_size = simulation_tile_style["figure_title_text_font_size"]
        x_y_coordinate_title = Title(text="x [m]: , y [m]: ")
        simulation_figure.add_layout(x_y_coordinate_title, 'below')
        slider = Slider(
            start=0,
            end=1,
            value=0,
            step=1,
            title="Frame",
            margin=simulation_tile_style["slider_margins"],
            css_classes=["scenario-frame-slider"],
        )
        slider.on_change("value", partial(self._slider_on_change, figure_index=figure_index))
        video_button = Button(
            label="Render video",
            margin=simulation_tile_style["video_button_margins"],
            css_classes=["scenario-video-button"],
        )
        video_button.on_click(partial(self._video_button_on_click, figure_index=figure_index))

        first_button = self._create_frame_control_button(first_button_config, self._first_button_on_click, figure_index)
        prev_button = self._create_frame_control_button(prev_button_config, self._prev_button_on_click, figure_index)
        play_button = self._create_frame_control_button(play_button_config, self._play_button_on_click, figure_index)
        next_button = self._create_frame_control_button(next_button_config, self._next_button_on_click, figure_index)
        last_button = self._create_frame_control_button(last_button_config, self._last_button_on_click, figure_index)

        assert len(selected_scenario_key.files) == 1, "Expected one file containing the serialized SimulationLog."
        simulation_file = next(iter(selected_scenario_key.files))
        simulation_log = SimulationLog.load_data(simulation_file)

        simulation_figure_data = SimulationFigure(
            figure=simulation_figure,
            file_path_index=selected_scenario_key.nuboard_file_index,
            figure_title_name=presented_planner_name,
            slider=slider,
            video_button=video_button,
            first_button=first_button,
            prev_button=prev_button,
            play_button=play_button,
            next_button=next_button,
            last_button=last_button,
            vehicle_parameters=self._vehicle_parameters,
            planner_name=planner_name,
            scenario=simulation_log.scenario,
            simulation_history=simulation_log.simulation_history,
            x_y_coordinate_title=x_y_coordinate_title,
        )
        selected_anchor_sources = self._load_selected_anchor_sources(
            simulation_file=simulation_file,
            simulation_history=simulation_log.simulation_history,
            scenario=simulation_log.scenario,
        )
        simulation_figure_data.selected_anchor_trajectory_plot.update_data_sources(  # type: ignore
            frame_sources=selected_anchor_sources,
            history=simulation_log.simulation_history,
        )

        return simulation_figure_data

    def _map_api(self, map_name: str) -> AbstractMap:
        """
        Get a map api.
        :param map_name: Map name.
        :return Map api.
        """
        if map_name not in self._maps:
            self._maps[map_name] = self._map_factory.build_map_from_name(map_name)

        return self._maps[map_name]

    @staticmethod
    def _find_anchor_log_dirs(simulation_file: pathlib.Path) -> List[pathlib.Path]:
        """
        Find anchor selection log directories associated with a simulation log.
        :param simulation_file: Serialized SimulationLog path.
        :return: Candidate anchor_selection_logs directories.
        """
        candidates = []
        explicit_anchor_log_dir = os.getenv("NUBOARD_ANCHOR_LOG_DIR", "")
        if explicit_anchor_log_dir:
            explicit_candidate = pathlib.Path(explicit_anchor_log_dir)
            if explicit_candidate.exists() and explicit_candidate.is_dir():
                candidates.append(explicit_candidate)
            else:
                logger.warning("NUBOARD_ANCHOR_LOG_DIR does not exist: %s", explicit_anchor_log_dir)

        for path in [simulation_file.parent] + list(simulation_file.parents):
            candidate = path / "anchor_selection_logs"
            if candidate.exists() and candidate.is_dir():
                candidates.append(candidate)

        # Backward compatibility for logs written before anchor logs were moved under
        # <NUPLAN_EXP_ROOT>/exp/simulation/<challenge>/<experiment_uid>/anchor_selection_logs.
        parts = simulation_file.parts
        for index in range(len(parts) - 2):
            if parts[index] == "exp" and parts[index + 1] == "simulation":
                root = pathlib.Path(*parts[:index])
                for end_index in range(index + 4, len(parts)):
                    experiment_uid = pathlib.Path(*parts[index + 3 : end_index])
                    legacy_candidate = root / experiment_uid / "anchor_selection_logs"
                    if legacy_candidate.exists() and legacy_candidate.is_dir():
                        candidates.append(legacy_candidate)
                break

        return list(dict.fromkeys(candidates))

    @staticmethod
    def _gt_l2_anchor_entries(
        record: Dict[str, Any],
        frame_index: int,
        history_samples: List[Any],
        expert_states: List[EgoState],
        expert_time_to_index: Dict[int, int],
        anchor_template_cache: Dict[Tuple[str, str], Optional[np.ndarray]],
    ) -> List[Dict[str, Any]]:
        """
        Get top3 anchors with the smallest L2 distance to expert future.
        :param record: Anchor selection record.
        :param frame_index: Current simulation frame.
        :param history_samples: Simulation history samples.
        :param expert_states: Expert ego states.
        :param expert_time_to_index: Expert state time to index map.
        :param anchor_template_cache: Cache of loaded anchor templates.
        :return: Anchor entries compatible with selected-anchor plotting.
        """
        if frame_index >= len(history_samples):
            return []

        anchor_path = str(record.get("anchor_path") or "")
        state_format = str(record.get("anchor_state_format") or "delta")
        cache_key = (anchor_path, state_format)
        if cache_key not in anchor_template_cache:
            anchor_template_cache[cache_key] = _load_anchor_xy_templates(anchor_path, state_format)

        anchor_templates = anchor_template_cache[cache_key]
        if anchor_templates is None or anchor_templates.size == 0:
            return []

        current_ego_state = history_samples[frame_index].ego_state
        expert_index = expert_time_to_index.get(current_ego_state.time_us)
        if expert_index is None:
            return []

        future_len = anchor_templates.shape[1]
        future_expert_states = expert_states[expert_index + 1 : expert_index + 1 + future_len]
        if not future_expert_states:
            return []

        expert_global_xy = [[state.rear_axle.x, state.rear_axle.y] for state in future_expert_states]
        expert_local_xy = _global_xy_to_local_xy(expert_global_xy, current_ego_state.rear_axle)
        horizon = min(anchor_templates.shape[1], expert_local_xy.shape[0])
        if horizon <= 0:
            return []

        ego_heading = current_ego_state.rear_axle.heading
        expert_local_headings = np.array([state.rear_axle.heading - ego_heading for state in future_expert_states[:horizon]])
        
        gt_cos = np.cos(expert_local_headings)[None, :]
        gt_sin = np.sin(expert_local_headings)[None, :]

        diff = anchor_templates[:, :horizon, :2] - expert_local_xy[None, :horizon, :2]
        
        lat_err = np.abs(diff[..., 0] * (-gt_sin) + diff[..., 1] * gt_cos)
        lon_err = np.abs(diff[..., 0] * gt_cos + diff[..., 1] * gt_sin)
        
        w_lat = 1.0
        w_lon = 0.2
        l2_distances = (w_lat * lat_err + w_lon * lon_err).mean(axis=1)
        top_indices = np.argsort(l2_distances)[:3]
        score_by_index = {
            item.get("index"): item.get("score")
            for item in record.get("anchor_score_topk", [])
            if isinstance(item, dict)
        }

        return [
            {
                "rank": rank,
                "index": int(anchor_index),
                "score": score_by_index.get(int(anchor_index)),
                "l2": float(l2_distances[anchor_index]),
                "xy": anchor_templates[anchor_index].tolist(),
                "source": "gt_l2_topk",
            }
            for rank, anchor_index in enumerate(top_indices, start=1)
        ]

    @staticmethod
    def _score_topk_anchor_entries(
        record: Dict[str, Any],
        anchor_template_cache: Dict[Tuple[str, str], Optional[np.ndarray]],
    ) -> List[Dict[str, Any]]:
        """
        Get top score anchors from the current record.
        :param record: Anchor selection record.
        :param anchor_template_cache: Cache of loaded anchor templates.
        :return: Top score anchor entries.
        """
        anchor_entries = record.get("topk_anchor_template_xy") or []
        if anchor_entries:
            return [
                {
                    **anchor_entry,
                    "l2": anchor_entry.get("l2"),
                    "source": anchor_entry.get("source", "score_topk"),
                }
                for anchor_entry in anchor_entries[:3]
                if isinstance(anchor_entry, dict)
            ]

        anchor_path = str(record.get("anchor_path") or "")
        state_format = str(record.get("anchor_state_format") or "delta")
        cache_key = (anchor_path, state_format)
        if cache_key not in anchor_template_cache:
            anchor_template_cache[cache_key] = _load_anchor_xy_templates(anchor_path, state_format)

        anchor_templates = anchor_template_cache[cache_key]
        if anchor_templates is not None and record.get("anchor_score_topk"):
            entries = []
            for rank, item in enumerate(record.get("anchor_score_topk", [])[:3], start=1):
                if not isinstance(item, dict):
                    continue
                anchor_index = item.get("index")
                if anchor_index is None:
                    continue
                anchor_index = int(anchor_index)
                if anchor_index < 0 or anchor_index >= anchor_templates.shape[0]:
                    continue
                entries.append(
                    {
                        "rank": rank,
                        "index": anchor_index,
                        "score": item.get("score"),
                        "l2": None,
                        "xy": anchor_templates[anchor_index].tolist(),
                        "source": "score_topk",
                    }
                )
            if entries:
                return entries

        return [
            {
                "rank": 1,
                "index": record.get("selected_anchor_index"),
                "score": record.get("anchor_score_selected"),
                "l2": None,
                "xy": record.get("selected_anchor_template_xy"),
                "source": "score_topk",
            }
        ]

    @staticmethod
    def _load_selected_anchor_sources(
        simulation_file: pathlib.Path, simulation_history: Any, scenario: Any
    ) -> Dict[int, ColumnDataSource]:
        """
        Load selected anchor trajectories from anchor selection JSONL logs.
        :param simulation_file: Serialized SimulationLog path.
        :param simulation_history: Simulation history used to align ego_time_us to frame index.
        :param scenario: Simulation scenario used to compute GT-nearest anchors.
        :return: Selected anchor data sources keyed by frame index.
        """
        history_samples = list(simulation_history.data)
        time_to_frame = {sample.ego_state.time_us: index for index, sample in enumerate(history_samples)}
        scenario_start_time_us = history_samples[0].ego_state.time_us if history_samples else None
        scenario_id_time_marker = f"_{scenario_start_time_us}_" if scenario_start_time_us is not None else ""
        frame_sources: Dict[int, ColumnDataSource] = {}
        expert_states = list(scenario.get_expert_ego_trajectory()) if scenario is not None else []
        expert_time_to_index = {state.time_us: index for index, state in enumerate(expert_states)}
        anchor_template_cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
        anchor_log_dirs = SimulationTile._find_anchor_log_dirs(simulation_file)
        if not anchor_log_dirs:
            logger.warning("No selected anchor log dirs found for simulation file: %s", simulation_file)
            return frame_sources

        loaded_records = 0
        matched_records = 0
        for anchor_log_dir in anchor_log_dirs:
            for anchor_log_file in sorted(anchor_log_dir.glob("*.jsonl")):
                try:
                    with open(anchor_log_file, "r") as file:
                        for line in file:
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            loaded_records += 1

                            record_scenario_id = str(record.get("scenario_id") or "")
                            scenario_id_matches_current = bool(
                                scenario_id_time_marker and scenario_id_time_marker in record_scenario_id
                            )
                            if record_scenario_id and not scenario_id_matches_current:
                                continue

                            frame_index = time_to_frame.get(record.get("ego_time_us"))
                            if frame_index is None:
                                iteration_index = record.get("iteration_index")
                                if (
                                    scenario_id_matches_current
                                    and isinstance(iteration_index, int)
                                    and 0 <= iteration_index < len(history_samples)
                                ):
                                    frame_index = iteration_index
                            ego = record.get("ego")
                            if frame_index is None or not ego:
                                continue

                            anchor_entries = SimulationTile._score_topk_anchor_entries(
                                record=record,
                                anchor_template_cache=anchor_template_cache,
                            )
                            anchor_entries += SimulationTile._gt_l2_anchor_entries(
                                record=record,
                                frame_index=frame_index,
                                history_samples=history_samples,
                                expert_states=expert_states,
                                expert_time_to_index=expert_time_to_index,
                                anchor_template_cache=anchor_template_cache,
                            )

                            xs_list = []
                            ys_list = []
                            anchor_sources = []
                            anchor_ranks = []
                            selected_anchor_indices = []
                            anchor_scores = []
                            anchor_l2s = []
                            line_colors = []
                            line_widths = []
                            for anchor_entry in anchor_entries:
                                local_xy = anchor_entry.get("xy")
                                if local_xy is None or len(local_xy) == 0:
                                    continue
                                local_xy = [[0.0, 0.0], *local_xy]

                                source = anchor_entry.get("source", "score_topk")
                                style_key = (
                                    "gt_l2_anchor_topk"
                                    if source == "gt_l2_topk"
                                    else "selected_anchor_topk"
                                )
                                topk_style = simulation_tile_trajectory_style[style_key]
                                rank = int(anchor_entry.get("rank", 1))
                                xs, ys = _transform_local_xy_to_global(local_xy, ego)
                                xs_list.append(xs)
                                ys_list.append(ys)
                                anchor_sources.append(source)
                                anchor_ranks.append(anchor_entry.get("rank", rank))
                                selected_anchor_indices.append(anchor_entry.get("index"))
                                anchor_scores.append(anchor_entry.get("score"))
                                anchor_l2s.append(anchor_entry.get("l2"))
                                line_colors.append(
                                    topk_style["line_colors"][
                                        min(rank - 1, len(topk_style["line_colors"]) - 1)
                                    ]
                                )
                                line_widths.append(
                                    topk_style["line_widths"][
                                        min(rank - 1, len(topk_style["line_widths"]) - 1)
                                    ]
                                )
                            if not xs_list:
                                continue

                            matched_records += 1
                            frame_sources[frame_index] = ColumnDataSource(
                                dict(
                                    xs=xs_list,
                                    ys=ys_list,
                                    selected_anchor_index=selected_anchor_indices,
                                    anchor_source=anchor_sources,
                                    anchor_rank=anchor_ranks,
                                    anchor_score=anchor_scores,
                                    anchor_l2=anchor_l2s,
                                    line_colors=line_colors,
                                    line_widths=line_widths,
                                )
                            )
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    logger.info("Unable to load selected anchor log %s: %s", anchor_log_file, exc)

        if loaded_records and not matched_records:
            logger.warning(
                "Loaded %d selected anchor records but matched 0 frames for simulation file: %s",
                loaded_records,
                simulation_file,
            )

        return frame_sources

    def init_simulations(self, figure_sizes: List[int]) -> None:
        """
        Initialization of the visualization of simulation panel.
        :param figure_sizes: Width and height in pixels.
        """
        self._figures = []
        for figure_index in range(len(self._selected_scenario_keys)):
            simulation_figure = self._create_initial_figure(figure_index=figure_index, figure_sizes=figure_sizes)
            self._figures.append(simulation_figure)

    @property
    def figures(self) -> List[SimulationFigure]:
        """
        Access bokeh figures.
        :return A list of bokeh figures.
        """
        return self._figures

    def _render_simulation_layouts(self) -> List[SimulationData]:
        """
        Render simulation layouts.
        :return: A list of columns or rows.
        """
        grid_layouts: List[SimulationData] = []
        for simulation_figure in self.figures:
            grid_layouts.append(
                SimulationData(
                    planner_name=simulation_figure.planner_name,
                    simulation_figure=simulation_figure,
                    plot=gridplot(
                        [
                            [simulation_figure.slider],
                            [
                                row(
                                    [
                                        simulation_figure.first_button,
                                        simulation_figure.prev_button,
                                        simulation_figure.play_button,
                                        simulation_figure.next_button,
                                        simulation_figure.last_button,
                                    ]
                                )
                            ],
                            [simulation_figure.figure],
                            [simulation_figure.video_button],
                        ],
                        toolbar_location="left",
                    ),
                )
            )
        return grid_layouts

    def render_simulation_tiles(
        self,
        selected_scenario_keys: List[SimulationScenarioKey],
        figure_sizes: List[int] = simulation_tile_style['figure_sizes'],
        hidden_glyph_names: Optional[List[str]] = None,
    ) -> List[SimulationData]:
        """
        Render simulation tiles.
        :param selected_scenario_keys: A list of selected scenario keys.
        :param figure_sizes: Width and height in pixels.
        :param hidden_glyph_names: A list of glyph names to be hidden.
        :return A list of bokeh layouts.
        """
        self._selected_scenario_keys = selected_scenario_keys
        self.init_simulations(figure_sizes=figure_sizes)
        for main_figure in tqdm(self._figures, desc="Rendering a scenario"):
            self._render_scenario(main_figure, hidden_glyph_names=hidden_glyph_names)

        layouts = self._render_simulation_layouts()
        return layouts

    @gen.coroutine
    @without_document_lock
    def _video_button_on_click(self, figure_index: int) -> None:
        """
        Callback to video button click event.
        Note that this callback in run on a background thread.
        :param figure_index: Figure index.
        """
        self._figures[figure_index].video_button.disabled = True
        self._figures[figure_index].video_button.label = "Rendering video now..."

        self._executor.submit(self._video_button_next_tick, figure_index)

    def _reset_video_button(self, figure_index: int) -> None:
        """
        Reset a video button after exporting is done.
        :param figure_index: Figure index.
        """
        self.figures[figure_index].video_button.label = "Render video"
        self.figures[figure_index].video_button.disabled = False

    def _update_video_button_label(self, figure_index: int, label: str) -> None:
        """
        Update a video button label to show progress when rendering a video.
        :param figure_index: Figure index.
        :param label: New video button text.
        """
        self.figures[figure_index].video_button.label = label

    def _video_button_next_tick(self, figure_index: int) -> None:
        """
        Synchronous callback to the video button on click event.
        :param figure_index: Figure index.
        """
        if not len(self._figures):
            return

        images = []
        scenario_key = self._selected_scenario_keys[figure_index]
        scenario_name = scenario_key.scenario_name
        scenario_type = scenario_key.scenario_type
        planner_name = scenario_key.planner_name
        video_name = scenario_type + "_" + planner_name + "_" + scenario_name + ".avi"
        nuboard_file_index = scenario_key.nuboard_file_index
        video_path = (
            Path(self._experiment_file_data.file_paths[nuboard_file_index].simulation_main_path) / "video_screenshot"
        )
        if not video_path.exists():
            video_path.mkdir(parents=True, exist_ok=True)
        video_save_path = video_path / video_name

        scenario = self.figures[figure_index].scenario
        database_interval = scenario.database_interval
        selected_simulation_figure = self._figures[figure_index]

        try:
            if len(selected_simulation_figure.ego_state_plot.data_sources):
                chrome_options = webdriver.ChromeOptions()
                chrome_options.headless = True
                driver = webdriver.Chrome(chrome_options=chrome_options)
                driver.set_window_size(1920, 1080)
                shape = None
                simulation_figure = self._create_initial_figure(
                    figure_index=figure_index,
                    backend="canvas",
                    figure_sizes=simulation_tile_style["render_figure_sizes"],
                )
                # Copy the data sources
                simulation_figure.copy_datasources(selected_simulation_figure)
                self._render_scenario(main_figure=simulation_figure)
                length = len(selected_simulation_figure.ego_state_plot.data_sources)
                for frame_index in tqdm(range(length), desc="Rendering video"):
                    self._render_plots(main_figure=simulation_figure, frame_index=frame_index)
                    image = get_screenshot_as_png(column(simulation_figure.figure), driver=driver)
                    shape = image.size
                    images.append(image)
                    label = f"Rendering video now... ({frame_index}/{length})"
                    self._doc.add_next_tick_callback(
                        partial(self._update_video_button_label, figure_index=figure_index, label=label)
                    )

                fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
                if database_interval:
                    fps = 1 / database_interval
                else:
                    fps = 20  # Assume default fps is 20
                video_obj = cv2.VideoWriter(filename=str(video_save_path), fourcc=fourcc, fps=fps, frameSize=shape)
                for index, image in enumerate(images):
                    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    video_obj.write(cv2_image)

                video_obj.release()
                logger.info("Video saved to %s" % str(video_save_path))
        except (RuntimeError, Exception) as e:
            logger.warning("%s" % e)

        self._doc.add_next_tick_callback(partial(self._reset_video_button, figure_index=figure_index))

    def _first_button_on_click(self, figure_index: int) -> None:
        """
        Will be called when the first button is clicked.
        :param figure_index: The SimulationFigure index to render.
        """
        figure = self._figures[figure_index]
        self._request_specific_frame(figure=figure, frame_index=0)

    def _prev_button_on_click(self, figure_index: int) -> None:
        """
        Will be called when the prev button is clicked.
        :param figure_index: The SimulationFigure index to render.
        """
        figure = self._figures[figure_index]
        self._request_previous_frame(figure)

    def _play_button_on_click(self, figure_index: int) -> None:
        """
        Will be called when the play button is clicked.
        :param figure_index: The SimulationFigure index to render.
        """
        figure = self._figures[figure_index]
        self._process_play_request(figure)

    def _next_button_on_click(self, figure_index: int) -> None:
        """
        Will be called when the next button is clicked.
        :param figure_index: The SimulationFigure index to render.
        """
        figure = self._figures[figure_index]
        self._request_next_frame(figure)

    def _last_button_on_click(self, figure_index: int) -> None:
        """
        Will be called when the last button is clicked.
        :param figure_index: The SimulationFigure index to render.
        """
        figure = self._figures[figure_index]
        self._request_specific_frame(figure=figure, frame_index=len(figure.simulation_history.data) - 1)

    def _slider_on_change(self, attr: str, old: int, frame_index: int, figure_index: int) -> None:
        """
        The function that's called every time the slider's value has changed.
        All frame requests are routed through slider's event handling since currently there's no way to manually
        set the slider's value programatically (to sync the slider value) without triggering this event.
        :param attr: Attribute name.
        :param old: Old value.
        :param frame_index: The new value of the slider, which is the requested frame index.
        :param figure_index: Figure index.
        """
        del attr, old  # unused params
        selected_figure = self._figures[figure_index]
        self._request_plot_rendering(figure=selected_figure, frame_index=frame_index)

    def _request_specific_frame(self, figure: SimulationFigure, frame_index: int) -> None:
        """
        Requests to render the previous frame of the specified SimulationFigure.
        :param figure: The SimulationFigure render.
        :param frame_index: The frame index to render
        """
        figure.slider.value = frame_index

    def _request_previous_frame(self, figure: SimulationFigure) -> None:
        """
        Requests to render the previous frame of the specified SimulationFigure.
        :param figure: The SimulationFigure render.
        """
        if self._current_frame_index > 0:
            figure.slider.value = self._current_frame_index - 1

    def _request_next_frame(self, figure: SimulationFigure) -> bool:
        """
        Requests to render next frame of the specified SimulationFigure.
        :param figure: The SimulationFigure render.
        :return True if the request is valid, False otherwise.
        """
        result = False

        if self._current_frame_index < len(figure.simulation_history.data) - 1:
            figure.slider.value = self._current_frame_index + 1
            result = True

        return result

    def _request_plot_rendering(self, figure: SimulationFigure, frame_index: int) -> None:
        """
        Request the SimulationTile to render a frame of the plot. The requested frame will be enqueued if frame rate cap
        is reached or the figure is currently rendering a frame.
        :param figure: The SimulationFigure to render.
        :param frame_index: The requested frame index to render.
        """
        current_time = time.time()
        if current_time - self._last_frame_time < self._minimum_frame_time_seconds or figure.is_rendering():
            logger.info("Frame deferred: %d", frame_index)
            self._plot_render_queue = (figure, frame_index)
        else:
            self._process_plot_render_request(figure=figure, frame_index=frame_index)
            self._last_frame_time = time.time()

    def _stop_playback(self, figure: SimulationFigure) -> None:
        """
        Stops the playback for the given figure.
        :param figure: SimulationFigure to stop rendering.
        """
        if self._playback_callback_handle:
            self._doc.remove_periodic_callback(self._playback_callback_handle)
            self._playback_callback_handle = None
            figure.play_button.label = "play"

    def _start_playback(self, figure: SimulationFigure) -> None:
        """
        Starts the playback for the given figure.
        :param figure: SimulationFigure to stop rendering.
        """
        callback_period_seconds = figure.simulation_history.interval_seconds
        callback_period_seconds = max(self._minimum_frame_time_seconds, callback_period_seconds)
        callback_period_ms = 1000.0 * callback_period_seconds
        self._playback_callback_handle = self._doc.add_periodic_callback(
            partial(self._playback_callback, figure), callback_period_ms
        )
        figure.play_button.label = "stop"

    def _playback_callback(self, figure: SimulationFigure) -> None:
        """The callback that will advance the simulation frame. Will automatically stop the playback once we reach the final frame."""
        if not self._request_next_frame(figure):
            self._stop_playback(figure)

    def _process_play_request(self, figure: SimulationFigure) -> None:
        """
        Processes play request. When play mode is activated, the frame auto-advances, at the rate of the currently set frame rate cap.
        :param figure: The SimulationFigure to render.
        """
        if self._playback_callback_handle:
            self._stop_playback(figure)
        else:
            self._start_playback(figure)

    def _process_plot_render_request(self, figure: SimulationFigure, frame_index: int) -> None:
        """
        Process plot render requests, coming either from the slider or the render queue.
        :param figure: The SimulationFigure to render.
        :param frame_index: The requested frame index to render.
        """
        if frame_index != len(figure.simulation_history.data):
            if self._async_rendering:
                thread = threading.Thread(
                    target=self._render_plots,
                    kwargs={'main_figure': figure, 'frame_index': frame_index},
                    daemon=True,
                )
                thread.start()
            else:
                self._render_plots(main_figure=figure, frame_index=frame_index)

    def _render_scenario(self, main_figure: SimulationFigure, hidden_glyph_names: Optional[List[str]] = None) -> None:
        """
        Render scenario.
        :param main_figure: Simulation figure object.
        :param hidden_glyph_names: A list of glyph names to be hidden.
        """
        if self._async_rendering:
            # Spawn 2 child threads to load the data for plots and render them once they are available.
            # We don't wait for the threads to join so this function can immediately return, shortening the time the
            # loading indicator is shown to the user.

            def render() -> None:
                """Wrapper for the non-map-dependent parts of the rendering logic."""
                main_figure.update_data_sources()
                self._render_expert_trajectory(main_figure=main_figure)

                mission_goal = main_figure.scenario.get_mission_goal()
                if mission_goal is not None:
                    main_figure.render_mission_goal(mission_goal_state=mission_goal)

                self._render_plots(main_figure=main_figure, frame_index=0, hidden_glyph_names=hidden_glyph_names)

            def render_map_dependent() -> None:
                """Wrapper for the map-dependent parts of the rendering logic."""
                self._load_map_data(main_figure=main_figure)
                main_figure.update_map_dependent_data_sources()
                self._render_map(main_figure=main_figure)

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            executor.submit(render)
            executor.submit(render_map_dependent)
            executor.shutdown(wait=False)
        else:
            # Trigger background threads to fetch non-map dependent data
            main_figure.update_data_sources()

            # Load map data and then trigger thread(s) to fetch data sources that depend on it
            self._load_map_data(main_figure=main_figure)
            main_figure.update_map_dependent_data_sources()

            # Render the scenario
            self._render_map(main_figure=main_figure)

            self._render_expert_trajectory(main_figure=main_figure)

            mission_goal = main_figure.scenario.get_mission_goal()
            if mission_goal is not None:
                main_figure.render_mission_goal(mission_goal_state=mission_goal)

            self._render_plots(main_figure=main_figure, frame_index=0, hidden_glyph_names=hidden_glyph_names)

    def _load_map_data(self, main_figure: SimulationFigure) -> None:
        """
        Load the map data of the simulation tile.
        :param main_figure: Simulation figure.
        """
        # Load map data
        map_name = main_figure.scenario.map_api.map_name
        map_api = self._map_api(map_name)
        layer_names = [
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.LANE,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.STOP_LINE,
            SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CARPARK_AREA,
        ]

        assert main_figure.simulation_history.data, "No simulation history samples, unable to render the map."
        ego_pose = main_figure.simulation_history.data[0].ego_state.center
        center = Point2D(ego_pose.x, ego_pose.y)

        self._nearest_vector_map = map_api.get_proximal_map_objects(center, self._radius, layer_names)
        # Filter out stop polygons in turn stop
        if SemanticMapLayer.STOP_LINE in self._nearest_vector_map:
            stop_polygons = self._nearest_vector_map[SemanticMapLayer.STOP_LINE]
            self._nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
                stop_polygon for stop_polygon in stop_polygons if stop_polygon.stop_line_type != StopLineType.TURN_STOP
            ]

        # Populate our figure's lane connectors. This variable is required by traffic_light_plot.update_data_sources
        main_figure.lane_connectors = {
            lane_connector.id: lane_connector
            for lane_connector in self._nearest_vector_map[SemanticMapLayer.LANE_CONNECTOR]
        }

    def _render_map_polygon_layers(self, main_figure: SimulationFigure) -> None:
        """Renders the polygon layers of the map."""
        polygon_layer_names = [
            (SemanticMapLayer.LANE, simulation_map_layer_color[SemanticMapLayer.LANE]),
            (SemanticMapLayer.INTERSECTION, simulation_map_layer_color[SemanticMapLayer.INTERSECTION]),
            (SemanticMapLayer.STOP_LINE, simulation_map_layer_color[SemanticMapLayer.STOP_LINE]),
            (SemanticMapLayer.CROSSWALK, simulation_map_layer_color[SemanticMapLayer.CROSSWALK]),
            (SemanticMapLayer.WALKWAYS, simulation_map_layer_color[SemanticMapLayer.WALKWAYS]),
            (SemanticMapLayer.CARPARK_AREA, simulation_map_layer_color[SemanticMapLayer.CARPARK_AREA]),
        ]
        roadblock_ids = main_figure.scenario.get_route_roadblock_ids()
        if roadblock_ids:
            polygon_layer_names.append(
                (SemanticMapLayer.ROADBLOCK, simulation_map_layer_color[SemanticMapLayer.ROADBLOCK])
            )

        for layer_name, color in polygon_layer_names:
            map_polygon = MapPoint(point_2d=[])
            # Render RoadBlock
            if layer_name == SemanticMapLayer.ROADBLOCK:
                layer = (
                    self._nearest_vector_map[SemanticMapLayer.LANE]
                    + self._nearest_vector_map[SemanticMapLayer.LANE_CONNECTOR]
                )
                for map_obj in layer:
                    roadblock_id = map_obj.get_roadblock_id()
                    if roadblock_id in roadblock_ids:
                        coords = map_obj.polygon.exterior.coords
                        points = [Point2D(x=x, y=y) for x, y in coords]
                        map_polygon.point_2d.append(points)
            else:
                layer = self._nearest_vector_map[layer_name]
                for map_obj in layer:
                    coords = map_obj.polygon.exterior.coords
                    points = [Point2D(x=x, y=y) for x, y in coords]
                    map_polygon.point_2d.append(points)

            polygon_source = ColumnDataSource(
                dict(
                    xs=map_polygon.polygon_xs,
                    ys=map_polygon.polygon_ys,
                )
            )
            layer_map_polygon_plot = main_figure.figure.multi_polygons(
                xs="xs",
                ys="ys",
                fill_color=color["fill_color"],
                fill_alpha=color["fill_color_alpha"],
                line_color=color["line_color"],
                source=polygon_source,
            )
            # underlay = default rendering level for grids, one level below `glyph`, the default level for plots
            # https://docs.bokeh.org/en/latest/docs/user_guide/styling/plots.html#setting-render-levels
            layer_map_polygon_plot.level = "underlay"
            main_figure.map_polygon_plots[layer_name.name] = layer_map_polygon_plot

    def _render_map_line_layers(self, main_figure: SimulationFigure) -> None:
        """Renders the line layers of the map."""
        line_layer_names = [
            (SemanticMapLayer.LANE, simulation_map_layer_color[SemanticMapLayer.BASELINE_PATHS]),
            (SemanticMapLayer.LANE_CONNECTOR, simulation_map_layer_color[SemanticMapLayer.LANE_CONNECTOR]),
        ]
        for layer_name, color in line_layer_names:
            layer = self._nearest_vector_map[layer_name]
            map_line = MapPoint(point_2d=[])
            for map_obj in layer:
                path = map_obj.baseline_path.discrete_path
                points = [Point2D(x=pose.x, y=pose.y) for pose in path]
                map_line.point_2d.append(points)

            line_source = ColumnDataSource(dict(xs=map_line.line_xs, ys=map_line.line_ys))
            layer_map_line_plot = main_figure.figure.multi_line(
                xs="xs",
                ys="ys",
                line_color=color["line_color"],
                line_alpha=color["line_color_alpha"],
                line_width=0.5,
                line_dash="dashed",
                source=line_source,
            )
            # underlay = default rendering level for grids, one level below `glyph`, the default level for plots
            # https://docs.bokeh.org/en/latest/docs/user_guide/styling/plots.html#setting-render-levels
            layer_map_line_plot.level = "underlay"
            main_figure.map_line_plots[layer_name.name] = layer_map_line_plot

    def _render_map(self, main_figure: SimulationFigure) -> None:
        """
        Render a map.
        :param main_figure: Simulation figure.
        """

        def render() -> None:
            """Wrapper for the actual render logic, for multi-threading compatibility."""
            self._render_map_polygon_layers(main_figure)
            self._render_map_line_layers(main_figure)

        self._doc.add_next_tick_callback(lambda: render())

    @staticmethod
    def _render_expert_trajectory(main_figure: SimulationFigure) -> None:
        """
        Render expert trajectory.
        :param main_figure: Main simulation figure.
        """
        expert_ego_trajectory = main_figure.scenario.get_expert_ego_trajectory()
        source = extract_source_from_states(expert_ego_trajectory)
        main_figure.render_expert_trajectory(expert_ego_trajectory_state=source)

    def _render_plots(
        self, main_figure: SimulationFigure, frame_index: int, hidden_glyph_names: Optional[List[str]] = None
    ) -> None:
        """
        Render plot with a frame index.
        :param main_figure: Main figure to render.
        :param frame_index: A frame index.
        :param hidden_glyph_names: A list of glyph names to be hidden.
        """
        # main_figure.lane_connectors might still be loading the first time the function is called (for the initial
        # frame), but if the user renders another frame and go back, it should be there if it's available for the tile.
        if main_figure.lane_connectors is not None and len(main_figure.lane_connectors):
            main_figure.traffic_light_plot.update_plot(
                main_figure=main_figure.figure,
                frame_index=frame_index,
                doc=self._doc,
            )

        # Update ego state plot.
        main_figure.ego_state_plot.update_plot(
            main_figure=main_figure.figure,
            frame_index=frame_index,
            radius=self._radius,
            doc=self._doc,
        )

        # Update ego pose trajectory state plot.
        main_figure.ego_state_trajectory_plot.update_plot(
            main_figure=main_figure.figure,
            frame_index=frame_index,
            doc=self._doc,
        )

        if main_figure.future_expert_trajectory_plot:
            # Update future expert trajectory plot.
            main_figure.future_expert_trajectory_plot.update_plot(
                main_figure=main_figure.figure,
                frame_index=frame_index,
                doc=self._doc,
            )

        # Update selected anchor trajectory plot.
        main_figure.selected_anchor_trajectory_plot.update_plot(
            main_figure=main_figure.figure,
            frame_index=frame_index,
            doc=self._doc,
        )

        # Update agent state plot.
        main_figure.agent_state_plot.update_plot(
            main_figure=main_figure.figure,
            frame_index=frame_index,
            doc=self._doc,
        )

        # Update agent heading plot.
        main_figure.agent_state_heading_plot.update_plot(
            main_figure=main_figure.figure,
            frame_index=frame_index,
            doc=self._doc,
        )

        def update_decorations() -> None:
            main_figure.figure.title.text = main_figure.figure_title_name_with_timestamp(frame_index=frame_index)
            main_figure.update_glyphs_visibility(glyph_names=hidden_glyph_names)

        self._doc.add_next_tick_callback(lambda: update_decorations())

        self._last_frame_index = self._current_frame_index
        self._current_frame_index = frame_index

    def _periodic_callback(self) -> None:
        """Periodic callback registered to the bokeh.Document."""
        # Process plot render queue
        if self._plot_render_queue:
            figure, frame_index = self._plot_render_queue

            # Prevent the frame from jumping back after settling
            last_frame_direction = math.copysign(1, self._current_frame_index - self._last_frame_index)
            request_frame_direction = math.copysign(1, frame_index - self._current_frame_index)
            if request_frame_direction != last_frame_direction:
                logger.info("Frame dropped %d", frame_index)
                self._plot_render_queue = None

            # Render the queue if it made it this far
            elif not figure.is_rendering():
                logger.info("Processing render queue for frame %d", frame_index)
                self._plot_render_queue = None
                self._process_plot_render_request(figure=figure, frame_index=frame_index)
