"""
 __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
|                                               |
|   Copyright (C) 2021, All rights reserved.    |
|   "THE GOOSE IS LOOSE"                        |
|   An agent to play the game Hungry Geese      |
|   using a reinforcement rrained network with  |
|   a lower cutoff bound search with a          |
|   value net and floodfill scoring as the      |
|   engine heuristic.                           |
|__ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __|

"""

# --------------------------------------------------
# utils.py

from kaggle_environments.envs.hungry_geese.hungry_geese import Observation as KaggleObservation, Configuration, Action, adjacent_positions
import getpass, sys, os, time
import numpy as np
import collections
import copy

USER = getpass.getuser()
LOCAL_MODE = USER in ['taaha', 'rob', 'anton']
VERBOSE = True

# Only use danger moves if 90 minutes have passed since submission
INIT_SAFETY_TIME = 1627329128 # Hardcode to time.time() of submission
CAUTIOUS = not ((time.time() - INIT_SAFETY_TIME > 5400 and not LOCAL_MODE) or LOCAL_MODE)

options = {

    # Detection parameters
    'CHECK_HRL': not LOCAL_MODE,          # Detect PubHRL agent movement
    'CHECK_AVOID_DANGER': not CAUTIOUS,   # Detect which agents don't use head positions

    # Net parameters
    'REMOVE_FOOD_TTA': False,             # Include TTA inferences with removed food layer
    'USE_POLICY': True,                   # Use Policy net when lookahead is inconclusive
    'CONFIDENT_POLICY': not CAUTIOUS,     # Use Policy net when confidence is more than 0.95

    # Tree parameters
    'MAX_MEAN': False,            # Anton idea - instead of min-max tree, use mean-max (slower, better analysis)
    'MASK_OPPONENTS': True,       # Mask distant agents to simplify tree (faster, slightly unstable)
    'GREEDY_MODE': False          # Become greedy during 1v1 endgames

}

if USER == 'taaha':  RESOURCE_DIR = 'agent/'
elif USER == 'rob':  RESOURCE_DIR = 'agent/'
elif not LOCAL_MODE: RESOURCE_DIR = '/kaggle_simulations/agent/'

# configure onnx for pinning to a single cpu
sys.path.append(RESOURCE_DIR)
os.environ['OMP_NUM_THREADS'] = '1'
import onnxruntime
opts = onnxruntime.SessionOptions() 
opts.inter_op_num_threads = 1 
opts.intra_op_num_threads = 1 
opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL 

WEIGHTS_DIR = 'weights/'

VALUE_FILE = 'run26_36090_opset13_d0617_bs4.onnx'

INF = float('inf')
EPS = 5e-2
ACTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']
DIRECTION = {
    'NORTH': lambda pos: ((pos[0] + 6) % 7, pos[1]),
    'SOUTH': lambda pos: ((pos[0] + 8) % 7, pos[1]),
    'EAST' : lambda pos: (pos[0], (pos[1] + 12) % 11),
    'WEST' : lambda pos: (pos[0], (pos[1] + 10) % 11)
}

class dotdict:
    """ Dictionary accessable from dict.key """

    def __init__(self, entries):
        self.__dict__.update(**entries)
    
    def __str__(self):
        return str(self.__dict__)
    
options = dotdict(options)

def dist(pos1, pos2):
    """ (float): Get Toroidal distance between points
        Manhatten distance that accounts for board wrapping
    """

    if int in (type(pos1), type(pos2)):
        pos1, pos2 = row_col(pos1), row_col(pos2)

    cols, rows = 11, 7

    dx = abs(pos2[0] - pos1[0])
    dy = abs(pos2[1] - pos1[1])

    if dx > rows // 2: dx = rows - dx
    if dy > cols // 2: dy = cols - dy
    
    return dx + dy

def get_neighbors(pos, walls = []):
    """ (list): Getting neighbors of position given walls """
    x, y = pos
    r, c = 7, 11
    result = []
    for i in (-1, 1):
        pos1 = ((x + i + r) % r, y)
        pos2 = (x, (y + i + c) % c)
        if pos1 not in walls:
            result.append(pos1)
        if pos2 not in walls:
            result.append(pos2)
    return result

def active_indexes(obs, indexes = range(4)):
    """ (list): Gets active indexes of opponents """
    return [agent for agent in indexes if len(obs.geese[agent]) and agent != obs.index]

def length_advantage(obs, index):
    """ (int): 1st place length advantage for index """
    lengths = list(map(len, obs.geese))
    length = lengths.pop(index)
    return length - max(lengths)

def change_index(obses, index):
    """ (list): Change list of observations to another index """
    output = []
    for obs in obses:
        copy = dict(obs.copy())
        copy['index'] = index
        output.append(KaggleObservation(copy))
    return output

def row_col(position):
    """ (tuple): Get tuple coordinates of position from index """
    return (position // 11, position % 11)

def index_pos(position):
    """ (int): Get integer index of position from coordinates """
    return int(np.arange(77).reshape(7, 11)[position])

def get_key(obses):
    """ (tuple): Unique hashed key for position """
    hashable = tuple(map(tuple, obses[-1].geese + [obses[-1].food] + [[obses[-1].index]]))
    key = (obses[-1].step, hash(hashable))
    return key

def get_direction(start, end):
    """ (str): Gets NSEW direction from start to end point """
    for direction in DIRECTION:
        if DIRECTION[direction](start) == end:
            return direction

class DepthCache:
    """ 
    Cache with dynamically changing size based on depth storage.
    Optimized to lower memory usage by setting an upper and 
    lower bound to cache data.
    """

    def __init__(self):
        self.cache = collections.defaultdict(dict)
        self.info = {'hits': 0, 'misses': 0}
        self.max_entry = -INF

    def get(self, key):
        """ (cache_type | None): Gets value if stored in cache """

        # Key step is past upper-bound for cache storage
        if key[0] > self.max_entry:
            return None

        # Key not in cache for key step
        if key not in self.cache[key[0]]:
            self.info['misses'] += 1
            return None
        
        # Return saved info
        self.info['hits'] += 1
        return self.cache[key[0]][key]

    def set(self, key, value):
        """ (void): Sets key to value in cache """
        self.max_entry = max(self.max_entry, key[0])
        self.cache[key[0]][key] = value
    
    def size(self):
        """ (int): Amount of entries in cache """
        return sum(list(map(len, self.cache.values())))

    def cache_info(self):
        """ (dict): Information on hits and misses of cache """
        data = self.info.copy(); data['size'] = self.size()
        data['hit_rate'] = round(data['hits'] / data['misses'], 3) if data['misses'] else 0.0
        return data

    def update(self, min_entry):
        """ (void): Updating lower-bound for cache storage """
        above = { key: values for key, values in self.cache.items() if key >= min_entry }
        self.cache = collections.defaultdict(dict, above)

SCORE_CACHE = collections.defaultdict(DepthCache)

def wrapper_cache(func):
    """ (function): Heuristic wrapper to cache outputs """

    global SCORE_CACHE
    name = str(func.__name__)

    def inner(obses, mode = 'SCORE', tta = None):
        if mode == 'TEST':
            return func(obses, mode, tta = tta)

        global SCORE_CACHE
        key = get_key(obses)
        entry = SCORE_CACHE[name].get(key)
        if entry != None:
            if mode in entry:
                return entry[mode]

        entry = entry.copy() if entry != None else {}
        output = func(obses, mode, tta = tta)
        entry[mode] = output

        SCORE_CACHE[name].set(key, entry)
        return output

    return inner

# Anton Libraries -------------------------------------------------

int_to_direction_name = {0: "NORTH", 1: "EAST", 2: "SOUTH", 3: "WEST"} 
direction_to_int = {'NORTH': 0, 'EAST': 1, 'SOUTH': 2, 'WEST': 3}

trans_action_map = {
    (-1, 0): Action.NORTH,
    (1, 0): Action.SOUTH,
    (0, 1): Action.EAST,
    (0, -1): Action.WEST,
}

trans_action_map_inv = {
    "NORTH": (-1, 0),
    "SOUTH": (1, 0),
    "EAST": (0, 1),
    "WEST": (0, -1)
}

import contextlib

@contextlib.contextmanager
def temp_dir():
    import tempfile
    import shutil
    dir_path = tempfile.mkdtemp()
    try:
        yield dir_path
    finally:
        shutil.rmtree(dir_path)

with temp_dir() as working_dir:

    def write_file(path, contents):

        def make_package(path):
            parts = path.split("/")
            partial_path = working_dir
            for part in parts:
                partial_path = os.path.join(partial_path, part)
                if not os.path.exists(partial_path):
                    os.mkdir(partial_path)
                    with open(os.path.join(partial_path, "__init__.py"), "w") as f:
                        f.write("\n")

        make_package(os.path.dirname(path))

        full_path = os.path.join(working_dir, path)
        with open(full_path, "w") as module_file:
            module_file.write(contents)

    sys.path.insert(0, working_dir)
    
    write_file('anton/lib/geometry.py', 'from typing import Dict, List, NamedTuple, Set, Tuple\n\nfrom kaggle_environments.envs.hungry_geese.hungry_geese import Action\n\nfrom .color_print import pos_color\n\n\nTranslation = Tuple[int, int]\n\n# TODO: should be Literal type but not supported on kaggle Python 3.7\ntrans_action_map: Dict[Translation, Action] = {\n    (-1, 0): Action.NORTH,\n    (1, 0): Action.SOUTH,\n    (0, 1): Action.EAST,\n    (0, -1): Action.WEST,\n}\n\n\nclass Pos(NamedTuple):\n    x: int\n    y: int\n\n    def __repr__(self):\n        return str(pos_color(f"[{self.x:X}:{self.y:X}]"))\n\n\nclass Geometry:\n    def __init__(self, size_x, size_y):\n        self.size_x = size_x\n        self.size_y = size_y\n\n    @property\n    def shape(self) -> Tuple[int, int]:\n        return (self.size_x, self.size_y)\n\n    def prox(self, pos: Pos, *, dists=(1,)) -> Set[Pos]:\n        translations = self.get_translations(dists)\n\n        return {self.translate(pos, translation) for translation in translations}\n\n    def get_translations(self, dists) -> List[Tuple[int, int]]:\n        result = []\n        for dist in dists:\n            if dist == 0:\n                result.append((0, 0))\n                continue\n\n            for d in range(dist):\n                result.append((d, dist - d))\n                result.append((-d, -dist + d))\n                result.append((-dist + d, +d))\n                result.append((dist - d, -d))\n\n        return result\n\n    def translate(self, pos: Pos, diff: Translation) -> Pos:\n        x, y = pos\n        dx, dy = diff\n        return Pos((x + dx) % self.size_x, (y + dy) % self.size_y)\n\n    def diff_to(self, pos1: Pos, pos2: Pos) -> Translation:\n        dx = pos2.x - pos1.x\n        dy = pos2.y - pos1.y\n\n        if dx <= self.size_x // 2:\n            dx += self.size_x\n\n        if dx > self.size_x // 2:\n            dx -= self.size_x\n\n        if dy <= self.size_y // 2:\n            dy += self.size_y\n\n        if dy > self.size_y // 2:\n            dy -= self.size_y\n\n        return (dx, dy)\n\n    def action_to(self, pos1, pos2):\n        diff = self.diff_to(pos1, pos2)\n\n        result = trans_action_map.get(diff)\n\n        if result is None:\n            raise ValueError(f"Cannot step from {pos1} to {pos2}")\n\n        return result\n')
    write_file('anton/lib/color_print.py', 'try:\n    import colorful\n\n    colorful.use_true_colors()  # type:ignore\n\n    pos_color = colorful.goldenrod1  # type: ignore\nexcept ImportError:\n    pos_color = lambda x: x\n')
    write_file('anton/lib/observation.py', 'import dataclasses\nimport itertools\nfrom dataclasses import dataclass\nfrom typing import Dict, List, Set, Optional\n\nfrom .geometry import Geometry, Pos\n\n\n@dataclass\nclass Goose:\n    poses: List[Pos]\n    head: Pos = dataclasses.field(init=False)\n\n    def __post_init__(self):\n        if not self.poses:\n            raise ValueError("Cannot have empty Goose")\n\n        self.head = self.poses[0]\n\n    def __repr__(self):\n        return f"Goose({self.head}<" + "-".join(map(str, self.poses[1:])) + ")"\n\n    def __len__(self):\n        return len(self.poses)\n\n\ndef field_idx_to_pos(field_idx: int, *, num_cols: int, num_rows: int) -> Pos:\n    x = field_idx // num_cols\n    y = field_idx % num_cols\n\n    if not (0 <= x < num_rows and 0 <= y < num_cols):\n        raise ValueError("Illegal field_idx {field_idx} with x={x} and y={y}")\n\n    return Pos(x, y)\n\n\n@dataclass(frozen=True)\nclass Observation:\n    """\n    Translates kaggle obs and conf into a processed format\n    """\n\n    geese: Dict[int, Goose]\n    geo: Geometry\n    my_index: int = 0\n    step: int = 0\n    food: Set[Pos] = dataclasses.field(default_factory=set)\n    my_goose: Optional[Goose] = dataclasses.field(init=False)\n    enemy_geese: List[Goose] = dataclasses.field(init=False)\n\n    def __post_init__(self):\n        object.__setattr__(self, "my_goose", self.geese.get(self.my_index, None))\n\n        object.__setattr__(\n            self,\n            "enemy_geese",\n            [goose for idx, goose in self.geese.items() if idx != self.my_index],\n        )\n\n    @classmethod\n    def from_obs_conf(cls, obs, conf={"rows": 7, "columns": 11}, *, my_index=None):\n        num_cols = conf["columns"]\n        num_rows = conf["rows"]\n        step = obs["step"]\n        my_index = obs["index"] if my_index is None else my_index\n\n        geese = {\n            idx: Goose(\n                poses=[\n                    field_idx_to_pos(idx, num_cols=num_cols, num_rows=num_rows)\n                    for idx in goose_data\n                ]\n            )\n            for idx, goose_data in enumerate(obs["geese"])\n            if goose_data\n        }\n\n        all_goose_poses = set(\n            itertools.chain.from_iterable(goose.poses for goose in geese.values())\n        )  # TODO: why food below goose?\n\n        food = {\n            field_idx_to_pos(idx, num_cols=num_cols, num_rows=num_rows)\n            for idx in obs["food"]\n        } - all_goose_poses\n\n        return cls(\n            food=food,\n            geese=geese,\n            my_index=my_index,\n            step=step,\n            geo=Geometry(size_x=num_rows, size_y=num_cols),\n        )\n\n    def __repr__(self):\n        return (\n            f"Observation(step:{self.step}, index:{self.my_index}, Geese("\n            + ",".join(f"{idx}:{len(goose.poses)}" for idx, goose in self.geese.items())\n            + f"), food:{len(self.food)})"\n        )\n\n')
    write_file('anton/lib/__init__.py', 'from .runlib import *\nfrom .plot import *\nfrom .download import *\nfrom .features import *\nfrom .scoreobs import *\nfrom .tree import *\nfrom .score_tree import *\nfrom .replay import *\n')
    write_file('anton/lib/runlib.py', 'import random\nimport datetime as dt\n\nimport numpy as np\nfrom kaggle_environments import make\n\n\ndef run(bots, steps=None, do_render=True, seed=123, scale=0.5):\n    if seed is None:\n        seed = int(dt.datetime.timestamp(dt.datetime.now()))\n    random.seed(seed)\n    np.random.seed(seed)\n\n    env = make("hungry_geese", debug=True, configuration=dict(episodeSteps=steps))\n    env.reset()\n    env.run(bots)\n\n    if do_render:\n        render(env, scale=scale)\n\n    return env\n\n\ndef run_train(bot, other_bots=["random", "random", "random"], steps=200, seed=123):\n    if seed is None:\n        seed = int(dt.datetime.timestamp(dt.datetime.now()))\n    random.seed(seed)\n    np.random.seed(seed)\n\n    env = make("hungry_geese", debug=True)\n\n    trainer = env.train([None] + other_bots)\n\n    obs = trainer.reset()  # type: ignore\n    for _ in range(steps):\n        action = bot(obs, env.configuration)\n        obs, reward, done, info = trainer.step(action)  # type: ignore\n\n        if done:\n            break\n\n    return env\n\n\ndef render(env, scale=0.5):\n    env.render(mode="ipython", width=int(800 * scale), height=int(700 * scale))\n\n')
    write_file('anton/lib/plot.py', 'from typing import List, Optional, Tuple, Union\nimport copy\n\nimport matplotlib as mpl\nimport matplotlib.pyplot as plt\nimport numpy as np\nfrom kaggle_environments import make\n\nfrom .geometry import Pos\nfrom .observation import Observation\n\ngoose_colors = ["#fcfcfc", "#68dcfb", "#7bdf4f", "#ec7e79"]\n\n\nflood_diff_format = dict(\n    cmap="coolwarm_r",\n    imshow_params=dict(norm=mpl.colors.TwoSlopeNorm(0, vmin=-5, vmax=5)),\n)\n\n\nclass CenterOn:\n    def __init__(self, x, y, size_x, size_y):\n        self.sub_x = x - size_x // 2\n        self.sub_y = y - size_y // 2\n        self.size_x = size_x\n        self.size_y = size_y\n\n    def roll_x(self, x, inv=False):\n        return (x - (-1 if inv else 1) * self.sub_x) % self.size_x\n\n    def roll_y(self, y, inv=False):\n        return (y - (-1 if inv else 1) * self.sub_y) % self.size_y\n\n    def roll_array(self, array):\n        array = np.roll(array, shift=-self.sub_x, axis=0)\n        array = np.roll(array, shift=-self.sub_y, axis=1)\n        return array\n\n\ndef showfield(\n    field,\n    *,\n    text_format="{:.0f}",\n    cmap="viridis_r",\n    missing=None,\n    imshow_params=None,\n    center=None,\n    ax=None,\n):\n    size_y, size_x = field.shape\n\n    center_on = center\n\n    if isinstance(center_on, tuple):\n        center_on = CenterOn(*center, *field.shape)\n\n    if isinstance(center_on, CenterOn):\n        field = center_on.roll_array(field)\n\n        xtick_labels = np.array([center_on.roll_y(i, inv=True) for i in range(size_x)])\n        ytick_labels = np.array([center_on.roll_x(i, inv=True) for i in range(size_y)])\n    else:\n        center_on = None  # type: ignore\n        xtick_labels = np.arange(size_x)\n        ytick_labels = np.arange(size_y)\n\n    cmap = copy.copy(mpl.cm.get_cmap(cmap))  # type: ignore\n    cmap.set_bad("lightgray")\n\n    if missing is not None:\n        field = np.where(field != missing, field, np.nan)\n\n    if ax is None:\n        fig, ax = plt.subplots()\n\n    ax.imshow(\n        field,\n        aspect="equal",\n        cmap=cmap,\n        **(imshow_params if imshow_params is not None else {}),\n    )\n\n    ax.set_xticks(np.arange(0, size_x, 1))\n    ax.set_yticks(np.arange(0, size_y, 1))\n\n    # Labels for major ticks\n    ax.set_xticklabels(xtick_labels)\n    ax.set_yticklabels(ytick_labels)\n\n    # Minor ticks\n    ax.set_xticks(np.arange(-0.5, size_x), minor=True)\n    ax.set_yticks(np.arange(-0.5, size_y), minor=True)\n\n    # Gridlines based on minor ticks\n    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)\n    ax.grid(False, which="major")\n\n    if text_format is not None:\n        for i in range(size_y):\n            for j in range(size_x):\n                cur_text = field[i, j]  # type: ignore\n                if not np.isnan(cur_text):\n                    text = ax.text(\n                        j,\n                        i,\n                        text_format.format(cur_text),\n                        ha="center",\n                        va="center",\n                        color="k",\n                    )\n\n    return ax\n\n\ndef render_env(env, scale=0.5):\n    env.render(mode="ipython", width=int(800 * scale), height=int(700 * scale))\n\n\ndef render_replay(replay, my_index=None):\n    print(f"Episode: {replay.epi_id}")\n\n    my_index = replay.my_index if my_index is None else my_index\n\n    if my_index is not None:\n        print(\n            f"My index: {my_index} (",\n            {0: "white", 1: "blue", 2: "green", 3: "red"}[my_index],\n            ")",\n            sep="",\n        )\n        print(f"Rank: {replay.my_rank}")\n        print(f"My last step: {replay.my_last_step}")\n\n    print("Final geese lengths:", list(map(len, replay.obses[-1]["geese"])))\n\n    env = make("hungry_geese", debug=True, steps=replay.json["steps"])\n    render_env(env)\n\n\ndef plot_obs(\n    obs: Observation,\n    *,\n    field=None,\n    field_plot_params=None,\n    center: Optional[Union[CenterOn, bool]] = None,\n):\n    if obs is not None and obs.my_goose is not None and center is True:\n        x, y = obs.my_goose.head\n        size_x, size_y = obs.geo.shape\n        center = CenterOn(x, y, size_x, size_y)\n    else:\n        center = None\n\n    if field is None:\n        field = np.zeros(shape=obs.geo.shape)\n\n    fig, ax = plt.subplots()\n\n    for goose_idx, goose in obs.geese.items():\n        poses: List[Tuple[Optional[int], Optional[int]]] = []\n        goose_poses = [\n            (center.roll_x(x), center.roll_y(y)) if center is not None else (x, y)\n            for x, y in goose.poses\n        ]\n\n        for i, (x, y) in enumerate(goose_poses):\n\n            poses.append((y, x))\n\n            if i + 1 < len(goose_poses):\n                next_x, next_y = goose_poses[i + 1]\n                if abs(next_x - x) + abs(next_y - y) > 1:\n                    poses.append((None, None))\n\n        color = goose_colors[goose_idx]\n\n        ax2 = ax.plot(*zip(*poses), c=color)\n\n        ax.plot(\n            *zip(*poses), linewidth=15, color=color,\n        )\n\n        ax.plot(\n            *zip(*poses[:1]),\n            marker="o",\n            markersize=25,\n            markerfacecolor=color,\n            markeredgecolor="none",\n        )\n\n        for i, pos2 in enumerate(poses):\n            if i + 2 < len(poses) and poses[i + 1] == (None, None):\n                next_pos2 = poses[i + 2]\n                dx, dy = obs.geo.diff_to(Pos(pos2[1], pos2[0]), Pos(next_pos2[1], next_pos2[0]))  # type: ignore\n                ax.plot(\n                    [pos2[0], pos2[0] + 0.5 * dy],  # type: ignore\n                    [pos2[1], pos2[1] + 0.5 * dx],  # type: ignore\n                    linewidth=15,\n                    color=color,\n                )\n                ax.plot(\n                    [next_pos2[0], next_pos2[0] - 0.5 * dy],  # type: ignore\n                    [next_pos2[1], next_pos2[1] - 0.5 * dx],  # type: ignore\n                    linewidth=15,\n                    color=color,\n                )\n\n    ax.plot(\n        *zip(\n            *[\n                (center.roll_y(pos.y), center.roll_x(pos.x))\n                if center is not None\n                else (pos.y, pos.x)\n                for pos in obs.food\n            ]\n        ),\n        marker="*",\n        linestyle="none",\n        markersize=30,\n        c="gray",\n    )\n\n    if field is not None:\n        if field_plot_params is None:\n            field_plot_params = {}\n        showfield(field, **field_plot_params, center=center, ax=ax)\n\n    return ax')
    write_file('anton/lib/download.py', 'import datetime as dt\nimport json\nimport os\nimport time\nfrom collections import deque\nfrom dataclasses import dataclass, field\nfrom typing import Any, Dict, List, Optional, Sequence\n\nimport pandas as pd\nimport requests\nfrom scipy.stats import rankdata\n\nfrom .filecache import FileCache\nfrom .replay import Replay, json_to_replay\n\nif "PROJECTDIR" in os.environ:\n    default_cache_directory: Optional[str] = os.environ[\n        "PROJECTDIR"\n    ] + "/data/download/"\nelse:\n    default_cache_directory = None\n\n\ndef default_make_cache(func, prefix):\n    return FileCache(\n        func,\n        directory=default_cache_directory,\n        key_func=lambda params: prefix\n        + "_".join(f"{arg}" for arg in params[0][1:])\n        + ".json",\n        store_func=lambda obj, file: json.dump(obj, file.open("w")),\n        read_func=lambda file: json.load(file.open()),\n    )\n\n\n@dataclass\nclass Submission:\n    id: int\n    epi_agents: pd.DataFrame\n    teams: pd.DataFrame\n    submissions: pd.DataFrame\n\n    def __repr__(self):\n        return f"Submissions({self.id}, {self.epi_agents.epi_id.nunique()} episodes)"\n\n    def epi_agent_names(self):\n        return (\n            self.epi_agents.groupby("epi_id")\n            .apply(lambda d: d.sort_values("agent_idx")["team_name"].tolist())\n            .to_dict()\n        )\n\n\ndef json_to_table(json_list, fields):\n    return pd.DataFrame(\n        [{field: json_data[field] for field in fields} for json_data in json_list]\n    )\n\n\nclass KaggleScraper:\n    """\n    .get_submission_episodes(submission_id) -> Submission\n    .get_replays(Submission, my_team_id) -> Dict[epi_id, Replay]\n\n    https://www.kaggle.com/c/halite/discussion/164932\n    """\n\n    base_url = "https://www.kaggle.com/requests/EpisodeService/"\n    list_url = base_url + "ListEpisodes"\n    replay_url = base_url + "GetEpisodeReplay"\n    max_requests_per_minute = 60\n    max_total_requests = 3600  # resets with each new import\n\n    def __init__(self, make_cache_func=default_make_cache):\n        self.num_requests = 0\n        self.last_request_times: "deque[dt.datetime]" = deque(\n            maxlen=self.max_requests_per_minute\n        )\n\n        if make_cache_func is not None:\n            self.list_episodes_for_submission = make_cache_func(\n                self._list_episodes_for_submission, "subm_"\n            )\n\n            self.get_episode_replay = make_cache_func(\n                self._get_episode_replay, "replay_"\n            )\n        else:\n            self.list_episodes_for_submission = self._list_episodes_for_submission\n            self.get_episode_replay = self._get_episode_replay\n\n    def _request_json(self, url, body):\n        if self.num_requests >= self.max_total_requests:\n            raise ValueError(f"Too many requests in total ({self.num_requests})")\n\n        timestamp_now = dt.datetime.now()\n        wait_seconds = int(\n            (timestamp_now - self.last_request_times[0]).total_seconds() - 60 + 1\n        )\n\n        if wait_seconds > 0:\n            print(\n                f"Waiting {wait_seconds} seconds due to limit of {self.max_requests_per_minute} requests per minute"\n            )\n            time.sleep(wait_seconds)\n            print("Continueing download")\n\n        response = requests.post(url, json=body)\n\n        self.num_requests += 1\n        timestamp_now = dt.datetime.now()\n        self.last_request_times.append(timestamp_now)\n\n        return response.json()\n\n    def _list_episodes_for_submission(self, submission_id: int):\n        return self._request_json(self.list_url, {"SubmissionId": submission_id})\n\n    def list_episodes(self, epi_ids: List[int]):\n        return self._request_json(self.list_url, {"Ids": epi_ids})\n\n    def _get_episode_replay(self, epi_id: int):\n        epi_id = int(epi_id)\n        return self._request_json(self.replay_url, {"EpisodeId": epi_id})\n\n    def get_submission_episodes(self, submission_id: int) -> Submission:\n        epi_data = self.list_episodes_for_submission(submission_id)\n\n        teams = (\n            json_to_table(epi_data["result"]["teams"], ["id", "teamName"])\n            .set_index("id")\n            .rename_axis(index="team_id")\n        )\n        if not teams["teamName"].is_unique:\n            print("Team names not unique")\n\n        submissions = (\n            json_to_table(epi_data["result"]["submissions"], ["id", "teamId"])\n            .set_index("id")\n            .rename_axis(index="submission_id")\n        )\n        submissions["teamName"] = submissions["teamId"].map(teams["teamName"])\n\n        epi_agents = pd.DataFrame(\n            [\n                {\n                    "epi_id": json_data["id"],\n                    "agent_id": agent_json_data["id"],\n                    "agent_idx": agent_json_data["index"],\n                    "agent_start_score": agent_json_data["initialScore"],\n                    "submission_id": agent_json_data["submissionId"],\n                }\n                for json_data in epi_data["result"]["episodes"]\n                for agent_json_data in json_data["agents"]\n            ]\n        )\n\n        epi_agents["team_name"] = epi_agents["submission_id"].map(\n            submissions["teamName"]\n        )\n\n        epi_agents["team_id"] = epi_agents["submission_id"].map(submissions["teamId"])\n\n        return Submission(\n            epi_agents=epi_agents,\n            teams=teams,\n            submissions=submissions,\n            id=submission_id,\n        )\n\n    def get_replays(\n        self, submission: Submission, my_team_id: Optional[int] = None\n    ) -> Dict[int, Replay]:\n        epi_ids = submission.epi_agents.epi_id.unique()\n\n        if my_team_id is not None:\n            epi_index = (\n                submission.epi_agents.query("team_id==@my_team_id")\n                .groupby("epi_id")["agent_idx"]\n                .min()\n            )\n        else:\n            epi_index = None\n\n        result = {}\n        for epi_id in epi_ids:\n            replay = self.get_replay(\n                epi_id,\n                my_index=epi_index.get(epi_id) if epi_index is not None else None,\n            )\n\n            assert epi_id not in result\n\n            result[epi_id] = replay\n\n        return result\n\n    def get_replay(self, epi_id: int, my_index=None) -> Replay:\n        epi_id = int(epi_id)\n        replay = self.get_episode_replay(epi_id)\n        data = json.loads(replay["result"]["replay"])\n\n        return json_to_replay(data, my_index=my_index)\n')
    write_file('anton/lib/filecache.py', 'from pathlib import Path\nimport pickle\n\n\nclass FileCache:\n    def __init__(\n        self,\n        func,\n        directory,\n        key_func=str,\n        store_func=pickle.dump,\n        read_func=pickle.load,\n    ):\n        self.func = func\n        self.directory = Path(directory)\n        self.key_func = key_func\n        self.store_func = store_func\n        self.read_func = read_func\n\n    def _filename(self, params):\n        return self.directory.joinpath(self.key_func(params))\n\n    def __call__(self, *args, **kwargs):\n        filename = self._filename(params=(args, kwargs))\n\n        if filename.exists():\n            result = self.read_func(filename)\n        else:\n            result = self.func(*args, **kwargs)\n            self.store_func(result, filename)\n\n        return result\n\n')
    write_file('anton/lib/replay.py', 'import json\nfrom dataclasses import dataclass, field\nfrom pathlib import Path\nfrom typing import Any, Dict, List, Optional\n\nimport numpy as np\nimport pandas as pd\n\nfrom .observation import Observation\n\n\n@dataclass\nclass Replay:\n    id: str\n    epi_id: Optional[int]\n    rewards: List[int]\n    statuses: List[str]\n    obses: List[Any]\n    conf: Dict[str, Any]\n    actions: pd.DataFrame\n    json: Any\n    my_index: Optional[int] = None\n    my_rank: Optional[int] = field(init=False, default=None)\n    my_last_step: Optional[int] = field(init=False, default=None)\n\n    def __post_init__(self):\n\n        if self.my_index is not None:\n            try:\n                self.my_index = int(self.my_index)\n                self.my_rank = int(\n                    rankdata(  # type:ignore\n                        -np.array(self.rewards), method="min"\n                    )[\n                        self.my_index\n                    ]\n                )\n\n                my_index = self.my_index\n                self.my_last_step = self.actions.query(  # type: ignore\n                    "index==@my_index and status==\'ACTIVE\'"\n                )["step"].max()\n            except Exception:\n                self.my_last_step = None\n                self.my_rank = None\n\n    def __repr__(self):\n        return f"Replay({self.id}, {len(self.obses)} obses)"\n\n    def agent_step(self, agent, step, index=None):\n        obs = self.obses[step]\n\n        if index is not None:\n            obs["index"] = index\n        else:\n            obs["index"] = self.my_index\n\n        agent(obs, self.conf)\n\n    def obs(self, step, index=None) -> Observation:\n        obs = self.obses[step]\n\n        if index is not None:\n            obs["index"] = index\n        else:\n            obs["index"] = self.my_index if self.my_index is not None else 0\n\n        return Observation.from_obs_conf(obs, self.conf)\n\n\ndef read_replay(obj, my_index=None) -> Replay:\n    if hasattr(obj, "toJSON"):\n        return json_to_replay(obj.toJSON(), my_index=my_index)\n    elif isinstance(obj, (str, Path)):\n        return json_to_replay(json.load(open(obj)))\n    else:\n        return json_to_replay(obj)\n\n\ndef json_to_replay(data, my_index=None):\n    obses = [data[0]["observation"] for data in data["steps"]]\n\n    actions = pd.DataFrame(\n        [\n            {\n                "step": step,\n                "action": player_data["action"],\n                "index": player_data["observation"]["index"],\n                "status": player_data["status"],\n            }\n            for step, data in enumerate(data["steps"])\n            for player_data in data\n        ]\n    )\n\n    return Replay(\n        obses=obses,\n        actions=actions,\n        conf=data["configuration"],\n        id=data["id"],\n        epi_id=data["info"]["EpisodeId"] if data["info"] else None,\n        rewards=data["rewards"],\n        statuses=data["statuses"],\n        json=data,\n        my_index=my_index,\n    )\n\n\ndef env_to_replay(env, my_index=None):\n    return json_to_replay(env.toJSON(), my_index=my_index)\n')
    write_file('anton/lib/features.py', 'from collections import defaultdict\nfrom dataclasses import dataclass\nfrom typing import Dict, Set, Tuple\n\nimport numpy as np\n\nfrom .floodfill2 import calc_flood_fill\nfrom .observation import Observation\n\n\n@dataclass\nclass Features:\n    obs: Observation\n    free_on_step: np.ndarray\n    all_enemy_floodfill: np.ndarray\n    my_floodfill: np.ndarray\n    is_enemy: np.ndarray\n    food_field: np.ndarray\n    food_dist: float\n\n\ndef make_features(\n    obs: Observation, my_late_arrival=False, enemy_late_arrival=True\n) -> Features:\n    free_on_step = calc_free_on_step(obs)\n\n    all_enemy_floodfill = calc_all_enemy_floodfill(\n        obs=obs, free_on_step=free_on_step, late_arrival=enemy_late_arrival\n    )\n\n    my_floodfill = calc_my_floodfill(\n        obs=obs, free_on_step=free_on_step, late_arrival=my_late_arrival\n    )\n\n    food_field = calc_food_field(obs=obs, field_shape=obs.geo.shape)\n\n    food_dist = calc_food_dist(\n        obs=obs, food_field=food_field, my_floodfill=my_floodfill\n    )\n\n    is_enemy = calc_is_enemy(obs=obs)\n\n    return Features(\n        obs=obs,\n        free_on_step=free_on_step,\n        all_enemy_floodfill=all_enemy_floodfill,\n        my_floodfill=my_floodfill,\n        is_enemy=is_enemy,\n        food_field=food_field,\n        food_dist=food_dist,\n    )\n\n\ndef calc_all_enemy_floodfill(*, obs, free_on_step, late_arrival=True) -> np.ndarray:\n    if obs.enemy_geese:\n        result = calc_flood_fill(\n            free_on_step,\n            seeds={0: [enemy_goose.head for enemy_goose in obs.enemy_geese]},\n            min_test_dist=1,\n            late_arrival=late_arrival,\n        ).field_dist\n    else:\n        result = np.full(fill_value=np.Inf, shape=free_on_step.shape)\n\n    return result\n\n\ndef calc_is_enemy(*, obs: Observation) -> np.ndarray:\n    result = np.zeros(shape=obs.geo.shape)\n    for enemy_goose in obs.enemy_geese:\n        for enemy_pos in enemy_goose.poses:\n            result[enemy_pos] = 1\n\n    return result\n\n\ndef calc_my_floodfill(*, obs, free_on_step, late_arrival=False) -> np.ndarray:\n    # TODO: consider last_step\n    if obs.my_goose is not None:\n        result = calc_flood_fill(\n            free_on_step,\n            seeds={0: [obs.my_goose.head]},\n            min_test_dist=1,\n            late_arrival=late_arrival,\n        ).field_dist\n    else:\n        result = np.full(fill_value=np.Inf, shape=free_on_step.shape)\n\n    return result\n\n\ndef calc_food_dist(*, obs, food_field, my_floodfill) -> float:\n    return np.nanmin(  # type: ignore\n        np.where(food_field == 1, my_floodfill, np.nan)\n    )\n\n\ndef calc_food_field(*, obs: Observation, field_shape: Tuple[int, int]):\n    result = np.full(fill_value=0, shape=field_shape)\n    for food_pos in obs.food:\n        result[food_pos] = 1\n    return result\n\n\ndef calc_free_on_step(obs: Observation):\n    # enemy_tail_penalty = 3\n    field = np.full(fill_value=0, shape=obs.geo.shape)\n\n    for idx, goose in obs.geese.items():\n        for free_on_step, pos in enumerate(reversed(goose.poses), 1):\n            # if idx != obs.index:\n            #    free_on_step += free_on_step // enemy_tail_penalty\n            field[pos.x, pos.y] = free_on_step\n\n    return field\n\n\ndef calc_free_on_step2(obs: Observation):\n    """\n    With food enlengthening\n    """\n    max_food_see_dist = 2\n    enemy_tail_penalty = 3\n\n    field = np.full(fill_value=0, shape=obs.geo.shape)\n\n    for idx, goose in obs.geese.items():\n        for free_on_step, pos in enumerate(reversed(goose.poses), 1):\n            # if idx != obs.index:\n            #    free_on_step += free_on_step // enemy_tail_penalty\n            field[pos.x, pos.y] = free_on_step\n\n    # add enlengthening due to food\n    enemy_goose_food_dist: Dict[int, Set] = defaultdict(set)\n\n    for goose_idx, enemy_goose in obs.geese.items():\n        if goose_idx == obs.my_index:\n            continue\n\n        enemy_floodfill = calc_flood_fill(\n            field, seeds={0: [enemy_goose.head]}, min_test_dist=1\n        )\n\n        for dist, frontier in enemy_floodfill.frontiers.items():\n            if dist == 0:\n                continue\n            if any(pos in obs.food for pos in frontier):\n                enemy_goose_food_dist[goose_idx].add(dist)\n\n    for goose_idx, dists in enemy_goose_food_dist.items():\n        tail_dists = [d - i for i, d in enumerate(sorted(dists))]\n        for dist in tail_dists:\n            last_i = -dist + 1\n            for pos in obs.geese[goose_idx].poses[: last_i if last_i != 0 else None]:\n                field[pos.x, pos.y] += 1\n\n    return field\n')
    write_file('anton/lib/floodfill2.py', 'from dataclasses import dataclass\nfrom typing import Dict, List, Set, Tuple\nfrom collections import defaultdict\n\nimport numpy as np\n\n\n@dataclass\nclass FloodfillResult:\n    field_dist: np.ndarray\n    frontiers: Dict[int, Set[int]]\n\n\ndef calc_flood_fill(\n    free_on_step: np.ndarray,\n    seeds: Dict[int, List[Tuple[int, int]]],\n    late_arrival: bool = False,\n    fill_value=np.Inf,\n    min_test_dist=0,\n    max_test_dist=np.Inf,\n) -> FloodfillResult:\n    if not seeds:\n        raise ValueError("Empty seeds")\n\n    size_x, size_y = free_on_step.shape\n    max_idx = size_x * size_y\n\n    free_on_step_flat = free_on_step.reshape(-1)  # type: ignore\n\n    field_dist: np.ndarray = np.full_like(  # type: ignore\n        free_on_step_flat, fill_value=fill_value, dtype=np.float64\n    )\n\n    test_frontiers: Dict[int, Set[int]] = defaultdict(set)\n\n    for dist, poses in seeds.items():\n        for x, y in poses:\n            test_frontiers[dist].add(y + x * size_y)\n\n    frontiers: Dict[int, Set[int]] = defaultdict(set)\n    dist = min(test_frontiers.keys())\n\n    while max(test_frontiers.keys()) >= dist and dist <= max_test_dist:\n        for idx in test_frontiers[dist]:\n            if field_dist[idx] == fill_value:\n                cur_free_on_step = free_on_step_flat[idx]\n\n                if dist < min_test_dist or cur_free_on_step <= dist:\n                    frontiers[dist].add(idx)\n                    field_dist[idx] = dist\n                elif late_arrival:\n                    test_frontiers[cur_free_on_step].add(idx)\n\n        this_frontier = frontiers[dist]\n\n        if this_frontier:\n            next_frontier_add = test_frontiers[dist + 1].add\n\n            for idx in this_frontier:\n                idx2 = idx + size_y\n                if idx2 >= max_idx:\n                    idx2 -= max_idx\n                next_frontier_add(idx2)\n\n                idx2 = idx - size_y\n                if idx2 < 0:\n                    idx2 += max_idx\n                next_frontier_add(idx2)\n\n                y = idx % size_y\n\n                if y == 0:\n                    next_frontier_add(idx - 1 + size_y)\n                    next_frontier_add(idx + 1)\n                elif y == size_y - 1:\n                    next_frontier_add(idx - 1)\n                    next_frontier_add(idx + 1 - size_y)\n                else:\n                    next_frontier_add(idx - 1)\n                    next_frontier_add(idx + 1)\n\n        dist += 1\n\n    field_dist = field_dist.reshape(free_on_step.shape)  # type: ignore\n\n    # TODO: frontiers to (x,y)\n    return FloodfillResult(field_dist=field_dist, frontiers=frontiers)\n')
    write_file('anton/lib/scoreobs.py', 'import numpy as np\n\n\ndef clip(x, *, low=None, high=None):\n    if low is not None:\n        x = max(x, low)\n    if high is not None:\n        x = min(x, high)\n    return x\n\n\ndef calc_score(features, max_needed_free_area=10, tree=None):\n    if features.obs.my_goose is None:\n        return (-np.Inf,)\n\n    enemyless_free_area = np.sum(features.my_floodfill != np.Inf)\n\n    guaranteed_free_area = np.sum(features.my_floodfill < features.all_enemy_floodfill)\n\n    clip_free_area = max_needed_free_area\n\n    if tree is None:\n        clip1 = clip_free_area\n    else:\n        clip1 = int(1.5 * len(tree.data.obs.my_goose))\n\n    return (\n        clip(enemyless_free_area, high=clip1),\n        clip(guaranteed_free_area, high=clip_free_area),\n        len(features.obs.my_goose),\n        100 - features.food_dist if not np.isnan(features.food_dist) else 0,\n    )\n')
    write_file('anton/lib/tree.py', 'from collections import defaultdict\nimport itertools\nfrom dataclasses import dataclass\nfrom typing import Any, Dict, List, Optional, Set, Tuple, NamedTuple\n\nfrom .geometry import Translation, trans_action_map\nfrom .observation import Observation\nfrom .simulate_step import simulate_step\n\n\nreverse_action: Dict[Translation, Translation] = {\n    (-1, 0): (1, 0),\n    (1, 0): (-1, 0),\n    (0, 1): (0, -1),\n    (0, -1): (0, 1),\n}\n\n\n@dataclass\nclass TreeData:\n    obs: Observation\n    last_actions: Optional[Dict[int, Translation]] = None\n\n    def __repr__(self):\n        return f"geese:{self.obs.geese}, food:{self.obs.food}, step:{self.obs.step}"\n\n\nclass NextNode(NamedTuple):\n    step_actions: Dict[int, Translation]\n    tree_node: "TreeNode"\n\n\n@dataclass\nclass TreeNode:\n    data: TreeData\n\n    next: Optional[List[NextNode]] = None\n\n    def __repr__(self):\n        return f"TreeNode({self.data}; {len(self.next) if self.next is not None else \'no\'} children)"\n\n    def groupby(self, key_func):\n        my_action_next_nodes: Dict[Any, List[NextNode]] = defaultdict(list)\n\n        for next_node in self.next:\n            my_action_next_nodes[key_func(next_node.step_actions)].append(next_node)\n\n        return my_action_next_nodes\n\n\ndef is_same_obs(obs1: Observation, obs2: Observation):\n    geese1 = {idx: goose.poses for idx, goose in obs1.geese.items()}\n    geese2 = {idx: goose.poses for idx, goose in obs2.geese.items()}\n\n    food1 = obs1.food\n    food2 = obs2.food\n\n    # TODO: test geese needed?\n    return geese1 == geese2 and food1 == food2\n\n\ndef calc_last_actions(obs1: Observation, obs2: Observation):\n    result = {}\n\n    for idx, goose2 in obs2.geese.items():\n        goose1 = obs1.geese[idx]\n        result[idx] = obs2.geo.diff_to(goose1.head, goose2.head)\n\n    return result\n\n\ndef make_tree(obs, *, level: int, old_tree_node: Optional[TreeNode] = None) -> TreeNode:\n    """\n    Will try to re-use old_tree_node\n    """\n    # if old_tree_node is not None and old_tree_node.next is not None:\n    #     for _actions, next_tree_node in old_tree_node.next:\n    #         if is_same_obs(next_tree_node.data.obs, obs):\n    #             expand_tree(\n    #                 next_tree_node, level=level, generate_leaves=generate_leaves\n    #             )\n    #             return next_tree_node\n\n    tree_data = TreeData(\n        obs,\n        last_actions=calc_last_actions(old_tree_node.data.obs, obs)\n        if old_tree_node is not None\n        else None,\n    )\n    tree_node = TreeNode(tree_data)\n\n    expand_tree(\n        tree_node,\n        level=level,\n        generate_leaves=generate_leaves,\n        stop_func=lambda tree_node: tree_node.data.obs.my_index\n        not in tree_node.data.obs.geese,\n    )\n\n    return tree_node\n\n\ndef generate_leaves(tree_data: TreeData) -> List[NextNode]:\n    if not tree_data.obs.geese:\n        return []\n\n    result: List[NextNode] = []\n\n    all_actions = {\n        idx: find_actions(\n            tree_data.obs,\n            idx=idx,\n            forbidden_actions={reverse_action[tree_data.last_actions[idx]]}\n            if tree_data.last_actions is not None\n            else None,\n        )\n        for idx in tree_data.obs.geese.keys()\n    }\n\n    for action_list in itertools.product(*all_actions.values()):\n        cur_actions = {\n            idx: action for idx, action in zip(all_actions.keys(), action_list)\n        }\n\n        new_obs = simulate_step(tree_data.obs, cur_actions)\n\n        new_tree_data = TreeData(new_obs, cur_actions)\n\n        result.append(NextNode(cur_actions, TreeNode(new_tree_data)))\n\n    return result\n\n\ndef expand_tree(tree_node: TreeNode, *, level: int, generate_leaves, stop_func=None):\n    if level == 0:\n        return\n\n    assert level > 0\n\n    if stop_func is not None and stop_func(tree_node):\n        return\n\n    if tree_node.next is None:\n        tree_node.next = generate_leaves(tree_node.data)\n\n    for _actions, tree_node in tree_node.next:\n        expand_tree(\n            tree_node,\n            level=level - 1,\n            generate_leaves=generate_leaves,\n            stop_func=stop_func,\n        )\n\n\ndef find_actions(\n    obs: Observation, *, idx: int, forbidden_actions: Optional[Set[Translation]] = None\n) -> Set[Translation]:\n    """\n    Find all actions for goose idx without an sure death\n    """\n    if forbidden_actions is None:\n        forbidden_actions = set()\n\n    # TODO: some geese might die next turn and not be dangerous\n    bodies = set(\n        itertools.chain.from_iterable(goose.poses[:-1] for goose in obs.geese.values())\n    )\n\n    head = obs.geese[idx].head\n\n    result = {\n        action\n        for action in trans_action_map.keys()\n        if action not in forbidden_actions\n        and obs.geo.translate(head, action) not in bodies\n    }\n\n    if not result:\n        result = {(1, 0)}  # single action if all actions make me die\n\n    return result\n')
    write_file('anton/lib/simulate_step.py', 'import itertools\nfrom collections import Counter\nfrom typing import Dict\n\nfrom .geometry import Translation\nfrom anton.lib.observation import Goose, Observation\n\n\ndef simulate_step(\n    obs: Observation, actions: Dict[int, Translation], move_action_none_tails=True\n) -> Observation:\n    """\n    :param obs: observation before actions (Observation)\n    :param actions: dictionary of {goose_idx: Optional[tuple(dx, dy)], ...} for action; e.g. {0: (1,0), 1:(0,-1), 2:(-1,0)}\n        `None` if that goose stays (not possible in real game)\n    :return: observation after actions (Observation)\n    It does not implement the rule that you are not allowed to go to your previous position!\n    If geese are allowed to stay, then head collisions are not sensible\n    use `Observation.from_obs_conf(obs, my_index=0)` to generate an Observation object from raw kaggle data\n    see Observation class for how to access the data\n    """\n    if actions.keys() != obs.geese.keys():\n        raise ValueError(\n            f"Action idxs and obs.geese idxs do not match: {actions.keys()}!={obs.geese.keys()}. You need to provide actions for all geese."\n        )\n\n    if not set(actions.values()) <= {(-1, 0), (1, 0), (0, 1), (0, -1), None}:\n        raise ValueError(f"Illegal values found amongst actions: {actions}")\n\n    pre_step_goose_heads = {idx: goose.head for idx, goose in obs.geese.items()}\n\n    new_heads = {\n        idx: obs.geo.translate(pre_step_goose_heads[idx], action)\n        if action is not None\n        else pre_step_goose_heads[idx]\n        for idx, action in actions.items()\n    }\n\n    head_collisions = {\n        pos for pos, count in Counter(new_heads.values()).items() if count > 1\n    }\n\n    # assuming that collisions happen before eating; also assumes that there is no food on bodies\n    active_food = obs.food - head_collisions\n\n    # simulate step, head collisions, food eating\n    geese_including_body_collisions = {\n        idx: (\n            [head] + obs.geese[idx].poses[: -1 if head not in active_food else None]\n        )  # goose moved\n        if head != pre_step_goose_heads[idx]\n        else (  # goose stays\n            obs.geese[idx].poses[:-1]\n            if move_action_none_tails and len(obs.geese[idx].poses) > 1\n            else obs.geese[idx].poses\n        )\n        for idx, head in new_heads.items()\n        if head not in head_collisions\n    }\n\n    bodies = set(\n        itertools.chain.from_iterable(\n            goose_poses[1:] for goose_poses in geese_including_body_collisions.values()\n        )\n    )\n\n    geese = {\n        idx: goose_poses\n        for idx, goose_poses in geese_including_body_collisions.items()\n        if goose_poses[0] not in bodies\n    }\n\n    alive_heads = {goose_poses[0] for goose_poses in geese.values()}\n\n    return Observation(\n        geese={idx: Goose(goose_poses) for idx, goose_poses in geese.items()},\n        geo=obs.geo,\n        my_index=obs.my_index,\n        step=obs.step + 1,\n        food=obs.food - alive_heads,\n    )')
    write_file('anton/lib/score_tree.py', 'from operator import itemgetter\nfrom typing import Dict, NamedTuple, Optional, Tuple\n\nfrom .geometry import Translation\nfrom .tree import TreeNode, make_tree\n\nScoreType = Tuple\n\n\nclass Score(NamedTuple):\n    action: Optional[Translation]\n    score: ScoreType\n    action_scores: Optional[Dict[Translation, ScoreType]] = None\n\n\nclass TreeScorer:\n    def __init__(self, leaf_score_func, agg_func, level=2):\n        self.leaf_score_func = leaf_score_func\n        self.agg_func = agg_func\n        self.level = level\n        self.old_tree_node = None\n\n    def __call__(self, obs):\n        tree = make_tree(obs, level=self.level, old_tree_node=self.old_tree_node)\n        self.old_tree_node = tree\n\n        return score_tree(\n            tree, leaf_score_func=self.leaf_score_func, agg_func=self.agg_func\n        )\n\n\ndef max_min_agg(key_treenodes, node_score_func):\n    best_score: ScoreType = None  # type: ignore\n    best_action = None\n    action_scores: Dict[Translation, ScoreType] = {}\n\n    for my_action, tree_nodes in key_treenodes.items():\n        # print(my_action)\n        min_score: ScoreType = None  # type: ignore\n\n        for next_node in tree_nodes:\n            score = node_score_func(next_node).score\n\n            if min_score is None or score < min_score:\n                min_score = score\n\n            if best_score is None or min_score < best_score:\n                # print(f"Cutting action {my_action} since {min_score}<{best_score}")\n                break\n\n        action_scores[my_action] = min_score\n\n        if best_score is None or min_score > best_score:\n            # print(f"Found new best action {my_action} since {min_score}>{best_score}")\n            best_score = min_score\n            best_action = my_action\n\n    return Score(action=best_action, score=best_score, action_scores=action_scores)\n\n\ndef max_mean_agg(key_treenodes, node_score_func, mean_func):\n    best_score: ScoreType = None  # type: ignore\n    best_action = None\n    action_scores: Dict[Translation, ScoreType] = {}\n\n    for my_action, tree_nodes in key_treenodes.items():\n        mean_score = mean_func(\n            node_score_func(next_node).score for next_node in tree_nodes\n        )\n\n        action_scores[my_action] = mean_score\n\n        if best_score is None or mean_score > best_score:\n            best_score = mean_score\n            best_action = my_action\n\n    return Score(action=best_action, score=best_score, action_scores=action_scores)\n\n\ndef score_tree(\n    tree_node: TreeNode, *, leaf_score_func, agg_func, top_tree_node=None\n) -> Score:\n    if top_tree_node is None:\n        top_tree_node = tree_node\n\n    if tree_node.next is None:\n        leaf_score = Score(\n            action=None, score=leaf_score_func(tree_node.data, tree=top_tree_node)\n        )\n        # print(f"Found leaf score {leaf_score} for {tree_node.data.obs}")\n        return leaf_score\n\n    my_index = tree_node.data.obs.my_index\n\n    return agg_func(\n        tree_node.groupby(itemgetter(my_index)),\n        lambda node: score_tree(\n            node.tree_node,\n            leaf_score_func=leaf_score_func,\n            agg_func=agg_func,\n            top_tree_node=top_tree_node,\n        ),\n    )\n\n')
    write_file('shared/scoreobs.py', '\nfrom anton.lib.features import make_features\nfrom anton.lib.observation import Observation\nfrom anton.lib.scoreobs import calc_score as calc_score_ant\n\n\ndef calc_score(raw_obs, max_needed_free_area=6):\n    """\n    :param raw_obs: observation dict from kaggle\n    :param my_index: index of my goose\n    :return: tuple of floats where max of these tuples is the best move\n    """\n    obs = Observation.from_obs_conf(raw_obs)\n    features = make_features(obs)\n    result = calc_score_ant(features, max_needed_free_area=max_needed_free_area)\n    return result\n\n')
    write_file('shared/nn_features.py', 'import numpy as np\nfrom anton.lib.features import make_features\nfrom anton.lib.simulate_step import simulate_step\nfrom functools import partial\n\n\ndef calc_corridor(obs, feats):\n    return np.sum(feats.my_floodfill != np.Inf)\n\n\ndef calc_my_area(obs, feats, min_diff=0):\n    return np.sum((feats.all_enemy_floodfill - feats.my_floodfill) > min_diff)\n\n\ndef calc_my_length(obs, feats):\n    return len(obs.geese[obs.my_index])\n\n\ndef calc_food_dist(obs, feats):\n    return feats.food_dist\n\n\ndef calc_tail_dist(obs, feats):\n    return feats.my_floodfill[obs.my_goose.poses[-1]]\n\n\ndef make_nn_features_each_direction(\n    obs,\n    feat_funcs=[\n        calc_corridor,\n        partial(calc_my_area, min_diff=1),\n        calc_my_length,\n        calc_food_dist,\n        calc_tail_dist,\n    ],\n) -> np.ndarray:\n    """\n    Features for each direction. Can be used in NN which work with fixed position NESW output (or any other order of those)\n    :param obs: observation before actions (Observation)\n    :return: np.array shape (num_feats, 4); second dimension is ordered by N E S W (N counterclockwise)\n    result contains NaN if our goose dies in that direction\n    and contains np.Inf sometimes, if food/tail is unreachable; you could clip that to a fixed value\n    for NN models with 4-direction output, you may flatten the array\n    default features are roughly ordered by importance\n    use `anton.lib.observation.Observation.from_obs_conf(obs_json)` to generate an Observation object from raw kaggle data;\n    argument `my_index=0` if you want to overwrite which goose index is us (otherwise it is read from the replay)\n    see Observation class for how to access the data\n    """\n    if obs.my_index is None:\n        raise ValueError(\n            "You need to set `my_index` of the observation to determine which Goose is us"\n        )\n\n    step_obses = [\n        simulate_step(\n            obs,\n            {idx: action if idx == obs.my_index else None for idx in obs.geese.keys()},  # type: ignore\n        )\n        for action in [(-1, 0), (0, 1), (1, 0), (0, -1)]\n    ]\n\n    step_obses_feats = [(step_obs, make_features(step_obs)) for step_obs in step_obses]\n\n    result = []\n    for feat_func in feat_funcs:\n        result.append(\n            [\n                feat_func(step_obs, feats)\n                if step_obs.my_index in step_obs.geese\n                else np.nan\n                for step_obs, feats in step_obses_feats\n            ]\n        )\n\n    return np.array(result)\n\n\ndef make_nn_features_for_position(\n    obs,\n    feat_funcs=[\n        calc_corridor,\n        calc_my_area,\n        calc_my_length,\n        calc_food_dist,\n        calc_tail_dist,\n    ],\n) -> np.ndarray:\n    """\n    Features for a single particular position. Does not look at the following step.\n    :param obs: observation before actions (Observation)\n    :return: np.array shape (num_feats,)\n    result contains NaN if our goose dies in that direction\n    and contains np.Inf sometimes, if food/tail is unreachable; you could clip that to a fixed value\n    use `anton.lib.observation.Observation.from_obs_conf(obs_json)` to generate an Observation object from raw kaggle data;\n    argument `my_index=0` if you want to overwrite which goose index is us (otherwise it is read from the replay)   \n    see Observation class for how to access the data\n    """\n    if obs.my_index is None:\n        raise ValueError(\n            "You need to set `my_index` of the observation to determine which Goose is us"\n        )\n\n    feats = make_features(obs)\n\n    result = [feat_func(obs, feats) for feat_func in feat_funcs]\n\n    return np.array(result)')

    # Import scoring from Anton libraries
    from shared.scoreobs import calc_score
    from anton.lib.geometry import *
    from anton.lib.observation import *
    from shared.nn_features import *

class rGeometry:
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y

    @property
    def shape(self):
        return (self.size_x, self.size_y)

    def prox(self, pos):
        return {
            self.translate(pos, direction)
            for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]
        }

    def translate(self, pos, diff):
        x, y = pos
        dx, dy = diff
        return Pos((x + dx) % self.size_x, (y + dy) % self.size_y)
    
    def field_idx_to_pos(self, field_idx):
        x = field_idx // 11
        y = field_idx % 11

        if not (0 <= x < 7 and 0 <= y < 11):
            raise ValueError("Illegal field_idx {field_idx} with x={x} and y={y}")

        return Pos(x, y)


# --------------------------------------------------
# net.py

import numpy as np
from sklearn.utils.extmath import softmax
import onnxruntime

# centering doesn't do much / anything, but has no disbenefit
def centerize(b):
    dy, dx = np.where(b[0])
    
    if not dy: return b

    centerize_y = (np.arange(0, 7) - 3 + dy[0]) % 7
    centerize_x = (np.arange(0, 11) - 5 + dx[0]) % 11

    b = b[:, centerize_y, :]
    b = b[:, :, centerize_x]

    return b

# input for our RL experiment 24
def make_input24(obses, agent_index, do_centerize=False, tta=False):
    b = np.zeros((16, 7 * 11), dtype=np.float32)
    obs = obses[-1]
    
    for p, pos_list in enumerate(obs['geese']):
        # head position
        for pos in pos_list[:1]:
            b[0 + (p - agent_index) % 4, pos] = 1  
            
        # tip position
        for pos in pos_list[-1:]:
            b[4 + (p - agent_index) % 4, pos] = 1

        for e, pos in enumerate(pos_list):
            b[8, pos] = 1
            
            b[9, pos] = min(30, len(pos_list) - e) / 30

            
    # previous head position
    if len(obses) > 1:
        obs_prev = obses[-2]
        
        for p, pos_list in enumerate(obs_prev['geese']):
            for pos in pos_list[:1]:
                b[10 + (p - agent_index) % 4, pos] = 1

    # food
    if len(obs['food'])>0: b[14, obs['food'][0]] = 1
    if len(obs['food'])==2: b[15, obs['food'][1]] = 1

    b = b.reshape(-1, 7, 11)

    if do_centerize: b = centerize(b)

    if tta:
        xs = np.zeros((4, 16, 7, 11), dtype=np.float32)
        xs[0] = b
        xs[1] = np.flip(b, axis=1)
        xs[2] = np.flip(b, axis=2)
        xs[3] = np.flip(np.flip(b, axis=1),axis=2)

        return xs

    else:

        return b

# input for our RL experiment 26
def make_input26(obses, agent_index, do_centerize=False, tta=False):
    b = np.zeros((20, 7 * 11), dtype=np.float32)
    obs = obses[-1]

    for p, pos_list in enumerate(obs['geese']):

        # head position
        for pos in pos_list[:1]:
            b[0 + (p - agent_index) % 4, pos] = 1

        # tail position
        for pos in pos_list[-1:]:
            b[4 + (p - agent_index) % 4, pos] = 1

        # shrink the tail of oppos
        if p!=agent_index:
            if len(pos_list)>1: pos_list = pos_list[:-1]

        # 2nd tail position
        for pos in pos_list[-1:]:
            b[8 + (p - agent_index) % 4, pos] = 1

        for e, pos in enumerate(pos_list):
            b[12, pos] = 1

            b[13, pos] = min(30, len(pos_list) - e) / 30

    # previous head position
    if len(obses) > 1:
        obs_prev = obses[-2]
        
        for p, pos_list in enumerate(obs_prev['geese']):
            for pos in pos_list[:1]:
                b[14 + (p - agent_index) % 4, pos] = 1

    # food
    if len(obs['food'])>0: b[18, obs['food'][0]] = 1
    if len(obs['food'])==2: b[19, obs['food'][1]] = 1

    b = b.reshape(-1, 7, 11)

    if do_centerize: b = centerize(b)
    
    if tta:

        xs = np.zeros((4, 20, 7, 11), dtype=np.float32)
        xs[0] = b
        xs[1] = np.flip(b, axis=1)
        xs[2] = np.flip(b, axis=2)
        xs[3] = np.flip(np.flip(b, axis=1),axis=2)

        return xs

    else:

        return b

WEIGHTS_AND_INPUTS = [ \
                 ['run26_36090_opset13_d0617_bs4.onnx', make_input26], \
                 ['run24_27268_opset13_d0617_bs4.onnx', make_input24], \
                 ]

model_sessions = [onnxruntime.InferenceSession(RESOURCE_DIR + WEIGHTS_DIR + FILE[0], sess_options = opts) for FILE in WEIGHTS_AND_INPUTS]
value_session = onnxruntime.InferenceSession(RESOURCE_DIR + WEIGHTS_DIR + VALUE_FILE, sess_options = opts)

def make_input(obses, agent_index, do_centerize=False):

    b = np.zeros((17, 7 * 11), dtype=np.float32)
    obs = obses[-1]

    for p, pos_list in enumerate(obs['geese']):
        # head position
        for pos in pos_list[:1]:
            b[0 + (p - agent_index) % 4, pos] = 1
        # tip position
        for pos in pos_list[-1:]:
            b[4 + (p - agent_index) % 4, pos] = 1
        # whole position
        for pos in pos_list:
            b[8 + (p - agent_index) % 4, pos] = 1
            
    # previous head position
    if len(obses) > 1 and obses[-2] != None:
        obs_prev = obses[-2]
        for p, pos_list in enumerate(obs_prev.geese):
            for pos in pos_list[:1]:
                b[12 + (p - agent_index) % 4, pos] = 1

    # food
    for pos in obs['food']:
        b[16, pos] = 1

    b = b.reshape(-1, 7, 11)

    if do_centerize:
        b = centerize(b)

    return b

# perform inference and TTA if requested
def inputs_to_predictions(obses, idx, mode, do_tta = False):

    if not do_tta:  

        stack = make_input26(obses, idx, do_centerize = True, tta=False)

        # LA value net scoring -- bare minimum calculations

        x = np.expand_dims(stack, axis=0)

        ort_inputs = {value_session.get_inputs()[0].name: x}
        o = value_session.run(None, ort_inputs)

        score = float(o[1])
        if mode == 'SCORE': return (None, score)

        p = softmax(o[0])[0]
        if mode == 'ACTION_SCORES':
            return dict(zip(ACTIONS, np.array([p[0], p[3], p[1], p[2]])))

    else:
        
        res = []
        scores = []

        for e, model_session in enumerate(model_sessions):
            
            input_func = WEIGHTS_AND_INPUTS[e][1]
            xs = input_func(obses, idx, do_centerize = True, tta=True)

            ort_inputs = {model_session.get_inputs()[0].name: xt}
            o = model_session.run(None, ort_inputs)
            op = softmax(o[0])[0]
            score = float(o[1])

            scores.append(score)

            # unwind TTA matrix
            if i in [1, 5]:
                op[1], op[0] = op[0], op[1]
            elif i in [2, 6]:
                op[2], op[3] = op[3], op[2]
            elif i in [3, 7]:
                op[1], op[0] = op[0], op[1]
                op[2], op[3] = op[3], op[2]

            res.append(op)

        p = np.mean(res, axis=0)
        score = np.mean(scores)

    # return NESW since RL net is NSWE
    return (np.array([p[0], p[3], p[1], p[2]]), score)
    
@wrapper_cache
def net_evaluate(obses, mode = 'SCORE', tta = None):
    if mode == 'TEST': return float

    tta = mode in ['ACTION', 'ACTION_SCORES'] if tta == None else tta

    action_scores, score = inputs_to_predictions(obses, obses[-1].index, mode = mode, do_tta = tta)
    if mode == 'SCORE': return score
    
    action = ACTIONS[np.argmax(action_scores)]
    action_scores = list(map(float, action_scores))
    dict_scores = dict(zip(ACTIONS, action_scores))
    return action if mode == 'ACTION' else dict_scores

## TTA is advantageous. But it takes time in Look-Ahead search. We'll use it for depth 1 only.

def unfurl_tta_single(op, mode):
    if mode==1:
        op[1], op[0] = op[0], op[1]
    elif mode==2:
        op[2], op[3] = op[3], op[2]
    elif mode==3:
        op[1], op[0] = op[0], op[1]
        op[2], op[3] = op[3], op[2]
    return op

def unfurl_tta_array(ops):
    res = []
    for i, op in enumerate(ops):
        op = unfurl_tta_single(op, i)
        res.append(op)
    return np.mean(res, axis=0)

def rob_inputs_to_predictions(obses, idx):
    """
    Returns the neural network predicted/suggested probs NESW
    """

    res = []
    rot = obses[-1]['remainingOverageTime'] / (200 - obses[-1].step)

    for e, model_core in enumerate(model_sessions):

        input_func = WEIGHTS_AND_INPUTS[e][1]
        xs = input_func(obses, idx, do_centerize=True, tta=True)

        ort_inputs = {model_core.get_inputs()[0].name: xs}
        o = model_core.run(None, ort_inputs)
        ops = softmax(o[0])

        op = unfurl_tta_array(ops)
        res.append(op)

    p = np.mean(res, axis=0)
    return [p[0], p[3], p[1], p[2]]

def rob_nn_calc_score(obses):
    action_scores = rob_inputs_to_predictions(obses, obses[-1].index)
    return action_scores

class NetRules:

    def __init__(self):
        self.geo = rGeometry(size_x = 7, size_y = 11)
        self.evalrob = rob_nn_calc_score
        self.next_poses = {}
    
    def geese_from_obs(self, obs):
        geese = {}
        for idx, goose_data in enumerate(obs["geese"]):
            if goose_data:
                poses = [self.geo.field_idx_to_pos(idx) for idx in goose_data]
                geese[idx] = Goose(poses)
        return geese

    def food_from_obs(self, obs):
        return {
            self.geo.field_idx_to_pos(idx)
            for idx in obs["food"]
        }

    def all_occupied_positions(self, geese):
        occupied = set()
        for i, goose in geese.items():
            for pos in goose.poses[:-1]:
                occupied.add(pos)
        return occupied

    def legal_moves_for_geese(self, geese):
        all_occupied_pos = self.all_occupied_positions(geese)
        geese_possible = {}
        for i, goose in geese.items():
            poss = set()
            for pos in self.geo.prox(goose.head):
                if pos not in all_occupied_pos:
                    poss.add(pos)
                geese_possible[i] = poss
        return geese_possible

    ########
    # This function analyses opponents and determines how to classify the danger they pose
    ########
    def danger_poses_from_geese(self, geese, obs, food, preds):

        self.next_poses = {} 
        
        all_occupied_pos = self.all_occupied_positions(geese)
        
        # for each goose, list of legal moves
        geese_possible = self.legal_moves_for_geese(geese)

        # for each goose, classify move as danger or not danger on the surface
        result = {}
        prox_poses = []
        for i, this_poses in geese_possible.items():
            unsafe = set()
            safe = this_poses.copy()
            allmoves = this_poses.copy()

            for this_pos in this_poses:
                for j, oppo_poses in geese_possible.items():
                    if j != i:
                        if this_pos in oppo_poses:
                            if this_pos not in unsafe: unsafe.add(this_pos)
                            if this_pos in safe: safe.remove(this_pos)

            if i != self.index: prox_poses.extend(this_poses)

            # if corridor of 1, it isn't safe for the oppo goose so they won't go there.
            if i != self.index:
                nnfs = self.nnfs_from_obs(obs, i)
                for d in range(4):
                    if nnfs[d] == 1:
                        corri_pos = self.geo.translate(geese[i].head, trans_action_map_inv[int_to_direction_name[d]])
                        if corri_pos in unsafe and len(allmoves)>1: unsafe.remove(corri_pos)
                        if corri_pos in safe and len(allmoves)>1: safe.remove(corri_pos)
                        if corri_pos in allmoves: allmoves.remove(corri_pos)

            result[i] = {'allmoves': list(allmoves), 'safe': list(safe), 'unsafe': list(unsafe)}

            # if the goose has no other choice, it becomes a known move
            if len(allmoves) == 1 and i != self.index: 
                self.next_poses[i] = list(allmoves)[0]

        lengths = list(map(len, obs.geese))
        danger_poses = []
        for i, this_poses in result.items():
            if i not in self.next_poses: # next poses is HRL public bots
                if i != self.index:
                    if CAUTIOUS: # early submission time window usually 1-2 hours
                        if lengths.count(0) < 2 or lengths[self.index] != max(lengths) or lengths[i]<4: 
                            danger_poses.extend(this_poses['unsafe'])
                    else: # normal operations, only avoid danger in edge cases
                        if lengths.count(0) == 2 and lengths[self.index] != max(lengths): # if we aren't largest of last 2
                            danger_poses.extend(this_poses['unsafe'])
                        elif lengths[i] < 6: # if a new geese, as they are unpredictable
                            danger_poses.extend(this_poses['unsafe'])
                        else:
                            for this_pos in this_poses['unsafe']: # if there is a head food risk
                                if this_pos in food or (this_pos in preds and preds[this_pos]<.7):
                                    if lengths.count(0) < 2 or lengths[self.index] != max(lengths):
                                        danger_poses.extend([this_pos])

        danger_poses = list(set(danger_poses).intersection(result[self.index]['unsafe'])) # only those near us
        return danger_poses, list(set(prox_poses))

    def last_action_from_obses(self, obses):
        if obses[0]: 
            bef = self.geo.field_idx_to_pos(obses[0].geese[obses[0].index][0])
            aft = self.geo.field_idx_to_pos(obses[1].geese[obses[1].index][0])
            for i in range(4):
                x, y = bef
                diff = trans_action_map_inv[int_to_direction_name[i]]
                dx, dy = diff
                if Pos((x + dx) % 7, (y + dy) % 11) == aft:
                    return int_to_direction_name[i]
        return None

    def get_length_advantage(self, geese):
        # Find our goose length advantage
        max_opp_len = 0
        for g in geese: 
            if g != self.index:
                if len(geese[g]) > max_opp_len:
                    max_opp_len = len(geese[g])
            else:
                my_len = len(geese[g])
        return max_opp_len - my_len

    def nn_actions_to_action(self, nn_actions, last_action):
        if sum(np.clip(nn_actions, 0, 1)) > 0:
            action = int_to_direction_name[np.argmax(nn_actions)]
        else:
            p =  p=[1/3,1/3,1/3,1/3]
            p[(direction_to_int[last_action]+2)%4] = 0 
            if VERBOSE: print('GAME OVER. Random move.', p)
            action = int_to_direction_name[np.random.choice(4, p=p)]
        
        return action

    def clip_feat(self, feat, clip_lo, clip_hi, nan_val):
        feat[np.isnan(feat)]=nan_val      
        return np.clip(feat, clip_lo, clip_hi)

    def clip_feat_corridor(self, feat):
        return self.clip_feat(feat, 0, 100, 0)

    def nnfs_from_obs(self, observation, idx):
        obsy = observation.copy()
        obsy["index"] = idx
        obsy = Observation.from_obs_conf(obsy)
        nns = make_nn_features_each_direction(obsy, feat_funcs=[calc_corridor])
        return self.clip_feat_corridor(nns[0])

    ########
    # This function refines NN policy predictions with heuristics
    ########
    def refine_eval(self, obses, raw_preds, index):

        nn_actions = raw_preds.copy()

        prev_obs, last_action = None, None
        observation = obses[-1]

        if len(obses) > 1:
            prev_obs = obses[-2]
            last_action = self.last_action_from_obses(obses[-2:])

        geese = self.geese_from_obs(observation)  
        my_goose = geese[self.index]
        food = self.food_from_obs(observation)

        these_poses = [self.geo.translate(my_goose.head, trans_action_map_inv[int_to_direction_name[i]]) for i in range(4)]
        these_preds = {}
        for i in range(4):
            these_preds[these_poses[i]] = nn_actions[i]

        danger_poses, prox_poses = self.danger_poses_from_geese(geese, obses[-1], food, these_preds)
        
        if VERBOSE and len(danger_poses)>0: print(obses[-1].step, 'Dangers', danger_poses)

        num_geese = len([o for o in observation['geese'] if o])
        len_diff =  self.get_length_advantage(geese) # how much larger we are than 1st

        if last_action is not None: # never go backwards
            nn_actions[(direction_to_int[last_action]+2)%4] = -1 

        # calculate opponent corridors and store in nnfss
        ff_obs = copy.deepcopy(observation)
        nnfss = []
        dead_on_next_step = []
        for i in range(4):
            nnfss.append(self.nnfs_from_obs(ff_obs, i))
            dead_on_next_step.append(max(nnfss[i]) == 0 and len(ff_obs['geese'][i])>0)

        # calculate our corridors and store in nnfs
        # we reduce opponent geese length by 1 to handle some corridor function oddities
        for i in range(4):
            if len(ff_obs['geese'][i])>1:
                if i!=self.index:
                    ff_obs['geese'][i] = ff_obs['geese'][i][:-1]

        nnfs = self.nnfs_from_obs(ff_obs, self.index)

        # avoid traps:
        #  take a corridor reading of current state
        #  for each goose, simulate each directional movement 
        #  take corridor measurements and watch for plunges 
        traps = [False]*4
        for j in range(4):
            if j != self.index: # for each opponent goose
                if j not in self.next_poses: # if not public HRL
                    if len(observation['geese'][j])>1:
                        for h, ap in enumerate(adjacent_positions(observation['geese'][j][0],11,7)): # for each move they could take
                            obs_n = copy.deepcopy(observation)
                            obs_n['geese'][j] = [ap] + obs_n['geese'][j][:-1]
                            new_nnfs = self.nnfs_from_obs(obs_n, self.index)
                            for k in range(4):
                                if 0<new_nnfs[k]<6 and nnfss[self.index][k]>=6:
                                    if VERBOSE: print(f'AVOID TRAP: Goose {j} may trap us from going {int_to_direction_name[k]}', int(nnfss[self.index][k]), int(new_nnfs[k]))
                                    traps[k] = True

        # make traps:
        #  take a corridor reading of current state for each goose
        #  for our goose, simulate each directional movement 
        #  take corridor measurements and watch for plunges 
        wraps = [False]*4
        if len(observation['geese'][self.index])>1:
            for h, ap in enumerate(adjacent_positions(observation['geese'][self.index][0],11,7)): # for each move they could take
                obs_n = copy.deepcopy(observation)
                obs_n['geese'][self.index] = [ap] + obs_n['geese'][self.index][:-1]
                for j in range(4):
                    if j!=self.index:
                        new_nnfs = self.nnfs_from_obs(obs_n, j)
                        for k in range(4):
                            if 0<new_nnfs[k]<6 and nnfss[j][k]>=6 and np.argmax(new_nnfs)==k:
                                print(f'POTENTIAL TRAP: Goose {j} by going {int_to_direction_name[h]}', int(nnfss[j][k]), int(new_nnfs[k]))
                                wraps[h] = True

        for i in range(4): # each dir through NESW

            if last_action is not None and i != ((direction_to_int[last_action]+2)%4):

                this_pos = self.geo.translate(my_goose.head, trans_action_map_inv[int_to_direction_name[i]])
                this_pos = these_poses[i]

                if VERBOSE: print(f'....Looking {int_to_direction_name[i]} to {this_pos}')

                # Rule: If an nn_pred<eps, set to eps
                if 1e-13<nn_actions[i]<1e-4: nn_actions[i]=1e-14
                
                # Rule: Simple body block checks
                for j in geese:
                    goose = geese[j]
                    if this_pos in goose.poses[:-1]:
                        nn_actions[i] = -1
                        if VERBOSE: print(f'....Looking {int_to_direction_name[i]} to {this_pos} blocked by Goose {j} body position')
                    if this_pos == goose.poses[-1]:
                        if j in self.next_poses: # i.e. a HandyRL goose
                            if self.next_poses[j] in list(food):
                                nn_actions[i] = -1
                                if VERBOSE: print(f'....Looking {int_to_direction_name[i]} to {this_pos} blocked by Goose {j} tail position')
                        else:
                            # The tail is risky if the head is next to food
                            if (self.geo.prox(goose.head) & food) and j!=self.index:
                                nn_actions[i] = 1e-14
                                if VERBOSE: print(f'...Looking {int_to_direction_name[i]} to {this_pos} blocked by Goose {j} tail position')

                ## Rule: handle possible impacts and potentially take/avoid danger poses 
                if this_pos in list(self.next_poses.values()):
                    if nn_actions[i]>0:
                        if num_geese == 2 and len_diff < 0:
                            nn_actions[i] = 100
                            if VERBOSE: print(f'... taking known winning impact at {this_pos}')
                        else:
                            nn_actions[i] = 1.1e-15
                            if VERBOSE: print(f'... avoiding known impact at {this_pos}')
                else:
                    if this_pos in danger_poses and nn_actions[i]>0:
                        if num_geese == 2:
                            if len_diff < 0:
                                nn_actions[i] = max(nn_actions) 
                                if VERBOSE: print(f'... raising winning impact opportunity at {this_pos}')
                            else:
                                nn_actions[i] = 2e-15
                                if VERBOSE: print(f'... avoiding losing impact at {this_pos}')
                        else:
                            if VERBOSE: print(f'... avoiding danger position at {this_pos}')
                            nn_actions[i] = 1e-14 

                # Rule: never go into a 1-hole deadend 
                # Floodfill corridor checking should handle this, but just in case
                surrounded = 0
                for k in range(4): # each dir
                    surrounding_pos = self.geo.translate(this_pos, trans_action_map_inv[int_to_direction_name[k]])
                    for j in geese: 
                        goose = geese[j]
                        if not dead_on_next_step[j]:
                            if j in self.next_poses: 
                                poses = [self.next_poses[j]] + goose.poses
                                # check if goose head moving to food
                                tail_offset = np.sign(len({self.next_poses[j]} & food))
                                poses = poses[:-2+tail_offset]
                            else:
                                # check if goose head prox to food
                                if j!=self.index:
                                    tail_offset = np.sign(len(self.geo.prox(goose.head) & food))
                                else:
                                    tail_offset = 0
                                poses = goose.poses[:-2+tail_offset]
                            if surrounding_pos in poses: 
                                surrounded +=1
                    if surrounded == 4 and nn_actions[i]>1e-15:
                        if nn_actions[i] != -1: # i.e. Not already blocked 
                            if VERBOSE: print(f'....Looking {int_to_direction_name[i]} to {this_pos} surrounded')
                        nn_actions[i] = 1e-15
                
                # Rule: avoid corridors
                if nnfs[i] in [2,3,4,5,6] and nn_actions[i]>=1e-15: 
                    #print('NNFS', observation.step, int_to_direction_name[i], nnfs[i])
                    nn_actions[i] = nnfs[i] * 1e-15

                # Rule: avoid traps we could wander into
                if traps[i] and nn_actions[i]>1e-14:
                    nn_actions[i]=1e-13
                    if VERBOSE: print(f'Closing trap going {int_to_direction_name[i]}')
                
                # Rule: set a trap if advisable
                if wraps[i] and nn_actions[i]>.05:
                    nn_actions[i]=1
                    if VERBOSE: print(f'SETTING TRAP going {int_to_direction_name[i]}')

        # if the top choice is a prox pos, if a non danger pos exists, if calc_corri>65, prioritise that
        # factor in known moves. Doesn't seem to add any value in practise.
        revised = None
        largest_indices = np.argsort(nn_actions)[::-1]
        for li in largest_indices:
            if nn_actions[li]>0 and li is None:
                this_pos = self.geo.translate(my_goose.head, trans_action_map_inv[int_to_direction_name[li]])
                if this_pos not in prox_poses: 
                    if nnfs[li] > 65:
                        revised = li
        if revised is not None:
            nn_actions[li] = max(nn_actions)*1.01

        # if top choice is a tie, rank them
        if nn_actions.count(max(nn_actions)) > 1 and max(nn_actions) > 0:
            indices = [i for i, x in enumerate(nn_actions) if x == max(nn_actions)]
            if VERBOSE: 
                print(f'Re-ranking tied top score for {[indices]} ... was {nn_actions}')
            nn_actions = [-3]*4
            for idx in indices:
                nn_actions[idx] = raw_preds[idx]
        

        # Return outcome

        if VERBOSE: print(f'[Policy] action {int_to_direction_name[np.argmax(nn_actions)]} -- {nn_actions}') 

        action = self.nn_actions_to_action(nn_actions, last_action)
        return (action, nn_actions)

    def evaluate(self, obses):
        raw_preds = self.evalrob(obses)
        action, refined_preds = self.refine_eval(obses, raw_preds, self.index)
        return (action, refined_preds)


# --------------------------------------------------
# hrl.py

import numpy as np
import onnxruntime

# make_input is only used to detect HandyRL agents
def make_input(obses, agent_index, do_centerize = False):

    b = np.zeros((17, 7 * 11), dtype=np.float32)
    obs = obses[-1]

    for p, pos_list in enumerate(obs['geese']):
        # head position
        for pos in pos_list[:1]:
            b[0 + (p - agent_index) % 4, pos] = 1
        # tip position
        for pos in pos_list[-1:]:
            b[4 + (p - agent_index) % 4, pos] = 1
        # whole position
        for pos in pos_list:
            b[8 + (p - agent_index) % 4, pos] = 1
            
    # previous head position
    if len(obses) > 1:
        obs_prev = obses[-2]
        for p, pos_list in enumerate(obs_prev['geese']):
            for pos in pos_list[:1]:
                b[12 + (p - agent_index) % 4, pos] = 1

    # food
    for pos in obs['food']:
        b[16, pos] = 1

    return b.reshape(-1, 7, 11)

handyrl_weights = RESOURCE_DIR + WEIGHTS_DIR + 'handyrl.onnx'
handyrl_session = onnxruntime.InferenceSession(handyrl_weights, sess_options = opts)

@wrapper_cache
def hrl_evaluate(obses, mode = 'SCORE', tta = None):
    if mode == 'TEST': return float
    
    index = obses[-1].index

    X = make_input(obses, index)
    X = np.expand_dims(X, axis=0)

    ort_inputs = {handyrl_session.get_inputs()[0].name: X}
    output = handyrl_session.run(None, ort_inputs)

    predictions = output[0][0]
    value = float(output[1][0][0])
    
    action = ['NORTH', 'SOUTH', 'WEST', 'EAST'][predictions.argmax()]
    score_actions = dict(zip(ACTIONS, predictions))

    output = {'ACTION': action, 'SCORE': value, 'ACTION_SCORES': score_actions}
    return output[mode]

# --------------------------------------------------
# tree.py

from collections import defaultdict, Counter
import numpy as np
import itertools
import time


def simulate_step_2(obs, actions):
    """
    Simulates action steps onto an observation (by Anton)
    
    Args:
        obs (Observation): Current position in observation format
        actions (list): Actions to step each index goose through,
            insert `None` if agent is dead
    
    Returns:
        (Observation): Next observation after environment step

    References:
        [1] https://github.com/Gerenuk/HungryGeese/blob/master/src/anton/lib/simulate_step.py
    """

    new_heads = {
        idx: DIRECTION[action](row_col(obs.geese[idx][0]))
        for idx, action in enumerate(actions) if len(obs.geese[idx]) }

    head_collisions = {
        pos for pos, count in Counter(new_heads.values()).items() if count > 1 }

    # assuming that collisions happen before eating; also assumes that there is no food on bodies
    active_food = list(set(obs.food) - head_collisions)

    # simulate step, head collisions, food eating, hunger rate
    geese_including_body_collisions = {
        idx: [index_pos(head)] + obs.geese[idx][: -1 if index_pos(head) not in active_food else None][: -1 if obs.step % 40 == 39 else None]
        for idx, head in new_heads.items()
        if head not in head_collisions }

    bodies = set(
        itertools.chain.from_iterable(
            goose_poses[1:] for goose_poses in geese_including_body_collisions.values()) )

    geese = {
        idx: goose_poses
        for idx, goose_poses in geese_including_body_collisions.items()
        if goose_poses[0] not in bodies }

    alive_heads = {goose_poses[0] for goose_poses in geese.values()}

    dead_indexes = set(range(4)) - geese.keys()
    for idx in dead_indexes:
        geese[idx] = []

    return KaggleObservation({
        'index': obs.index,
        'step' : obs.step + 1,
        'geese': [geese[idx] for idx in range(4)],
        'food' : list(set(obs.food) - alive_heads),
    })

class Tree:
    
    def __init__(self, config, scorer, options = options):

        self.config = dotdict(dict(config))
        self.options = options

        self.eval = scorer
        self.score_type = scorer(None, mode = 'TEST')

        self.net_rules = NetRules()
        self.cache = DepthCache()

        self.MAX_SCORE = 5.0 if not self.options.GREEDY_MODE else 5e2
        if self.options.MAX_MEAN: self.MAX_SCORE /= 5.0
        self.MIN_SCORE = -self.MAX_SCORE
        
        # Scores for terminal positions [fourth, third, second, first]
        self.terminal_scores  = [self.MIN_SCORE, -0.35, 0.2, self.MAX_SCORE]
        self.collision_scores = [score + EPS for score in self.terminal_scores]
        if self.options.MAX_MEAN: self.terminal_scores = self.collision_scores = [self.MIN_SCORE, -1/3, 1/3, self.MAX_SCORE]
        self.terminal_scores += self.collision_scores + [s + EPS for s in self.collision_scores]

        if self.score_type == tuple:
            self.MAX_SCORE = (self.MAX_SCORE,)
            self.MIN_SCORE = (self.MIN_SCORE,)

        self.policy_epsilon = 0.01

        self.hrl_agents = []
        self.avoid_danger_agents = []

        self.prev_depth = 2
        self.count = 0
        self.nps = 0

    def set_hrl(self, indexes):
        """ (void): Setting HRL agents for search simplification """
        self.hrl_agents = indexes
        for agent in self.hrl_agents:
            next_move = hrl_evaluate(self.obses, mode = 'ACTION')
            self.net_rules.next_poses[agent] = DIRECTION[next_move](row_col(self.obs.geese[agent][0]))
    
    def set_avoid_danger(self, indexes):
        """ (void): Setting agents that choose not to go 
            into spaces with possible head collisions """
        self.avoid_danger_agents = indexes

    def update(self, obses):
        """ (void): Resetting tree to new settings """
        self.obses = obses
        self.start_time = time.perf_counter()
        self.hrl_agents = []
        self.count = 0
        self.obs = obses[-1]
        self.index = obses[-1].index
        self.net_rules.index = obses[-1].index
    
    def valid_moves(self, obses, index, danger_moves = False):
        """ 
        Getting valid actions for index agent 
        
        Args:
            obses (list): Previous and current observations
            index (int): Index of goose to get valid moves from
            danger_moves (bool): Whether or not to check
                possible head collision moves
        
        Returns:
            (list): List of NSEW valid moves that index agent 
                can take, ['NORTH'] if no valid moves
        """

        head = row_col(obses[-1].geese[index][0])
        food = set(map(row_col, obses[-1].food))
        lengths = list(map(len, obses[-1].geese))
        last_head = []

        if len(obses) > 1:
            last_head = [row_col(obses[-2].geese[index][0])]

        walls = set(last_head)
        danger_poses = set()
        tail_poses = set()

        for i, body in enumerate(obses[-1].geese):

            if len(body):

                # Removing bodies
                for seg in body[:-1]: # [:-1 if obses[-1].step % self.config.hunger_rate == 39 else 0]:
                    walls.add(row_col(seg))
                
                # Next head positions
                head_neighbors = set(get_neighbors(row_col(body[0])))
                if i in self.hrl_agents and index == self.index:
                    hrl_action = hrl_evaluate(change_index(obses, i), mode = 'ACTION')
                    head_neighbors = { DIRECTION[hrl_action](row_col(body[0])) }

                # Don't go into tails if food is near head
                if head_neighbors & food and obses[-1].step % self.config.hunger_rate != 39 and i != index:
                    tail_poses.add(row_col(body[-1]))
                
                # Danger positions - Possible head collisions
                if i != index and (lengths.count(0) < 2 or lengths[index] != max(lengths)):
                    for adj in head_neighbors:
                        # Always assume opponents will go for food
                        if adj not in food:
                            danger_poses.add(adj)

        # Nested unrolling rules
        neighbors = get_neighbors(head, walls.union(danger_poses).union(tail_poses))
        if len(neighbors) == 0 or danger_moves: # If you want to go into a head
            neighbors = get_neighbors(head, walls.union(tail_poses))
            if len(neighbors) == 0: # If you want to go into a tail with food near head
                neighbors = get_neighbors(head, walls)
                if len(neighbors) == 0: return ['NORTH']

        # Getting NSEW from positions
        actions = [get_direction(head, pos) for pos in neighbors]
        return actions
    
    def score_action(self, obses, index, action):
        """
        Heuristic that scores actions without simulating next step

        Args:
            obses (list): Previous and current observations
            index (int): Index of agent to check
            action (str): NSEW action to score
        
        Returns:
            (tuple): Comparable score for action

        References:
            [1] https://github.com/m-schier/battlesnake-2019/blob/master/AI/Heuristics/ReflexEvasionHeuristic.cs
        """

        obs = obses[-1]
        next_head = DIRECTION[action](row_col(obs.geese[index][0]))

        opp_tails = list(map(lambda goose: row_col(goose[-1]) if len(goose) else (), obs.geese))
        opp_tails.pop(index)

        food = list(map(row_col, obs.food))
        lengths = list(map(len, obs.geese))

        food_score = float(next_head in food)
        tail_step = -float(next_head in opp_tails)
        if tail_step == 0:
            tail_step = float(next_head == row_col(obs.geese[index][-1]))

        rush_score = -dist(next_head, row_col(obs.geese[self.index][0]))
        food_dist_score = -min([dist(next_head, f) for f in food]) if len(food) else 0
        noise_score = float(np.random.random_sample())
        
        potential_collision = 0.0
        for i in active_indexes(change_index(obses, index)[-1]):
            if dist(row_col(obs.geese[i][0]), next_head) == 1:
                if (lengths[index] == max(lengths) and lengths.count(0) == 2) or (i == self.index and lengths[index] > lengths[i]):
                    potential_collision += 1.0
                else: potential_collision -= 1.0
        
        return (potential_collision, food_score, tail_step, rush_score, food_dist_score, noise_score)
    
    def depth_0(self, obses):
        """ (tuple): Net policy output with Rob's complex rules and FF """
        action, refined_policy = self.net_rules.evaluate(obses)
        return (action, self.eval(obses), refined_policy)

    def depth_0_basic(self, obses):
        """ (tuple): Net policy output masked with basic rules """

        raw_action_scores = self.eval(obses, mode = 'ACTION_SCORES')
        valid_moves = self.valid_moves(obses, obses[-1].index, danger_moves = False)

        action_scores = {move: raw_action_scores[move] for move in valid_moves}
        refined_action = max(action_scores, key = action_scores.get)

        return (refined_action, self.eval(obses), [raw_action_scores[act] for act in ACTIONS])
    
    def check_timeout(self):
        """ (raises TimeoutError): Quickly exiting recursion by raising exception """
        rot = self.obs['remainingOverageTime'] / (self.config.episodeSteps - self.obs.step)
        if time.perf_counter() - self.start_time > self.config.actTimeout + rot - 0.05: # or (LOCAL_MODE and self.count > 380):
            raise TimeoutError

    def terminal(self, position, index, depth, force_terminal = False):
        """ (tuple): Gives terminality and scores leaf nodes of tree """

        obs = position[-1]
        score = None

        # Lengths of position
        prev_lengths, lengths = [list(map(len, position[i].geese)) for i in range(2)]
        dead_agents = lengths.count(0)

        time_score = (obs.step / self.config.episodeSteps) * 1e-4

        # Collision or body hit score
        if lengths[index] == 0:
            score = self.terminal_scores[dead_agents - 1] + time_score
            if dead_agents >= prev_lengths.count(0) + 2:
                opp_index = [i for i in range(4) if 0 == lengths[i] < prev_lengths[i] and i != index][0]
                # Make sure we were actually close to them on the last step
                if dist(position[-2].geese[opp_index][0], position[-2].geese[index][0]) <= 2:
                    lost_collision = prev_lengths[index] < prev_lengths[opp_index]
                    score = self.collision_scores[dead_agents - lost_collision - 1] + time_score

        # Final score for ending simulation or last goose standing
        elif obs.step >= self.config.episodeSteps - 1 or (dead_agents == 3 and lengths[index] > 0):
            score = self.collision_scores[sorted(lengths).index(lengths[index])] + EPS + time_score

        # Heuristic call for depth exhaustion
        elif depth <= 0 or force_terminal: # and index == self.index:
            score = self.eval(change_index(position, index))
            if self.options.GREEDY_MODE and dead_agents == 2:
                advantage = lengths[index]
                if self.score_type == tuple:
                    score = score[:1] + (advantage,) + score[1:]
                else: score += advantage

        if type(None) != type(score) != self.score_type == tuple:
            score = (score,)
        
        terminal = score != None
        return (terminal, score)

    def alphabeta(self, position, depth, max_depth, alpha = -INF, beta_initial = INF):
        """ (dict): Runs a recursive alpha-beta search to score reachable positions. """

        # Checking timeouts
        self.check_timeout()

        # Main keys
        obs = KaggleObservation(position[-1])
        key = get_key(position)

        # Checking cache for repeat nodes
        entry = self.cache.get(key)
        if entry != None:
            if entry['DEPTH'] >= depth:
                self.count += 1
                return entry['ACTION_SCORES']

        # Score leaf nodes
        (terminal, score) = self.terminal(position, self.index, depth)
        if terminal:
            self.count += 1
            return { 'NORTH': score }

        # Search variables
        active_agents = active_indexes(obs) + [self.index]
        self_head = row_col(obs.geese[self.index][0])
        agent_moves = {}

        # TODO: No opponents ever kill eachother in tree

        # Generating valid moves for agents
        for agent in active_agents:

            # Predict next action of HandyRL copies
            if agent in self.hrl_agents:
                agent_moves[agent] = [hrl_evaluate(change_index(position, agent), mode = 'ACTION')]
                continue

            # TODO: Add logic for no reason that opponent will hit us
            check_danger = agent not in self.avoid_danger_agents or len(obs.geese[agent]) >= len(obs.geese[self.index])
            agent_moves[agent] = self.valid_moves(position, agent, danger_moves = check_danger)

            # Masking distant/unaffecting opponents
            if self.options.MASK_OPPONENTS and agent != self.index and len(agent_moves[agent]) > 1:

                # If head is too far away to affect us
                head_distance = dist(self_head, row_col(obs.geese[agent][0]))
                if head_distance > (depth * 2) + 2:
                    
                    # Check if all body parts are farther than threshold
                    distances = list(map(
                        lambda pos: dist(row_col(pos), self_head), 
                        obs.geese[agent][1:]))

                    # Pruning since body is too far to affect us
                    if len(distances) == 0 or min(distances) > depth + 1:
                        score_actions = { act: self.score_action(position, agent, act) for act in agent_moves[agent] }
                        agent_moves[agent] = [max(score_actions, key = score_actions.get)]

            if len(agent_moves[agent]) > 1:
                agent_moves[agent].sort(key = lambda a: self.score_action(position, agent, a), reverse = True)

        # Generating all opponent move combinations
        groups = [(agent_moves[i] if i in active_agents and i != self.index else [None]) for i in range(4)]
        chains = list(itertools.product(*groups))

        # Actions for all unmasked agents
        self_actions = agent_moves[self.index]
        action_chains = { act: chains for act in self_actions }

        # Updating Max-mean step knowledge
        max_steps = len(chains) * len(self_actions)
        steps = 0

        # Sort actions based on previous scores
        if entry != None:
            self_actions.sort(key = lambda a: entry['ACTION_SCORES'][a], reverse = True)
            for act in self_actions:
                action_chains[act].sort(key = lambda c: entry['CHAIN_SCORES'][act][c])

        # Scores for position
        action_scores = defaultdict(lambda: -INF)
        chain_scores = defaultdict(lambda: defaultdict(lambda: INF))
        sort_scores = defaultdict(lambda: self.MAX_SCORE)

        # Core actions loop
        for action in self_actions:

            # Reset beta for this branch
            beta = beta_initial
            mean_scores = []

            # Sort next action chains based on previous data
            if entry == None and action != self_actions[0]:
                action_chains[action].sort(key = lambda c: sort_scores[c])

            # Core opponent loop
            for chain in action_chains[action]:

                # Updating finished steps
                steps += 1

                # Setting up move chain
                move_chain = list(chain)
                move_chain[self.index] = action

                # Expand search step through environment
                next_position = [obs] + [simulate_step_2(obs, move_chain)]
                score = max(self.alphabeta(next_position, depth - 1, max_depth, alpha, beta).values())
                
                # Scoring chains for next sorting
                sort_scores[chain] = min(sort_scores[chain], score)
                chain_scores[action][chain] = score

                # Alpha cutoff
                if self.options.MAX_MEAN:
                    mean_scores.append(score)
                    beta = float(np.mean(mean_scores))
                    best_score = self.terminal_scores[min(4 - len(active_agents) + 2, 3)]
                    if np.mean(mean_scores + [best_score] * (max_steps - steps)) <= alpha:
                        break
                else:
                    beta = min(beta, score)
                    if beta <= alpha: break

            # Scoring action
            action_scores[action] = beta
            alpha = max(alpha, beta)

            # Beta cutoff (suboptimal)
            # if alpha >= beta_initial: break

        # Storing entry in cache
        self.cache.set(key, {
            'DEPTH': depth, 
            'ACTION_SCORES': action_scores, 
            'CHAIN_SCORES': chain_scores
        })

        # Returning values
        return action_scores

    def search(self, obses, start_depth = 1):
        """
        Runs an iterative deepening alpha-beta search before timeout.

        Args:
            obses (list): Previous and current observations
            start_depth (int = 1, optional): Start depth for iterative deepening

        Returns:
            (str): Action with the best minimax value
        """

        start_depth = max(start_depth, 1)
        depth = start_depth

        # Updating cache bounds to useful sections
        self.cache.update(obses[-1].step)
        for cache in SCORE_CACHE:
            SCORE_CACHE[cache].update(obses[-1].step)

        # Get output from net directional scores
        refined_action, raw_score, refined_policy = self.depth_0(obses)
        scores = { refined_action: raw_score }
        search_scores = scores.copy()
        using_policy = False

        # Iterative deepening loop
        while True:

            if obses[-1].step + depth > self.config.episodeSteps or round(max(search_scores.values()), 2) in self.terminal_scores or (self.options.CONFIDENT_POLICY and self.options.USE_POLICY and max(refined_policy) > 0.95):
                break

            try:

                search_scores = self.alphabeta(obses, depth, depth)
                depth += 1

                # Budgeting overage time: return only valid action
                if len(search_scores.keys()) == 1:
                    break

            except TimeoutError:
                break

        # Last fully completed depth
        depth -= 1

        # Using policy scores to settle LA ties
        # Might be unstable if didn't complete at least depth 1
        if self.options.USE_POLICY:
            close_scores = [act for act in search_scores.keys() if abs(search_scores[act] - max(search_scores.values())) < self.policy_epsilon]
            using_policy = (len(close_scores) > 1 or depth < start_depth or (self.options.CONFIDENT_POLICY and max(refined_policy) > 0.95))
            if using_policy: scores = { refined_action: raw_score}
            else: scores = search_scores

        # If used LA action, start next step from where we left off
        if max(scores, key = scores.get) == max(search_scores, key = search_scores.get):
            self.prev_depth = depth
        else: self.prev_depth -= 1

        action = max(scores, key = scores.get)
        v_action = action
        score = max(scores.values())

        if refined_policy[direction_to_int[action]] <= 1e-13 and action != refined_action:
            action = refined_action
            if VERBOSE: print(f'Overrode LA {v_action} with Rule {refined_action}', refined_policy)
        
        # Logging data

        if VERBOSE:

            elapsed = time.perf_counter() - self.start_time
            self.nps = round(self.count / elapsed, 3)

            # Rounding scores to clean up printing
            print(f'[Search] depth {start_depth} -> {depth}: { {a: round(s, 3) for a, s in sorted(search_scores.items(), key = lambda scores: scores[1], reverse = True)} }')
            
            # Printing step data
            print(f'[{self.obs.step:03}] | time: {round(elapsed, 3)} | depth = {depth} | {"Policy:" if using_policy else "Search:"} {v_action} = {round(scores[v_action], 3)} | nodes: {self.count} | NPS: {self.nps}\n')

        return action


# --------------------------------------------------
# main.py

# Clearing pycache
import sys
sys.dont_write_bytecode = True

# Importing modules

@wrapper_cache
def floodfill_net_evaluate(obses, mode = 'SCORE', clip = 6.0, tta = None):
    """ (float): Leaf node heuristic with net and FF """
    if mode == 'TEST': return float
    if mode in ('ACTION', 'ACTION_SCORES'):
        return net_evaluate(obses, mode, tta = tta)
    space = calc_score(obses[-1], max_needed_free_area = clip)[0]
    floodfill_score = (space - (clip / 2)) - float(space <= (clip / 2))
    net_score = max(net_evaluate(obses, mode), -0.4)
    return floodfill_score * (100 if options.GREEDY_MODE else 1) + net_score

class Agent:

    def __init__(self, config_dict, options = options):
        
        # Tracking positions
        self.config = Configuration(config_dict)
        self.options = options
        self.obses = []

        # Finding which opponents are HRL copies
        self.hrl_agents = []

        # Finding which opponents go into head positions
        self.danger_evidence = collections.defaultdict(bool)
        self.avoid_danger_agents = []
    
        self.tree = Tree(self.config, 
            scorer = floodfill_net_evaluate
            # scorer = net_evaluate
        )

    def __call__(self, obs_dict):

        obs = KaggleObservation(obs_dict)
        self.obses.append(obs)

        self.tree.update(self.obses)

        # Don't run search on first step
        if obs.step == 0:
            self.hrl_agents = self.avoid_danger_agents = active_indexes(obs)
            return self.tree.depth_0(self.obses)[0]

        # Opponent behavior detection
        if self.options.CHECK_HRL or self.options.CHECK_AVOID_DANGER:

            # Only check active agents
            self.hrl_agents = active_indexes(obs, self.hrl_agents)
            self.avoid_danger_agents = active_indexes(obs, self.avoid_danger_agents)

            # Checking which agents follow some detectable rules
            for agent in active_indexes(obs):
                
                # Checking previous action's direction
                last_position = row_col(self.obses[-2].geese[agent][0])
                curr_position = row_col(self.obses[-1].geese[agent][0])

                change_obses = change_index(self.obses[:-1], agent)
                last_action = get_direction(last_position, curr_position)

                # Check HRL rules
                if self.options.CHECK_HRL and agent in self.hrl_agents:

                    # Agent didn't use HRL action last step
                    if hrl_evaluate(change_obses, mode = 'ACTION') != last_action:
                        self.hrl_agents.remove(agent)
                
                # Check danger avoidance
                if self.options.CHECK_AVOID_DANGER and agent in self.avoid_danger_agents:

                    # Dangerous actions that the agent could take
                    danger_moves = set(self.tree.valid_moves(change_obses, agent, 
                        danger_moves = True)) - set(self.tree.valid_moves(change_obses, agent))

                    # Found evidence of dangerous encounter
                    if len(danger_moves):
                        self.danger_evidence[agent] = True

                    # Agent took a dangerous move when it had other safe moves
                    if last_action in danger_moves:
                        self.avoid_danger_agents.remove(agent)

            # Simplifying tree for predicted actions
            if self.options.CHECK_HRL and obs.step > 15 and len(self.hrl_agents):
                if VERBOSE: print(f'Active HRL Indexes: {self.hrl_agents}')
                self.tree.set_hrl(self.hrl_agents)
            
            # Setting agents that avoid danger positions (only if evidence is found)
            if self.options.CHECK_AVOID_DANGER and obs.step > 15 and len(self.avoid_danger_agents):
                danger_agents = [i for i in self.avoid_danger_agents if self.danger_evidence[i]]
                if VERBOSE and len(danger_agents): print(f'Danger Avoiders: {danger_agents}')
                self.tree.set_avoid_danger(danger_agents)

        action = self.tree.search(self.obses, start_depth = self.tree.prev_depth - 1)
        return action

agents = {}

def main(obs, config):
    """ Main agent runner """
    global agents
    if obs.index not in agents:
        agents[obs.index] = Agent(config)
    return agents[obs.index](obs)
