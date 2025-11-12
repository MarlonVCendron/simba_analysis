import re
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from analysis.consts import COMPLETE_RATS, BASE_PATH

# ELM_MS_PATH = Path('/home/marlon/edu/mestrado/comp_neuroetho/keypoint_comp_neuroetho/projects/elm_ms/data/vids/')
# SESSION_DIRS = {
#   'H1': ELM_MS_PATH / 'H1',
#   'H2': ELM_MS_PATH / 'H2',
#   'H3': ELM_MS_PATH / 'H3',
#   'S1': ELM_MS_PATH / 'S1',
#   'S2': ELM_MS_PATH / 'S2',
#   'T': ELM_MS_PATH / 'T'
# }

GROUP_INDEX_PATH = BASE_PATH / 'data' / 'group_index.csv'
H1_DIR = BASE_PATH / 'simba' / 'dist_H1' / 'project_folder' / 'csv' / 'input_csv'
H2_DIR = BASE_PATH / 'simba' / 'dist_H2' / 'project_folder' / 'csv' / 'input_csv'
H3_DIR = BASE_PATH / 'simba' / 'dist_H3' / 'project_folder' / 'csv' / 'input_csv'
S1_DIR = BASE_PATH / 'simba' / 's1_zones' / 'project_folder' / 'csv' / 'input_csv'
S2_DIR = BASE_PATH / 'simba' / 's2_zones' / 'project_folder' / 'csv' / 'input_csv'
T_DIR = BASE_PATH / 'simba' / 'TEST_zones' / 'project_folder' / 'csv' / 'input_csv'
POLYGONS_S1_PATH = BASE_PATH / 'data' / 'polygons' / 'polygons_s1.csv'
POLYGONS_S2_PATH = BASE_PATH / 'data' / 'polygons' / 'polygons_s2.csv'
POLYGONS_T_PATH = BASE_PATH / 'data' / 'polygons' / 'polygons_t.csv'

# S1_DIR = BASE_PATH / 'simba' / 's1_zones' / 'project_folder' / 'csv' / 'machine_results'
# S2_DIR = BASE_PATH / 'simba' / 's2_zones' / 'project_folder' / 'csv' / 'machine_results'
# T_DIR = BASE_PATH / 'simba' / 'TEST_zones' / 'project_folder' / 'csv' / 'machine_results'


SESSION_DIRS = {
  'H1': H1_DIR,
  'H2': H2_DIR,
  'H3': H3_DIR,
  'S1': S1_DIR,
  'S2': S2_DIR,
  'T': T_DIR
}
   
POLYGON_PATHS = {
  'S1': POLYGONS_S1_PATH,
  'S2': POLYGONS_S2_PATH,
  'T': POLYGONS_T_PATH
}


def load_group_index():
    df = pd.read_csv(GROUP_INDEX_PATH)
    group_map = {}
    for _, row in df.iterrows():
        name = row['name']
        match = re.match(r'(R\d+G\d+)', name)
        if match:
            rat_id = match.group(1)
            group = row['group'].lower()
            if group == 'salina':
                group = 'saline'
            group_map[rat_id] = group
    return group_map


def load_polygons():
    polygons_dict = {'S1': None, 'S2': None, 'T': None}
    
    for session, path in POLYGON_PATHS.items():
        df = pd.read_csv(path)
        polygons_dict[session] = df
    
    return polygons_dict


def get_rat_id(file: str) -> str:
    match = re.match(r'(R\d+G\d+)', file)
    if not match:
        return None
    return match.group(1)

def load_single_file(args: Tuple[Path, str]) -> Optional[Tuple[str, str, pd.DataFrame]]:
    file, session = args
    rat_id = get_rat_id(file.stem)
    if rat_id is None:
        return None

    try:
        df = pd.read_csv(file, header=[0, 1, 2], low_memory=False)
        return (rat_id, session, df)
    except Exception:
        return None


def read_raw_dlc(skip_incomplete = True) -> Dict[str, Dict[str, pd.DataFrame]]:
    data = {}
    
    files_to_load = []
    for session, directory in SESSION_DIRS.items():
        for file in directory.glob('*.csv'):
            rat_id = get_rat_id(file.stem)
            if rat_id is None or (skip_incomplete and rat_id not in COMPLETE_RATS):
                continue
            files_to_load.append((file, session))
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(load_single_file, files_to_load),
            total=len(files_to_load),
            desc="Loading DLC files",
            unit="file"
        ))
    
    for result in results:
        if result is None:
            continue
        rat_id, session, df = result
        if rat_id not in data:
            data[rat_id] = {}
        data[rat_id][session] = df
    
    return data


def find_rat_files(rat_id: str):
    files = {}
    for sess, directory in SESSION_DIRS.items():
        found = next(directory.glob(f"{rat_id}{sess}.csv"), None)
        files[sess] = found
    return files


def get_bodypart_coords(df: pd.DataFrame, bodypart: str) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    x = None
    y = None
    for col in df.columns:
        if isinstance(col, tuple) and len(col) == 3:
            if col[1] == bodypart:
                if col[2] == 'x':
                    x = df[col]
                elif col[2] == 'y':
                    y = df[col]
    return x, y


def iter_bodyparts(df: pd.DataFrame, bodyparts):
    for bodypart in bodyparts:
        x, y = get_bodypart_coords(df, bodypart)
        if x is not None and y is not None:
            yield bodypart, x, y

