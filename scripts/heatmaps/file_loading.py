import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

BASE_PATH = Path(__file__).parent.parent.parent
H1_DIR = BASE_PATH / 'simba' / 'dist_H1' / 'project_folder' / 'csv' / 'input_csv'
H2_DIR = BASE_PATH / 'simba' / 'dist_H2' / 'project_folder' / 'csv' / 'input_csv'
H3_DIR = BASE_PATH / 'simba' / 'dist_H3' / 'project_folder' / 'csv' / 'input_csv'
S1_DIR = BASE_PATH / 'simba' / 's1_zones' / 'project_folder' / 'csv' / 'machine_results'
S2_DIR = BASE_PATH / 'simba' / 's2_zones' / 'project_folder' / 'csv' / 'machine_results'
T_DIR = BASE_PATH / 'simba' / 'TEST_zones' / 'project_folder' / 'csv' / 'machine_results'
GROUP_INDEX_PATH = BASE_PATH / 'data' / 'group_index.csv'
POLYGONS_S1_PATH = BASE_PATH / 'data' / 'polygons' / 'polygons_s1.csv'
POLYGONS_S2_PATH = BASE_PATH / 'data' / 'polygons' / 'polygons_s2.csv'
POLYGONS_T_PATH = BASE_PATH / 'data' / 'polygons' / 'polygons_t.csv'


def load_group_index() -> Dict[str, str]:
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


def load_polygons() -> Dict[str, Optional[pd.DataFrame]]:
    polygons_dict = {'S1': None, 'S2': None, 'T': None}
    
    polygon_paths = {
        'S1': POLYGONS_S1_PATH,
        'S2': POLYGONS_S2_PATH,
        'T': POLYGONS_T_PATH
    }
    
    for session, path in polygon_paths.items():
        try:
            if not path.exists():
                print(f"Warning: Polygons file not found at {path}")
                continue
            df = pd.read_csv(path)
            polygons_dict[session] = df
        except Exception as e:
            print(f"Error loading polygons for {session}: {e}")
    
    return polygons_dict


def read_dlc_csv(filepath: Path, body_part: str = 'mid_mid', include_rearing: bool = False) -> Optional[pd.DataFrame]:
    try:
        df = None
        is_multi_index = False
        
        try:
            df = pd.read_csv(filepath, skipinitialspace=True)
        except:
            pass
        
        if df is None:
            try:
                df = pd.read_csv(filepath, header=[0, 1, 2], skipinitialspace=True)
                if isinstance(df.columns[0], tuple):
                    is_multi_index = True
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                return None
        
        if df is None:
            return None
        
        bodypart_x_col = None
        bodypart_y_col = None
        bodypart_likelihood_col = None
        rearing_col = None
        
        for col in df.columns:
            if is_multi_index and isinstance(col, tuple) and len(col) == 3:
                if col[1] == body_part:
                    if col[2] == 'x':
                        bodypart_x_col = col
                    elif col[2] == 'y':
                        bodypart_y_col = col
                    elif col[2] == 'likelihood' or col[2] == 'p':
                        bodypart_likelihood_col = col
            elif not is_multi_index:
                col_str = str(col).lower()
                if col_str == f'{body_part}_x':
                    bodypart_x_col = col
                elif col_str == f'{body_part}_y':
                    bodypart_y_col = col
                elif col_str == f'{body_part}_p' or col_str == f'{body_part}_likelihood':
                    bodypart_likelihood_col = col
                elif 'rearing' in col_str:
                    rearing_col = col
        
        if bodypart_x_col is None or bodypart_y_col is None:
            return None
        
        bodypart_x = df[bodypart_x_col].values
        bodypart_y = df[bodypart_y_col].values
        
        if bodypart_x.ndim > 1:
            bodypart_x = bodypart_x.flatten()
        if bodypart_y.ndim > 1:
            bodypart_y = bodypart_y.flatten()
        
        mask = np.isfinite(bodypart_x) & np.isfinite(bodypart_y)
        
        if bodypart_likelihood_col is not None:
            bodypart_likelihood = df[bodypart_likelihood_col].values
            if bodypart_likelihood.ndim > 1:
                bodypart_likelihood = bodypart_likelihood.flatten()
            mask = mask & (bodypart_likelihood > 0.1)
        
        if not np.any(mask):
            return None
        
        result_data = {
            'x': bodypart_x[mask],
            'y': bodypart_y[mask]
        }
        
        if include_rearing and rearing_col is not None:
            rearing_values = df[rearing_col].values
            if rearing_values.ndim > 1:
                rearing_values = rearing_values.flatten()
            result_data['rearing'] = rearing_values[mask]
        
        result = pd.DataFrame(result_data)
        
        return result if len(result) > 0 else None
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_rat_files(rat_id: str) -> Dict[str, Optional[Path]]:
    sessions = {
        'H1': H1_DIR,
        'H2': H2_DIR,
        'H3': H3_DIR,
        'S1': S1_DIR,
        'S2': S2_DIR,
        'T': T_DIR
    }
    files = {}
    for sess, directory in sessions.items():
        found = next(directory.glob(f"{rat_id}{sess}.csv"), None)
        files[sess] = found
    return files

