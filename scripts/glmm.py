import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from statsmodels.formula.api import mixedlm
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM, BinomialBayesMixedGLM
from statsmodels.genmod.families.family import Poisson, Binomial, Gamma


from utils import fig_path, get_rearing, session_types, get_area_and_direction_columns, group_areas_and_directions


def glmm(df, session_type):
    data = df.copy()
    data, area_columns, direction_columns = group_areas_and_directions(data, session_type)
    
    formula = 'rearing ~ group'
    for area in area_columns:
        formula += f' + {area}'
    for direction in direction_columns:
        formula += f' + {direction}'
    
    vcf = {"video": "0 + C(video)", "group": "0 + C(group)"}
    
    
    model = PoissonBayesMixedGLM.from_formula(formula, vc_formulas=vcf, data=data)
    # model = BinomialBayesMixedGLM.from_formula(formula, vc_formulas=vcf, data=data)

    result = model.fit_vb()

    print(result.summary())
    