from pydantic import BaseModel

class SatImage(BaseModel):
    r_energy     : float
    r_correlation: float
    r_contrast   : float
    r_homogeneity: float
    g_energy     : float
    h_correlation: float
    s_correlation: float
    s_contrast   : float