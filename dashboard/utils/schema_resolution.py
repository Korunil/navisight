# dashboard/utils/schema_resolution.py
from navisight.pipeline.feature_registry import SUPERCLASS_VOCAB

def resolve_superclass_id(df_window) -> int:
    """
    Natively parses structural token assignments on Polars dataframe structures.
    Prevents schema-layer code leaking directly inside inference service instances.
    """
    if "vessel_superclass_id" in df_window.columns and df_window["vessel_superclass_id"][0] is not None:
        return int(df_window["vessel_superclass_id"][0])
    if "shiptype" in df_window.columns and df_window["shiptype"][0] is not None:
        return SUPERCLASS_VOCAB.get(str(df_window["shiptype"][0]).lower().strip(), 0)
    return 0