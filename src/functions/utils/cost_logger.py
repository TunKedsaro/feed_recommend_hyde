from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict
import pandas as pd

COST_LOG_PATH = "cost_log.xlsx"

def append_cost_log(record: Dict[str, Any], file_path: str = COST_LOG_PATH) -> None:
    """
    Append one cost record into Excel file.
    Creates the file if it does not exist.
    """
    row = dict(record)
    row["logged_at"] = datetime.utcnow().isoformat()

    new_df = pd.DataFrame([row])

    if os.path.exists(file_path):
        old_df = pd.read_excel(file_path)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df

    df.to_excel(file_path, index=False)