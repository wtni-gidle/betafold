"""Utils for geometry library."""

import dataclasses
from typing import List


def get_field_names(cls) -> List[str]:
    fields = dataclasses.fields(cls)
    field_names = [f.name for f in fields]
    return field_names