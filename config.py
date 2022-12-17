from utils.data_utils import BasicArguments
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple


@dataclass
class TaskArguments:

    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use: IDoFT, FlakeFlagger"}
    )

    mode: Optional[str] = field(
        default=None,
        metadata={"help": "final_code, full_code"}
    )

    fold: int = field(
        default=None,
        metadata={"help": "folds:1-10"}
    )


parser = HfArgumentParser((BasicArguments(), TaskArguments()))
args = parser.parse_args()
