from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseTask(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_data(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        pass