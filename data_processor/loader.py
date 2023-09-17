from abc import ABC, abstractmethod
from typing import List


class Loader(ABC):
    @abstractmethod
    def get_data(self) -> List[str]:
        ...
