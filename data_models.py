from enum import Enum


class WealthIncomeMismatchType(str, Enum):
    IncomeRichWealthPoor = "Income-rich, wealth-poor"
    Aligned = "Aligned"
    IncomePoorWealthRich = "Income-poor, wealth-rich"

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
