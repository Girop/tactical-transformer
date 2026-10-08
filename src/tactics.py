from enum import IntEnum
from z3alpha.tactics.catalog import SOLVER_CATALOG, PREPROCESS_CATALOG, NAME_TO_ID


# The tactic ids from z3alpha are not densly packed, some ids correspond to nothing.
# As such, we need to map them to something that the model can work with, without the possibility of producing invalid strategies.
class NewCatalog:
    def __init__(self, preprocessors: dict[int, str], solvers: dict[int, str]) -> None:
        solver_actions, solver_to_old_id = self.__map_ids(solvers)
        preprocessor_actions, preprocessor_to_old_id = self.__map_ids(preprocessors, max(solver_actions.keys()) + 1)

        self.__to_name = solver_actions | preprocessor_actions
        self.__from_old_id = solver_to_old_id | preprocessor_to_old_id
        self.last_strat_id = max(self.__to_name.keys())
        self.valid_tactic_names = [*solver_actions.values(), *preprocessor_actions.values()]


    @staticmethod
    def __map_ids(catalog: dict[int, str], start=0) -> tuple[dict[int, str], dict[int, int]]:
        new_mapping = {}
        back_mapping = {}
        for idx, (old_idx, tactic) in enumerate(catalog.items()):
            new_mapping[idx + start] = tactic
            back_mapping[old_idx] = idx + start
        return new_mapping, back_mapping


    def special_tokens(self):
        return SpecialTacticsTokens.tokens()

    def vocab_size(self):
        return self.last_strat_id + len(self.special_tokens()) + 1

    def id_to_name(self, id: int) -> str:
        assert id in self.__to_name.keys(), f"Tried to convert invalid tactic into string: {id}"
        return self.__to_name[id]


    def name_to_id(self, name: str) -> int:
        return self.__from_old_id[NAME_TO_ID[name]]


    def tactics_to_text(self, strats: list[int]) -> str:
        ids = [id for id in strats if id not in SpecialTacticsTokens.tokens()]
        if len(ids) == 0:
            return ""
        names = [self.id_to_name(i) for i in ids]
        return f"(then {' '.join(names)})" if len(names) > 1 else names[0]


CATALOG = NewCatalog(PREPROCESS_CATALOG, SOLVER_CATALOG)

class SpecialTacticsTokens(IntEnum):
    __LAST_STRAT_ID = CATALOG.last_strat_id

    PAD_ID = __LAST_STRAT_ID + 1
    UNK_ID = __LAST_STRAT_ID + 2
    BOS_ID = __LAST_STRAT_ID + 3
    EOS_ID = __LAST_STRAT_ID + 4

    @classmethod
    def tokens(cls):
        return [cls.PAD_ID, cls.UNK_ID, cls.BOS_ID, cls.EOS_ID]

__all__ = ["SpecialTacticsTokens", "CATALOG"]

