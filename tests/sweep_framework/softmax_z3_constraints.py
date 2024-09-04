from abc import ABC
from dataclasses import dataclass
import datetime
from time import time
from typing import Dict, List, Optional, Tuple
from z3 import *
from enum import Enum


class TTNNLayout(Enum):
    TILE_LAYOUT = 0
    ROW_MAJOR_LAYOUT = 1


class TTNNDataFormat(Enum):
    FLOAT32 = 0
    BFLOAT16 = 1
    BFLOAT8_B = 2


class TTNNShardingStrategy(Enum):
    NONE = 0
    BLOCK = 1
    HEIGHT = 2
    WIDTH = 3


class TTNNShardOrientation(Enum):
    ROW_MAJOR = 0
    COL_MAJOR = 1


class TTNNMemoryConfig(Enum):
    DRAM = 0
    L1 = 1


def make_int_var(solver: Solver, name: str, min_val: Optional[int], max_val: Optional[int]) -> z3.Int:
    int_var = Int(name)
    if min_val is not None:
        solver.add(int_var >= min_val)
    if max_val is not None:
        solver.add(int_var <= max_val)

    return int_var


def enum_min_max(enum_type: type) -> Tuple[int, int]:
    enum_vals = [x.value for x in enum_type]
    return min(enum_vals), max(enum_vals)


def make_int_var_from_enum(solver: Solver, name: str, enum_type: type) -> z3.Int:
    return make_int_var(solver, name, *enum_min_max(enum_type))


class ConstraintHolder(ABC):
    def __init__(self, solver: Solver, var_prefix: str) -> None:
        self._solver = solver
        self._var_prefix = var_prefix

        self.solver_vars = {}

    def register_var(self, var_name: str, min_val: Optional[int] = None, max_val: Optional[int] = None) -> None:
        prefixed_var_name = f"{self._var_prefix}_{var_name}"
        setattr(
            self,
            var_name,
            make_int_var(self._solver, prefixed_var_name, min_val, max_val),
        )

        self.solver_vars[prefixed_var_name] = getattr(self, var_name)

    def register_enum_var(self, var_name: str, enum_type: type) -> None:
        self.register_var(var_name, *enum_min_max(enum_type))

    def register_bool_var(self, var_name: str) -> None:
        self.register_var(var_name, 0, 1)


class SoftmaxInput(ConstraintHolder):
    def __init__(self, solver: Solver, var_prefix: str) -> None:
        super().__init__(solver, var_prefix)

        self.register_var("height", 1)
        self.register_var("width", 1)

        self.register_enum_var("dtype", TTNNDataFormat)
        self.register_enum_var("layout", TTNNLayout)
        self.register_enum_var("memory_config", TTNNMemoryConfig)
        self.register_enum_var("sharding_strategy", TTNNShardingStrategy)
        self.register_enum_var("shard_orientation", TTNNShardOrientation)


class MultiCoreProgramConfig(ConstraintHolder):
    def __init__(self, solver: Solver, var_prefix: str) -> None:
        super().__init__(solver, var_prefix)

        self.register_var("compute_with_storage_grid_size_x", 1, 8)
        self.register_var("compute_with_storage_grid_size_y", 1, 8)
        self.register_var("subblock_w", 1)
        self.register_var("block_h", 1)
        self.register_var("block_w", 1)


class SoftmaxConfig(ConstraintHolder):
    def __init__(self, solver: Solver, var_prefix: str) -> None:
        super().__init__(solver, var_prefix)

        self.register_var("batch_size", 1)
        self.register_var("num_inputs", 1, 2)

        self.register_bool_var("is_scale_causal_mask_hw_dims_softmax")
        self.register_bool_var("is_inplace")
        self.register_bool_var("is_causal_mask")


def different_solution_constraint_smt2_str(vars_values: Dict[str, int], vars_names: List[str]) -> str:
    if not vars_names:
        return "(assert true)"

    smt2_string = "(assert (or"

    for var_name in vars_names:
        smt2_string = f"{smt2_string} (distinct {var_name} {vars_values[var_name]})"

    smt2_string = f"{smt2_string}))"

    return smt2_string


def main():
    solver = Solver()

    input_a = SoftmaxInput(solver, "input_a")
    input_b = SoftmaxInput(solver, "input_b")
    multi_core_program_config = MultiCoreProgramConfig(solver, "multi_core_program_config")
    softmax_config = SoftmaxConfig(solver, "softmax_config")

    # Op constraints
    solver.add(
        input_a.layout == TTNNLayout.TILE_LAYOUT.value,
        If(
            softmax_config.num_inputs == 2,
            True,  # TODO: Implement constraints for 2 inputs
            softmax_config.is_scale_causal_mask_hw_dims_softmax == 1,
        ),
    )

    # Optimizer constraints
    solver.add(
        softmax_config.batch_size == 1,
        softmax_config.num_inputs == 2,
        input_a.height == 1024,
        input_a.width == 1024,
        input_b.height == 1024,
        input_b.width == 1024,
        # input_a.sharding_strategy == TTNNShardingStrategy.BLOCK.value,
        # input_a.shard_orientation == TTNNShardOrientation.ROW_MAJOR.value,
        # input_a.layout == TTNNLayout.TILE_LAYOUT.value,
        # input_a.memory_config == TTNNMemoryConfig.L1.value,
        # input_b.sharding_strategy == TTNNShardingStrategy.BLOCK.value,
        # input_b.shard_orientation == TTNNShardOrientation.ROW_MAJOR.value,
        # input_b.layout == TTNNLayout.TILE_LAYOUT.value,
        # input_b.memory_config == TTNNMemoryConfig.L1.value,
        multi_core_program_config.compute_with_storage_grid_size_x == 8,
        multi_core_program_config.compute_with_storage_grid_size_y == 8,
        multi_core_program_config.subblock_w == 8,
        multi_core_program_config.block_h == 12 * 2,
        multi_core_program_config.block_w == 24,
    )

    solver_vars = {
        **input_a.solver_vars,
        **input_b.solver_vars,
        **multi_core_program_config.solver_vars,
        **softmax_config.solver_vars,
    }

    solutions = []
    i = 0

    t1 = time()
    print(f"Start time: {datetime.datetime.utcnow()}")

    while True:
        if solver.check() == z3.unsat:
            print(f"Failed to generate new solutions. Total generated solutions {i}")
            break

        model = solver.model()

        solver_vars_values = {}
        for name, z3_var in solver_vars.items():
            solver_vars_values[name] = model[z3_var].as_long()

        solutions.append(solver_vars_values)

        solver.add(
            parse_smt2_string(
                different_solution_constraint_smt2_str(solver_vars_values, solver_vars.keys()),
                decls=solver_vars,
            )
        )

        i += 1
        if i % 1000 == 0:
            print(f"Generated solution number {i}")

    t2 = time()
    print(f"Elapsed: {t2 - t1} seconds")

    for solution in solutions:
        print("=================")
        print(solution)
        print("=================")


if __name__ == "__main__":
    main()
