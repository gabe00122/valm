from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx.filterlib import Filter
from jax.sharding import Mesh


class Checkpointer:
    def __init__(self, directory: str):
        if not directory.startswith("gs://"):
            directory = Path(directory).absolute().as_posix()
        self.mngr = ocp.CheckpointManager(directory)

    def save(
        self, data: dict[str, Any], global_step: int, param_filter: Filter = nnx.Param
    ):
        data_state = {}
        for key, value in data.items():
            data_state[key] = ocp.args.StandardSave(nnx.state(value, param_filter))

        self.mngr.save(global_step, args=ocp.args.Composite(**data_state))

    def restore(
        self, data: dict[str, Any], step: int, param_filter: Filter = nnx.Param
    ):
        device = jax.devices()[0]
        mesh = Mesh((device,), ("batch",))

        data_abstract_state = {}
        restorable_keys = []

        for key, value in data.items():
            if value is not ocp.PLACEHOLDER:
                restorable_keys.append(key)
                value_state = nnx.state(value, param_filter)
                abstract_state = jax.tree.map(
                    lambda x, s: jax.ShapeDtypeStruct(
                        shape=x.shape, dtype=x.dtype, sharding=s
                    ),
                    value_state,
                    nnx.get_named_sharding(value_state, mesh),
                )
                data_abstract_state[key] = ocp.args.StandardRestore(abstract_state)

        restored_state = self.mngr.restore(
            step, args=ocp.args.Composite(**data_abstract_state)
        )

        for key, value in restored_state.items():
            nnx.update(data[key], value)

    def restore_latest(self, model, param_filter: Filter = nnx.Param) -> int:
        step = self.mngr.latest_step()
        if step is None:
            return 0
        self.restore(model, step, param_filter)
        return step

    def close(self):
        self.mngr.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
