import flax.nnx as nnx
import orbax.checkpoint as ocp
import os

def load_checkpoint(model, optimizer, path):
    """Restores model and optimizer state from an Orbax checkpoint.

    Args:
        model: An initialized nnx.Module (the target structure).
        optimizer: An initialized nnx.Optimizer (the target structure).
        path: Path to the specific checkpoint directory (e.g. "ckpt/step_100").
    """
    
    # 1. Create the checkpointer
    checkpointer = ocp.StandardCheckpointer()

    # 2. Define the target structure
    # This acts as a template. Orbax uses this to know what shapes/types 
    # it should expect to find in the file.
    target_state = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
    }

    # 3. Restore data
    # This reads the files and returns a dictionary matching the structure of `target_state`
    restored_data = checkpointer.restore(os.path.abspath(path), target_state)

    # 4. Inject data back into the objects
    # This updates the mutable weights inside `model` and `optimizer` in-place.
    nnx.update(model, restored_data["model"])
    nnx.update(optimizer, restored_data["optimizer"])

    print(f"Successfully loaded checkpoint from {path}")

# --- Example Usage ---

# model = MyModel(...)
# optimizer = nnx.Optimizer(model, ...)
# load_checkpoint(model, optimizer, "/path/to/checkpoints/checkpoint_0")