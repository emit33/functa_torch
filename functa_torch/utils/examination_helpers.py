import os


def get_last_checkpoint_path(checkpoint_dir):
    """Get the last checkpoint file in the directory."""

    # Sort files by modification time
    ckpts = os.listdir(checkpoint_dir)
    ckpts.sort(key=lambda f: int(f.split("_")[-1].removesuffix(".pth")))

    return os.path.join(checkpoint_dir, ckpts[-1])  # Return the most recent file
