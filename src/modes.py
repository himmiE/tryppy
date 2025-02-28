class TransformationMode:
    FINAL_ONLY = "final_only"
    SAVE_EVERY_STEP = "save_every_step"

def get_mode(mode):
    if mode == TransformationMode.FINAL_ONLY:
        return TransformationMode.FINAL_ONLY
    elif mode == TransformationMode.SAVE_EVERY_STEP:
        return TransformationMode.SAVE_EVERY_STEP
    else:
        raise ValueError(f"Unknown mode: {mode}")