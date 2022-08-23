from tensorflow.python.keras import backend as K


def get_schedule(base_lr, warmup_steps):
    def schedule(optimizer, current_step):
        current_step = max(1, current_step)

        if current_step < warmup_steps:
            warmup_lr = base_lr * current_step / warmup_steps
            K.set_value(optimizer.lr, K.get_value(warmup_lr))
        return

    return schedule