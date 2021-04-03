"""
optimizer learning rate scheduling helpers
"""
from math import ceil


def noam_schedule(step, warmup_step=4000):
    if step <= warmup_step:
        return step / warmup_step
    return (warmup_step ** 0.5) * (step ** -0.5)


def warmup_linear(step, warmup_step, tot_step):
    if step < warmup_step:
        return step / warmup_step
    return max(0, (tot_step-step)/(tot_step-warmup_step))


def vqa_schedule(step, warmup_interval, decay_interval,
                 decay_start, decay_rate):
    """ VQA schedule from MCAN """
    if step < warmup_interval:
        return 1/4
    elif step < 2 * warmup_interval:
        return 2/4
    elif step < 3 * warmup_interval:
        return 3/4
    elif step >= decay_start:
        num_decay = ceil((step - decay_start) / decay_interval)
        return decay_rate ** num_decay
    else:
        return 1


def get_lr_sched(global_step, opts):
    # learning rate scheduling
    if opts.decay == 'linear':
        lr_this_step = opts.learning_rate * warmup_linear(
            global_step, opts.warmup_steps, opts.num_train_steps)
    elif opts.decay == 'invsqrt':
        lr_this_step = opts.learning_rate * noam_schedule(
            global_step, opts.warmup_steps)
    elif opts.decay == 'constant':
        lr_this_step = opts.learning_rate
    elif opts.decay == 'vqa':
        lr_this_step = opts.learning_rate * vqa_schedule(
            global_step, opts.warm_int, opts.decay_int,
            opts.decay_st, opts.decay_rate)
    if lr_this_step <= 0:
        # save guard for possible miscalculation of train steps
        lr_this_step = 1e-8
    return lr_this_step
