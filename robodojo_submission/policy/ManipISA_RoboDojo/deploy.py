"""One observation per action; the official evaluator owns scoring and termination."""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    try:
        while not TASK_ENV.is_episode_end():
            model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
            actions = model_client.call(func_name="get_action")
            if len(actions) != 1:
                raise ValueError("Expected exactly one action before a new observation")
            TASK_ENV.take_action(actions[0])
    finally:
        model_client.call(func_name="reset")


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError("Use eval_batch=false and --num_envs 1")
