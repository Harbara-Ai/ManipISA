"""Use only RoboDojo's public evaluation methods."""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    try:
        while not TASK_ENV.is_episode_end():
            model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
            actions = model_client.call(func_name="get_action")
            if len(actions) != 1:
                raise ValueError("ManipISA requires exactly one action per observation")
            TASK_ENV.take_action(actions[0])
    finally:
        # Terminate the agent even if the final take_action ended the episode
        # before another observation was delivered. Do not send score labels.
        model_client.call(func_name="reset")


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError("ManipISA integration requires eval_batch=false")
