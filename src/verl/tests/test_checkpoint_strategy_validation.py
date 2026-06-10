import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import _assert_checkpoint_strategy_mutually_exclusive


def test_checkpoint_strategy_conflict_raises():
    trainer_cfg = OmegaConf.create(
        {
            "max_actor_ckpt_to_keep": 2,
            "max_critic_ckpt_to_keep": None,
            "best_ckpt": {"enable": True, "top_k": 3},
        }
    )

    with pytest.raises(ValueError):
        _assert_checkpoint_strategy_mutually_exclusive(trainer_cfg)


def test_checkpoint_strategy_fifo_only_allowed():
    trainer_cfg = OmegaConf.create(
        {
            "max_actor_ckpt_to_keep": 2,
            "max_critic_ckpt_to_keep": None,
            "best_ckpt": {"enable": False, "top_k": None},
        }
    )

    _assert_checkpoint_strategy_mutually_exclusive(trainer_cfg)


def test_checkpoint_strategy_best_only_allowed():
    trainer_cfg = OmegaConf.create(
        {
            "max_actor_ckpt_to_keep": None,
            "max_critic_ckpt_to_keep": None,
            "best_ckpt": {"enable": True, "top_k": 3},
        }
    )

    _assert_checkpoint_strategy_mutually_exclusive(trainer_cfg)


def test_checkpoint_strategy_zero_fifo_treated_as_disabled():
    trainer_cfg = OmegaConf.create(
        {
            "max_actor_ckpt_to_keep": 0,
            "max_critic_ckpt_to_keep": 0,
            "best_ckpt": {"enable": True, "top_k": 1},
        }
    )

    _assert_checkpoint_strategy_mutually_exclusive(trainer_cfg)
