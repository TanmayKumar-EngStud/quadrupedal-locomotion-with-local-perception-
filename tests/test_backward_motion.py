from types import SimpleNamespace
import pytest
torch = pytest.importorskip('torch')
from quadruped_nav.rl import rewards
from quadruped_nav.rl.config import EnvConfig, default_rewards


def test_backward_motion_charges_only_backward_body_speed_while_walking_to_a_target():
    s = SimpleNamespace(lin_vel_body=torch.tensor([[.5, .2, 0.], [-.6, .1, 0.], [-.4, 0., 0.]]),
                        moving=torch.tensor([True, True, False]))
    assert torch.allclose(rewards.TERMS['backward_motion'](s, {}), torch.tensor([0., .6, 0.]))


def test_backward_motion_is_off_unless_a_config_sets_it():
    assert default_rewards()['backward_motion'] == 0
    assert EnvConfig().rewards['backward_motion'] == 0
