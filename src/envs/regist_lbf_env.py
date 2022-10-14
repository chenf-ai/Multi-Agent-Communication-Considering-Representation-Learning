import lbforaging
from gym.envs.registration import registry, register, make, spec

# register(
#     id="Foraging-8x8-2p-2f-s2",
#     entry_point="lbforaging.foraging:ForagingEnv",
#     kwargs={
#         "players": 2,
#         "max_player_level": 2,
#         "field_size": (8, 8),
#         "max_food": 2,
#         "sight": 2,
#         "max_episode_steps": 50,
#         "force_coop": True,
#         "grid_observation": [True, False],
#     },
# )

for i in range(8):
    register(
        id="Foraging-8x8-2p-2f-s{}".format(str(i+1)),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": 2,
            "max_player_level": 2,
            "field_size": (8, 8),
            "max_food": 2,
            "sight": i+1,
            "max_episode_steps": 50,
            "force_coop": True,
            "grid_observation": [True, False],
        },
    )

for i in range(11):
    register(
        id="Foraging-11x11-2p-2f-s{}".format(str(i+1)),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": 2,
            "max_player_level": 2,
            "field_size": (11, 11),
            "max_food": 2,
            "sight": i+1,
            "max_episode_steps": 50,
            "force_coop": True,
            "grid_observation": [True, False],
        },
    )

for i in range(12):
    register(
        id="Foraging-11x11-3p-4f-s{}".format(str(i+1)),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": 3,
            "max_player_level": 2,
            "field_size": (11, 11),
            "max_food": 4,
            "sight": i+1,
            "max_episode_steps": 50,
            "force_coop": True,
            "grid_observation": [True, False],
        },
    )