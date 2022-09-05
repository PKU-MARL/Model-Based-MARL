import numpy as np
from copy import deepcopy
import gym
from gym.spaces import Box, Discrete
from gym.envs.registration import register

from ..envs.flow.envs.ring.wave_attenuation import WaveAttenuationEnv
from ..envs.flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from ..envs.flow.core.params import VehicleParams, SumoCarFollowingParams
from ..envs.flow.controllers import RLController, IDMController, ContinuousRouter
from ..envs.flow.networks import RingNetwork

class RingAttenuationWrapper(WaveAttenuationEnv):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator=simulator)
        self.n_agent = self.initial_vehicles.num_vehicles
        self.n_s_ls, self.n_a_ls, self.coop_gamma, self.distance_mask, self.neighbor_mask \
            = [], [], -1, np.zeros((self.n_agent, self.n_agent)), np.zeros((self.n_agent, self.n_agent))
        self.init_neighbor_mask()
        self.init_distance_mask()
        self.n_s_ls = [2] * self.n_agent
        self.n_a_ls = [1] * self.n_agent
        self.target_vel = 20.

        self.action_mean = (self.action_space.high - self.action_space.low) / 2
        self.action_bias = (self.action_space.high + self.action_space.low) / 2
        self.state_mean = (self.observation_space.high - self.observation_space.low) / 2
        self.state_bias = (self.observation_space.high + self.observation_space.low) / 2
        
        
        #self.neighbor_mask = np.identity(self.n_agent)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ['Velocity', 'Absolute_pos']
        return Box(
            low=0,
            high=1,
            shape=(2, ),
            dtype=np.float32)

    @property
    def sorted_ids(self):
        """Sort the vehicle ids of vehicles in the network by position.

        This environment does this by sorting vehicles by their absolute
        position, defined as their initial position plus distance traveled.

        Returns
        -------
        list of str
            a list of all vehicle IDs sorted by position
        """
        return self.k.vehicle.get_ids()

    def get_state_(self):
        """See class definition."""

        # for veh_id in self.sorted_ids:
        #     print('speed=',self.k.network.max_speed())
        #     print('po=',self.k.network.length())
        
        speed = [self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
                 for veh_id in self.sorted_ids]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.network.length()
               for veh_id in self.sorted_ids]
        speed = np.array(speed).reshape((-1, 1))
        pos = np.array(pos).reshape((-1, 1))
        # print('speed=',speed)
        # print('position=',pos)
        return np.concatenate([speed, pos], axis=-1)
    
    def step(self, rl_actions):
        if rl_actions is not None:
            while rl_actions.ndim > 1:
                rl_actions = rl_actions.squeeze(-1)
        _, _, d, info = super().step(rl_actions)
        s1 = self.get_state_()
        # print('action=',rl_actions)        
        # print('state=',s1)
        
        r = self.get_reward_(rl_actions)
        d = np.array([d] * self.n_agent, dtype=np.bool)
        return s1, r, d, info
        
    
    def get_reward_(self, rl_actions):
        if rl_actions is None:
            return np.zeros((self.n_agent,), dtype=np.float)
        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.sorted_ids
        ])

        if any(vel < -100):
            return np.zeros((self.n_agent,), dtype=np.float)

        # reward average velocity
        eta_2 = 4.
        reward = eta_2 * (self.target_vel - np.abs(self.target_vel - np.array(vel))) / (20 * self.n_agent)

        # punish accelerations (should lead to reduced stop-and-go waves)
        eta = 4  # 0.25
        mean_actions = np.abs(np.array(rl_actions)) / self.n_agent
        accel_threshold = 0
        reward += eta * (accel_threshold - mean_actions)

        return reward

    def init_neighbor_mask(self):
        n = self.n_agent
        for i in range(n):
            self.neighbor_mask[i][i] = 1
            self.neighbor_mask[i][(i+1)%n] = 1
            self.neighbor_mask[i][(i+n-1)%n] = 1

    def init_distance_mask(self):
        n = self.n_agent
        for i in range(n):
            for j in range(n):
                self.distance_mask[i][j] = min((i-j+n)%n, (j-i+n)%n)
    
    def rescaleReward(self, ep_return, ep_len):
        return ep_return

def makeRingAttenuation(evaluate=False, version=0, render=None):
    # time horizon of a single rollout
    HORIZON = 3000
    NUM_VEHICLES = 22

    vehicles = VehicleParams()
        # Add one automated vehicle.
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
        ),
        num_vehicles=NUM_VEHICLES)


    flow_params = dict(
        # name of the experiment
        exp_tag="ring_attenuation",

        # name of the flow environment the experiment is running on
        env_name=RingAttenuationWrapper,

        # name of the network class the experiment is running on
        network=RingNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.1,
            render=render,
            restart_instance=False,
            no_step_log=True,
            print_warnings=False
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            warmup_steps=0,
            clip_actions=False,
            additional_params={
                "max_accel": 1,
                "max_decel": 1,
                "ring_length": [220, 270],
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            additional_params={
                "length": 260,
                "lanes": 1,
                "speed_limit": 30,
                "resolution": 40,
            }, ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(),
    )

    params = flow_params
    exp_tag = params["exp_tag"]
    base_env_name = params["env_name"].__name__

    # deal with multiple environments being created under the same name
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    while "{}-v{}".format(base_env_name, version) in env_ids:
        version += 1
    env_name = "{}-v{}".format(base_env_name, version)
    network_class = params["network"]

    env_params = params['env']
    net_params = params['net']
    initial_config = params.get('initial', InitialConfig())
    sim_params = deepcopy(params['sim'])
    vehicles = deepcopy(params['veh'])

    network = network_class(
        name=exp_tag,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
    )

    # accept new render type if not set to None
    sim_params.render = render or sim_params.render

    entry_point = params["env_name"].__module__ + ':' + params["env_name"].__name__

    # register the environment with OpenAI gym
    register(
        id=env_name,
        entry_point=entry_point,
        kwargs={
            "env_params": env_params,
            "sim_params": sim_params,
            "network": network,
            "simulator": params['simulator']
        })

    return gym.envs.make(env_name)
