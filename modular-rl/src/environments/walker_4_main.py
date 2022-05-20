import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from utils import *
import os


class ModularEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml):
        self.xml = xml
        mujoco_env.MujocoEnv.__init__(self, xml, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        qpos_before = self.sim.data.qpos
        qvel_before = self.sim.data.qvel
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter = self.sim.data.qpos[0]
        torso_height, torso_ang = self.sim.data.qpos[1:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (
            torso_height > (0.8 - 0.136)
            and torso_height < (2.0 - 0.136)
            and torso_ang > -1.0
            and torso_ang < 1.0
        )
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        def _get_obs_per_limb(b):
            if b == "torso":
                limb_type_vec = np.array((1, 0, 0, 0))
            elif "1" in b:
                limb_type_vec = np.array((0, 1, 0, 0))
            elif "2" in b:
                limb_type_vec = np.array((0, 0, 1, 0))
            elif "3" in b:
                limb_type_vec = np.array((0, 0, 0, 1))
            else:
                limb_type_vec = np.array((0, 0, 0, 0))
            torso_x_pos = self.data.get_body_xpos("torso")[0]
            xpos = self.data.get_body_xpos(b)
            xpos[0] -= torso_x_pos
            q = self.data.get_body_xquat(b)
            expmap = quat2expmap(q)
            
            
            # include current joint angle and joint range as input
            if b == "torso":
                angle = 0.0
                joint_range = [0.0, 0.0]
            else:

                body_id = self.sim.model.body_name2id(b)
                jnt_adr = self.sim.model.body_jntadr[body_id]
                qpos_adr = self.sim.model.jnt_qposadr[
                    jnt_adr
                ]  # assuming each body has only one joint
                angle = np.degrees(
                    self.data.qpos[qpos_adr]
                )  # angle of current joint, scalar
                joint_range = np.degrees(
                    self.sim.model.jnt_range[jnt_adr]
                )  # range of current joint, (2,)
                # normalize
                angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
                joint_range[0] = (180.0 + joint_range[0]) / 360.0
                joint_range[1] = (180.0 + joint_range[1]) / 360.0
            
            obs = np.zeros(15 + len(limb_type_vec))
            idx = 12 + len(limb_type_vec)
            obs[0:3] = xpos
            obs[3:6] = np.clip(self.data.get_body_xvelp(b), -10, 10)
            obs[6:9] = self.data.get_body_xvelr(b)
            obs[9:12] = expmap
            obs[12:idx] = limb_type_vec
            obs[idx] = angle
            obs[idx+1:] = joint_range
            
            return obs
        
        full_obs = np.concatenate(
            [_get_obs_per_limb(b) for b in self.model.body_names[1:]]
        )

        return full_obs.ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
