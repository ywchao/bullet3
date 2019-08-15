import numpy as np
import os

import pybullet_data

from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
from pybullet_envs.hand_trackers import HumanHand20DOFFixedBaseMSRAP05, HumanHand20DOFFixedBaseMSRAP05Play
from pybullet_envs.hand_trackers import HumanHand20DOFFreedBaseMSRAP05, HumanHand20DOFFreedBaseMSRAP05Play


class HandTrackerBulletEnv(MJCFBaseBulletEnv):

  def __init__(self, robot, render=False):
    MJCFBaseBulletEnv.__init__(self, robot, render)
    self.error = None

  def create_single_player_scene(self, bullet_client):
    return SinglePlayerStadiumScene(bullet_client,
                                    gravity=9.8,
                                    timestep=0.0165 / 4,
                                    frame_skip=4)

  def reset(self):
    return MJCFBaseBulletEnv.reset(self)

  def step(self, a):
    self.pre_step()

    self.robot.apply_action(a)
    self.scene.global_step()

    # Needs to increase frame counter before self.robot.calc_state().
    self.robot.frame += 1
    state = self.robot.calc_state()

    self.post_step()

    self.rewards = self.calc_rewards()

    self.error = self.calc_error()

    done = not self.is_alive()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True
    if self.robot.frame == len(self.robot.qpos) - 2:
      done = True

    return state, sum(self.rewards), bool(done), {}

  def pre_step(self):
    pass

  def post_step(self):
    pass

  def calc_rewards(self):
    raise NotImplementedError

  def calc_error(self):
    raise NotImplementedError

  def is_alive(self):
    raise NotImplementedError


class HumanHand20DOFFixedBaseMSRAP05BulletEnv(HandTrackerBulletEnv):

  def __init__(self, robot=HumanHand20DOFFixedBaseMSRAP05(), render=False):
    HandTrackerBulletEnv.__init__(self, robot, render)

  def pre_step(self):
    dist = self.calc_dist(self.robot.frame + 1)
    self.prev_pot = -dist / self.scene.dt

  def post_step(self):
    dist = self.calc_dist(self.robot.frame)
    self.curr_pot = -dist / self.scene.dt
    self.dist = dist

  def calc_rewards(self):
    r_prog = np.mean(self.curr_pot - self.prev_pot)

    curr_jpos = np.array([j.get_position() for j in self.robot.ordered_joints], dtype=np.float32)
    real_jpos = self.robot.qpos[self.robot.frame][0:len(self.robot.ordered_joints) * 2:2]
    r_jpos = np.exp(-10.0000 * np.sum((curr_jpos - real_jpos)**2))

    return [0.8333 * r_prog, 0.1667 * r_jpos]

  def calc_error(self):
    return np.mean(self.dist) * 1000

  def is_alive(self):
    return True

  def calc_dist(self, frame):
    kpts = np.vstack([self.robot.parts[k].get_position() for k in self.robot.kpts_names])
    return np.linalg.norm(kpts - self.robot.kpts[frame], axis=1)


class HumanHand20DOFFixedBaseMSRAP05BulletEnvPlay(HumanHand20DOFFixedBaseMSRAP05BulletEnv):

  def __init__(self,
               robot=HumanHand20DOFFixedBaseMSRAP05Play(),
               truth=HumanHand20DOFFixedBaseMSRAP05Play(),
               render=False):
    HumanHand20DOFFixedBaseMSRAP05BulletEnv.__init__(self, robot, render)
    self.truth = truth
    self.spheres = []

  def reset(self):
    r = HumanHand20DOFFixedBaseMSRAP05BulletEnv.reset(self)

    truth_loaded = self.truth.loaded
    self.truth.np_random = self.np_random
    self.truth.reset(self._p)

    if not truth_loaded:
      body_id = self.truth.robot_body.bodies[self.truth.robot_body.bodyIndex]
      for j in range(-1, self._p.getNumJoints(body_id)):
        self._p.setCollisionFilterGroupMask(body_id, j, collisionFilterGroup=0, collisionFilterMask=0)
        self._p.changeVisualShape(body_id, j, rgbaColor=[0.7, 0.7, 0.7, 0.4])

    if not self.spheres:
      for i in range(5):
        self.spheres.append(
            self._p.loadURDF(os.path.join(pybullet_data.getDataPath(), 'HumanHand20DOF/sphere.urdf')))
        self._p.changeDynamics(self.spheres[i], -1, mass=0.0)
        self._p.changeVisualShape(self.spheres[i], -1, rgbaColor=[1, 0, 0, 1])

    return r

  def step(self, a):
    state, reward, done, info = HumanHand20DOFFixedBaseMSRAP05BulletEnv.step(self, a)

    self.truth.reset_joint_position(self.robot.qpos[self.robot.frame])

    for j, d in enumerate(self.robot.kpts_names):
      self._p.resetBasePositionAndOrientation(self.spheres[j], self.robot.kpts[self.robot.frame][j], [0, 0, 0, 1])

    return state, reward, done, info

  def is_alive(self):
    return True


class HumanHand20DOFFreedBaseMSRAP05BulletEnv(HandTrackerBulletEnv):

  def __init__(self, robot=HumanHand20DOFFreedBaseMSRAP05(), render=False):
    HandTrackerBulletEnv.__init__(self, robot, render)

  def post_step(self):
    self.dist = self.calc_dist(self.robot.frame)

  def calc_rewards(self):
    r_dist = np.exp(-2000 * np.sum(self.dist ** 2))
    return [1.0000 * r_dist]

  def calc_error(self):
    return np.mean(self.dist) * 1000

  def is_alive(self):
    return np.max(self.dist) < 0.05

  def calc_dist(self, frame):
    kpts = np.vstack([self.robot.parts[k].get_position() for k in self.robot.kpts_names])
    return np.linalg.norm(kpts - self.robot.kpts[frame], axis=1)


class HumanHand20DOFFreedBaseMSRAP05BulletEnvPlay(HumanHand20DOFFreedBaseMSRAP05BulletEnv):

  def __init__(self,
               robot=HumanHand20DOFFreedBaseMSRAP05Play(),
               truth=HumanHand20DOFFreedBaseMSRAP05Play(),
               render=False):
    HumanHand20DOFFreedBaseMSRAP05BulletEnv.__init__(self, robot, render)
    self.truth = truth
    self.spheres = []

  def reset(self):
    r = HumanHand20DOFFreedBaseMSRAP05BulletEnv.reset(self)

    truth_loaded = self.truth.loaded
    self.truth.np_random = self.np_random
    self.truth.reset(self._p)

    if not truth_loaded:
      body_id = self.truth.robot_body.bodies[0]
      for j in range(-1, self._p.getNumJoints(body_id)):
        self._p.setCollisionFilterGroupMask(body_id, j, collisionFilterGroup=0, collisionFilterMask=0)
        self._p.changeVisualShape(body_id, j, rgbaColor=[0.7, 0.7, 0.7, 0.4])
      body_id = self.robot.robot_body.bodies[0]
      for j in range(-1, self._p.getNumJoints(body_id)):
        self._p.changeVisualShape(body_id, j, rgbaColor=[0.7, 0.7, 0.7, 0.7])

    if not self.spheres:
      for i in range(34):
        self.spheres.append(
            self._p.loadURDF(os.path.join(pybullet_data.getDataPath(), 'HumanHand20DOF/sphere.urdf')))
        self._p.changeDynamics(self.spheres[i], -1, mass=0.0)
        if i < 17:
          self._p.changeVisualShape(self.spheres[i], -1, rgbaColor=[1, 0, 0, 1])
        else:
          self._p.changeVisualShape(self.spheres[i], -1, rgbaColor=[0, 1, 0, 1])

    return r

  def step(self, a):
    state, reward, done, info = HumanHand20DOFFreedBaseMSRAP05BulletEnv.step(self, a)

    self.truth.reset_joint_position(self.robot.qpos[self.robot.frame])

    for j, d in enumerate(self.robot.kpts_names):
      self._p.resetBasePositionAndOrientation(self.spheres[j], self.robot.kpts[self.robot.frame][j], [0, 0, 0, 1])
      self._p.resetBasePositionAndOrientation(self.spheres[j + 17], self.robot.parts[d].get_position(), [0, 0, 0, 1])

    return state, reward, done, info

  def is_alive(self):
    return True
