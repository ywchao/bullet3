import numpy as np
import os

import pybullet
import pybullet_data

from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
from pybullet_envs.hand_trackers import HumanHand20DOF, HumanHand20DOFPlay


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
    self.prev_jpos = None
    return MJCFBaseBulletEnv.reset(self)

  def step(self, a):
    if self.prev_jpos is None:
      self.prev_jpos = np.array([j.get_position() for j in self.robot.ordered_joints], dtype=np.float32)

    prev_pot = self.calc_potential(self.robot.frame + 1)

    self.robot.apply_action(a)
    self.scene.global_step()

    # Needs to increase frame counter before self.robot.calc_state().
    self.robot.frame += 1
    state = self.robot.calc_state()

    curr_pot = self.calc_potential(self.robot.frame)
    r_prog = np.mean(curr_pot - prev_pot)

    curr_jpos = np.array([j.get_position() for j in self.robot.ordered_joints], dtype=np.float32)
    real_jpos = self.robot.qpos[self.robot.frame][0:len(self.robot.ordered_joints) * 2:2]
    r_jpos = np.exp(-10.0000 * np.sum((curr_jpos - real_jpos)**2))

    self.rewards = [0.8333 * r_prog, 0.1667 * r_jpos]

    self.error = np.mean(-curr_pot * self.scene.dt) * 1000

    done = False
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True
    # Termination due to training data: only enforced in training.
    if self.robot.frame == len(self.robot.qpos) - 2:
      done = True

    self.prev_jpos = curr_jpos

    return state, sum(self.rewards), bool(done), {}

  def calc_potential(self, frame):
    kpts = np.vstack([self.robot.parts[k].get_position() for k in ('index3', 'mid3', 'ring3', 'pinky3', 'thumb3')])
    dist = np.linalg.norm(kpts - self.robot.kpts[frame], axis=1)
    return -dist / self.scene.dt


class HumanHand20DOFBulletEnv(HandTrackerBulletEnv):

  def __init__(self, robot=HumanHand20DOF(), render=False):
    HandTrackerBulletEnv.__init__(self, robot, render)


class HumanHand20DOFBulletEnvPlay(HumanHand20DOFBulletEnv):

  def __init__(self, robot=HumanHand20DOFPlay(), truth=HumanHand20DOF(), render=False):
    HumanHand20DOFBulletEnv.__init__(self, robot, render)
    self.truth = truth
    self.spheres = []

  def reset(self):
    r = HumanHand20DOFBulletEnv.reset(self)

    self.truth.np_random = self.np_random
    self.truth.reset(self._p)

    body_id = self.truth.robot_body.bodies[0]
    for j in range(-1, self.truth._p.getNumJoints(body_id)):
      self._p.setCollisionFilterGroupMask(body_id, j, collisionFilterGroup=0, collisionFilterMask=0)
    for j in range(-1, self.truth._p.getNumJoints(body_id)):
      self._p.changeVisualShape(body_id, j, rgbaColor=[0.7, 0.7, 0.7, 0.4])

    self.truth.reset_joint_position(self.robot.qpos[self.robot.frame])

    if not self.spheres:
      for i in range(5):
        self.spheres.append(self.truth._p.loadURDF(os.path.join(pybullet_data.getDataPath(), 'HumanHand20DOF/sphere.urdf')))
        self.truth._p.changeDynamics(self.spheres[i], -1, mass=0.0)
        self.truth._p.changeVisualShape(self.spheres[i], -1, rgbaColor=[1, 0, 0, 1])

    return r

  def step(self, a):
    state, reward, done, info = HumanHand20DOFBulletEnv.step(self, a)

    self.truth.reset_joint_position(self.robot.qpos[self.robot.frame])

    for j, d in enumerate(('index3', 'mid3', 'ring3', 'pinky3', 'thumb3')):
      self._p.resetBasePositionAndOrientation(self.spheres[j], self.robot.kpts[self.robot.frame][j], [0, 0, 0, 1])

    return state, reward, done, info
