import numpy as np
import os

import pybullet_data

from pybullet_envs.robot_bases import URDFBasedRobot


class HandTrackerBase(URDFBasedRobot):

  def __init__(self, fn, robot_name, action_dim, obs_dim):
    URDFBasedRobot.__init__(self,
                            fn,
                            robot_name,
                            action_dim,
                            obs_dim,
                            basePosition=[0, 0, 0],
                            baseOrientation=[0, 0, 0, 1],
                            fixed_base=True,
                            self_collision=True)

  def robot_specific_reset(self, bullet_client):
    raise NotImplementedError

  def apply_action(self, a):
    raise NotImplementedError

  def calc_state(self):
    raise NotImplementedError


class HumanHand20DOF(HandTrackerBase):

  def __init__(self, fn, robot_name, action_dim, obs_dim):
    HandTrackerBase.__init__(self, fn, robot_name, action_dim, obs_dim)
    self.loaded = False

  def robot_specific_reset(self, bullet_client):
    if not self.loaded:
      self.loaded = True
      # Change RGBA color.
      body_id = self.robot_body.bodies[self.robot_body.bodyIndex]
      for j in range(-1, self._p.getNumJoints(body_id)):
        self._p.changeVisualShape(body_id, j, rgbaColor=[0.7, 0.7, 0.7, 1])
      # Set visualizer camera.
      self._p.resetDebugVisualizerCamera(
          cameraDistance=1, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 1])


class HumanHand20DOFFixedBase(HumanHand20DOF):

  def __init__(self):
    HumanHand20DOF.__init__(self, 'HumanHand20DOF/HumanHand20DOF.urdf', 'base_link', action_dim=20, obs_dim=60)

  def robot_specific_reset(self, bullet_client):
    HumanHand20DOF.robot_specific_reset(self, bullet_client)

    self.robot_body.reset_position([0, 0, 1])
    self.robot_body.reset_orientation([-1/np.sqrt(2), 0, 1/np.sqrt(2), 0])

    self.frame = self.reset_frame()
    self.reset_joint_position(self.qpos[self.frame])

  def reset_joint_position(self, qpos):
    for j, joint in enumerate(self.ordered_joints):
      joint.reset_current_position(qpos[2 * j], qpos[2 * j + 1])

  def apply_action(self, a):
    assert (np.isfinite(a).all())
    curr_jpos = np.array([j.get_position() for j in self.ordered_joints], dtype=np.float32)
    curr_jvel = np.array([j.get_velocity() for j in self.ordered_joints], dtype=np.float32)
    next_jvel = curr_jvel + a * 400 * 0.0165
    next_jpos = curr_jpos + next_jvel * 0.0165
    for n, j in enumerate(self.ordered_joints):
      j.set_position(next_jpos[n])

  def calc_state(self):
    curr_jpos = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
    next_jpos = self.qpos[self.frame + 1][0:len(self.ordered_joints) * 2:2].copy()
    return np.concatenate([curr_jpos] + [next_jpos])


class HumanHand20DOFFixedBaseMSRAP05(HumanHand20DOFFixedBase):

  def __init__(self):
    HumanHand20DOFFixedBase.__init__(self)
    data = np.load(os.path.join(pybullet_data.getDataPath(), 'HumanHand20DOF/motions/msra_P0_5_fixed_base.npz'))
    self.qpos = data['qpos']
    self.kpts = data['kpts']
    self.kpts_names = ('index3', 'mid3', 'ring3', 'pinky3', 'thumb3')

  def reset_frame(self):
    return self.np_random.randint(len(self.qpos) - 100)


class HumanHand20DOFFixedBaseMSRAP05Play(HumanHand20DOFFixedBaseMSRAP05):

  def reset_frame(self):
    return 0
