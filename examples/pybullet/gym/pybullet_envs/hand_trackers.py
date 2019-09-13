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
    self.kpts_names = ('index3', 'mid3', 'ring3', 'pinky3', 'thumb3')

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

  def reset_frame(self):
    return self.np_random.randint(len(self.qpos) - 100)


class HumanHand20DOFFixedBaseMSRAP05Play(HumanHand20DOFFixedBaseMSRAP05):

  def reset_frame(self):
    return 0


class HumanHand20DOFFreedBase(HumanHand20DOF):

  def __init__(self):
    HumanHand20DOF.__init__(self, 'HumanHand20DOF/HumanHand20DOFBaseJoint.urdf', 'world', action_dim=26, obs_dim=102)
    self.kpts_names = ('base_link', 'palm', 'index1', 'index2', 'index3', 'mid1', 'mid2', 'mid3', 'ring1', 'ring2',
                       'ring3', 'pinky1', 'pinky2', 'pinky3', 'thumb1', 'thumb2', 'thumb3')

  def robot_specific_reset(self, bullet_client):
    HumanHand20DOF.robot_specific_reset(self, bullet_client)

    self.frame = self.reset_frame()
    self.reset_joint_position(self.qpos[self.frame])

  def reset_joint_position(self, qpos):
    self.ordered_joints[0].reset_current_position(qpos[-12], qpos[-6])
    self.ordered_joints[1].reset_current_position(qpos[-11], qpos[-5])
    self.ordered_joints[2].reset_current_position(qpos[-10] + 1, qpos[-4])
    self._p.resetJointStateMultiDof(self.ordered_joints[3].bodies[self.ordered_joints[3].bodyIndex],
                                    self.ordered_joints[3].jointIndex,
                                    targetValue=self._p.getQuaternionFromEuler(qpos[-9:-6]),
                                    targetVelocity=qpos[-3:])
    for j, joint in enumerate(self.ordered_joints[4:24]):
      joint.reset_current_position(qpos[2 * j], qpos[2 * j + 1])

  def apply_action(self, a):
    assert (np.isfinite(a).all())
    # Base translation.
    curr_ppos, curr_pvel = np.transpose(np.array([j.get_state() for j in self.ordered_joints[0:3]], dtype=np.float32))
    next_pacc = np.array(self._p.multiplyTransforms([0, 0, 0], self.base_rot, a[:3], [0, 0, 0, 1])[0], dtype=np.float32)
    next_pvel = curr_pvel + next_pacc * 100 * 0.0165
    next_ppos = curr_ppos + next_pvel * 0.0165
    self._p.setJointMotorControl2(self.ordered_joints[0].bodies[self.ordered_joints[0].bodyIndex],
                                  self.ordered_joints[0].jointIndex,
                                  self._p.POSITION_CONTROL,
                                  targetPosition=next_ppos[0],
                                  positionGain=0.1)
    self._p.setJointMotorControl2(self.ordered_joints[1].bodies[self.ordered_joints[1].bodyIndex],
                                  self.ordered_joints[1].jointIndex,
                                  self._p.POSITION_CONTROL,
                                  targetPosition=next_ppos[1],
                                  positionGain=0.1)
    self._p.setJointMotorControl2(self.ordered_joints[2].bodies[self.ordered_joints[2].bodyIndex],
                                  self.ordered_joints[2].jointIndex,
                                  self._p.POSITION_CONTROL,
                                  targetPosition=next_ppos[2],
                                  positionGain=0.1)
    # Base rotation.
    curr_opos, curr_ovel, _, _ = self._p.getJointStateMultiDof(
        self.ordered_joints[3].bodies[self.ordered_joints[3].bodyIndex],
        self.ordered_joints[3].jointIndex)
    next_ovel = curr_ovel + a[3:6] * 10000 * 0.0165
    axis, angle = next_ovel / np.linalg.norm(next_ovel), np.linalg.norm(next_ovel) * 0.0165
    _, next_opos = self._p.multiplyTransforms(
        [0, 0, 0], curr_opos, [0, 0, 0], self._p.getQuaternionFromAxisAngle(axis, angle))
    self._p.setJointMotorControlMultiDof(self.ordered_joints[3].bodies[self.ordered_joints[3].bodyIndex],
                                         self.ordered_joints[3].jointIndex,
                                         self._p.POSITION_CONTROL,
                                         targetPosition=next_opos,
                                         positionGain=0.5)
    # Joint positions.
    curr_jpos = np.array([j.get_position() for j in self.ordered_joints[4:24]], dtype=np.float32)
    curr_jvel = np.array([j.get_velocity() for j in self.ordered_joints[4:24]], dtype=np.float32)
    next_jvel = curr_jvel + a[6:26] * 500 * 0.0165
    next_jpos = curr_jpos + next_jvel * 0.0165
    for n, j in enumerate(self.ordered_joints[4:24]):
      j.set_position(next_jpos[n])

  def calc_state(self):
    pos, rot, _, _, _, _ = self._p.getLinkState(self.parts['base_link'].bodies[self.parts['base_link'].bodyIndex],
                                                self.parts['base_link'].bodyPartIndex)
    self.base_pos = pos
    self.base_rot = rot
    inv_pos, inv_rot = self._p.multiplyTransforms(
        [0, 0, 0], [-rot[0], -rot[1], -rot[2], rot[3]], [-pos[0], -pos[1], -pos[2]], [0, 0, 0, 1])
    curr_pos = np.vstack([
        self._p.multiplyTransforms(
            inv_pos, inv_rot, self.parts[k].get_position(), [0, 0, 0, 1])[0] for k in self.kpts_names
    ])
    next_pos = np.vstack([
        self._p.multiplyTransforms(
            inv_pos, inv_rot, kpt, [0, 0, 0, 1])[0] for kpt in self.kpts[self.frame + 1]
    ])

    def speed(bodyUniqueId, linkIndex):
      _, _, _, _, _, _, (vx, vy, vz), _ = self._p.getLinkState(bodyUniqueId, linkIndex, computeLinkVelocity=1)
      return np.array([vx, vy, vz])

    curr_vel = np.vstack([
        self._p.multiplyTransforms([0, 0, 0],
                                   inv_rot,
                                   speed(self.parts[k].bodies[self.parts[k].bodyIndex], self.parts[k].bodyPartIndex),
                                   [0, 0, 0, 1])[0] for k in self.kpts_names
    ])
    return np.concatenate([(next_pos - curr_pos).flatten()] + [curr_vel.flatten()])


class HumanHand20DOFFreedBaseMSRAP05(HumanHand20DOFFreedBase):

  def __init__(self):
    HumanHand20DOFFreedBase.__init__(self)
    data = np.load(os.path.join(pybullet_data.getDataPath(), 'HumanHand20DOF/motions/msra_P0_5_freed_base.npz'))
    self.qpos = data['qpos']
    self.kpts = data['kpts']

  def reset_frame(self):
    return self.np_random.randint(len(self.qpos) - 100)


class HumanHand20DOFFreedBaseMSRAP05Play(HumanHand20DOFFreedBaseMSRAP05):

  def reset_frame(self):
    return 0
