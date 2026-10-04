"""Robot binding tests; USD frame chains and live contacts are checked in Isaac Lab."""
from dataclasses import replace
from types import SimpleNamespace
import subprocess
import sys
import unittest
import numpy as np

from manipisa.adapters import DualUr5WujiConfig, SideBinding, ToolFrame, RigidTransform, WujiContact
from manipisa.adapters.dual_ur5_wuji import ARM_JOINTS, _actor_bindings, contact_body_name


def robot():
    names = list(ARM_JOINTS)+["L_arm_"+s for s in ARM_JOINTS]
    names += [f"{side}_finger{i}_joint{j}" for side in ("right","left") for i in range(1,6) for j in range(1,5)]
    # The physical articulation order must not determine instruction coordinates.
    return SimpleNamespace(joint_names=list(reversed(names)))


class BindingTests(unittest.TestCase):
    def test_import_does_not_load_isaac_or_robot_backends_into_runtime(self):
        code = "import sys,manipisa; assert 'isaaclab' not in sys.modules; assert 'manipisa.adapters.dual_ur5_wuji' not in sys.modules; from manipisa.adapters import DualUr5WujiAdapter; assert 'isaaclab' not in sys.modules and 'pxr' not in sys.modules"
        subprocess.run([sys.executable,"-c",code],check=True)

    def test_transform_composition_and_inverse_preserve_rotation_order(self):
        a=RigidTransform((1.,2.,3.),(2**-.5,0.,0.,2**-.5))
        b=RigidTransform((1.,0.,0.),(2**-.5,2**-.5,0.,0.))
        c=a.compose(b)
        np.testing.assert_allclose(c.position,(1.,3.,3.))
        np.testing.assert_allclose(c.orientation_wxyz,(.5,.5,.5,.5),atol=1e-12)
        ident=c.inverse().compose(c)
        np.testing.assert_allclose(ident.position,(0.,0.,0.),atol=1e-12)
        np.testing.assert_allclose(ident.orientation_wxyz,(1.,0.,0.,0.),atol=1e-12)

    def test_all_physical_joints_owned_once_with_deterministic_order(self):
        r=robot(); actors=_actor_bindings(r,DualUr5WujiConfig(),{"right":RigidTransform(),"left":RigidTransform()})
        owned=[n for b in actors.values() for n in b.joint_names]
        self.assertEqual(len(owned),52)
        self.assertEqual(len(set(owned)),52)
        self.assertEqual(set(owned),set(r.joint_names))
        self.assertEqual(actors['arm_a'].joint_names,ARM_JOINTS)
        self.assertEqual(actors['hand_b'].joint_names[:4],tuple(f"left_finger1_joint{i}" for i in range(1,5)))

    def test_distinct_left_right_palm_offsets_and_custom_actor_names(self):
        c=DualUr5WujiConfig(right=SideBinding("r_arm","r_hand","r_tcp","r_palm",ToolFrame("palm",RigidTransform((.01,0.,0.)))),
                           left=SideBinding("l_arm","l_hand","l_tcp","l_palm",ToolFrame("palm",RigidTransform((.02,0.,0.)))))
        transforms={"right":RigidTransform((0.,0.,.06)),"left":RigidTransform((0.,0.,.07),(0.,0.,0.,1.))}
        actors=_actor_bindings(robot(),c,transforms)
        np.testing.assert_allclose(actors['r_arm'].tcp_position,(.01,0.,.06))
        np.testing.assert_allclose(actors['l_arm'].tcp_position,(-.02,0.,.07))
        self.assertEqual(actors['l_hand'].body_name,"left_palm_link")

    def test_default_wrist_tcp_keeps_previous_move_semantics(self):
        actors=_actor_bindings(robot(),DualUr5WujiConfig(),{"right":RigidTransform((0.,0.,.1)),"left":RigidTransform((0.,0.,.2))})
        for actor in ("arm_a","arm_b"):
            self.assertEqual(actors[actor].tcp_position,(0.,0.,0.))

    def test_coupling_declarations_are_forwarded(self):
        config=DualUr5WujiConfig()
        config=replace(config,right=replace(config.right,arm_coupling_groups=("fixture",),hand_coupling_groups=("fixture",)))
        actors=_actor_bindings(robot(),config,{"right":RigidTransform(),"left":RigidTransform()})
        self.assertEqual(actors['arm_a'].coupling_groups,("fixture",))
        self.assertEqual(actors['hand_a'].coupling_groups,("fixture",))

    def test_wrong_robot_missing_joint_is_rejected(self):
        r=robot(); r.joint_names.remove("right_finger1_joint1")
        with self.assertRaisesRegex(ValueError,"right_finger1_joint1"):
            _actor_bindings(r,DualUr5WujiConfig(),{"right":RigidTransform(),"left":RigidTransform()})

    def test_contact_region_resolves_tip_not_another_finger(self):
        self.assertEqual(contact_body_name('right','thumb_tip'),'right_finger1_tip_link')
        self.assertEqual(contact_body_name('left','index_link4'),'left_finger2_link4')
        self.assertEqual(contact_body_name('left','little_tip'),'left_finger5_tip_link')
        self.assertEqual(contact_body_name('right','palm'),'right_palm_link')

    def test_bad_calibration_ids_and_regions_rejected(self):
        for f in (lambda:RigidTransform((0.,float('nan'),0.)),lambda:RigidTransform(orientation_wxyz=(2.,0.,0.,0.)),
                  lambda:ToolFrame('finger'),lambda:DualUr5WujiConfig(left=DualUr5WujiConfig().right),
                  lambda:WujiContact('x','left','index_tip','obj','/World/*'),
                  lambda:WujiContact('x','left','no_finger','obj','/World/Object'),
                  lambda:WujiContact('x','left','index_tip','obj','/World/Object',friction_lower_bound=-1.)):
            with self.assertRaises(ValueError): f()


if __name__ == '__main__': unittest.main()
