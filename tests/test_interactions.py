"""Adversarial evidence/contract tests; physics acceptance is a separate script."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np

from manipisa import (Runtime, Instruction, Opcode, Mode, Status, Truth, Evidence, Pose, JointState,
    StateSnapshot, ContactPoint, ContactState, ContactRequirement, BoundedMotion,
    ShapeGoal, ContactGoal, GraspGoal, WrenchGoal)
from manipisa.types import PreparedTask, validate_instruction, AdapterError
from manipisa.evidence import evaluate_goal
from manipisa.interaction import wrench_at, grasp_capacity
from manipisa.adapters.interaction_control import InteractionControl, InteractionPlan
from manipisa.adapters.physx_contacts import PhysXContactSource, ContactBinding


REQ = (ContactRequirement("left",.1,10.,.003), ContactRequirement("right",.1,10.,.003))
MOTIONS = (BoundedMotion("hand",ShapeGoal((.01,-.01)),max_configuration_delta=.03),)


def contact(name, time=0., force=1., present=True, gap=0.):
    left = name == "left"
    normal = (1.,0.,0.) if left else (-1.,0.,0.)
    position = (-.02,0.,0.) if left else (.02,0.,0.)
    return ContactState("hand","object",time,present,force,tuple(force*x for x in normal),(0.,0.,0.),
                        (ContactPoint(position,normal,force),) if present else (),gap,True,True)


def state(time=0., force=1., present=True, gap=0.):
    pose = Pose((0.,0.,0.),(1.,0.,0.,0.),(0.,0.,0.),(0.,0.,0.),time)
    return StateSnapshot(time,0,{"hand":JointState((0.,0.),(0.,0.),time)}, {"object":pose,"palm":pose},
                         contacts={n:contact(n,time,force,present,gap) for n in ("left","right")})


def request(op=Opcode.MAKE_CONTACT, **kw):
    if op == Opcode.CONTROL_GRASP:
        goal = GraspGoal(REQ,MOTIONS,"object","palm",((0.,0.,-.5,0.,0.,0.),),.5)
    elif op == Opcode.APPLY_WRENCH:
        goal = WrenchGoal(REQ,"palm",(0.,0.,0.,0.,0.,0.),"world",(0.,0.,0.))
    else:
        goal = ContactGoal(REQ,MOTIONS)
    return Instruction(op,("hand",),goal,mode=Mode.SUSTAIN if op == Opcode.APPLY_WRENCH else Mode.REACH,
                       execution_domain="bounded_contact",**kw)


class ContactAdapter(InteractionControl):
    def __init__(self):
        self.now, self.dt = 0., .05
        self.force, self.present, self.gap = 0.,False,.01
        self.valid = True
        self.next = None
        self.calls = 0
        self.contact_source = SimpleNamespace(bindings={n:SimpleNamespace(actor="hand") for n in ("left","right")})
    def observe(self):
        s = state(self.now,self.force,self.present,self.gap)
        return replace(s, contacts={k:replace(c,valid=self.valid) for k,c in s.contacts.items()})
    def prepare(self,command,snapshot):
        if not isinstance(command.goal,(ContactGoal,GraspGoal,WrenchGoal)):
            raise AdapterError("UNSUPPORTED_CAPABILITY","Contact test double")
        return PreparedTask(frozenset(("joints","left_pair","right_pair")),frozenset(("object",)),binding=
            InteractionPlan(command.actors,{"hand":(0.,0.)},{"hand":snapshot.frames["palm"]},(np.zeros(3),np.eye(3))))
    def command(self,command,prepared):
        self.calls += 1
        self.next = command.operation
    def hold(self,prepared):
        self.next = None
    def step(self):
        self.now = round(self.now+self.dt,10)
        if self.next == Opcode.MAKE_CONTACT:
            self.present,self.force,self.gap = True,1.,0.
        if self.next == Opcode.BREAK_CONTACT:
            self.present,self.force,self.gap = False,0.,.01


class InteractionTests(unittest.TestCase):
    def test_make_contact_requires_all_actual_pairs(self):
        s = state()
        s.contacts["right"] = replace(s.contacts["right"],present=False,normal_force=0.)
        self.assertEqual(evaluate_goal(request(),s).evidence.truth,Truth.VIOLATED)
    def test_break_zero_force_is_insufficient(self):
        cmd = request(Opcode.BREAK_CONTACT)
        self.assertEqual(evaluate_goal(cmd,state(force=0.,present=False,gap=0.)).evidence.truth,Truth.VIOLATED)
        self.assertEqual(evaluate_goal(cmd,state(force=0.,present=False,gap=.01)).evidence.truth,Truth.SATISFIED)
    def test_break_missing_gap_is_unknown(self):
        self.assertEqual(evaluate_goal(request(Opcode.BREAK_CONTACT),state(gap=None)).evidence.truth,Truth.UNKNOWN)
    def test_contact_stale_and_invalid_cannot_succeed(self):
        for c in (replace(contact("left"),timestamp=-1.),replace(contact("left"),valid=False),replace(contact("left"),normal_force=float("nan"))):
            s = state(); s.contacts["left"] = c
            self.assertEqual(evaluate_goal(request(),s).evidence.truth,Truth.UNKNOWN)
    def test_wrench_requires_friction_validity(self):
        s = state(); s.contacts["left"] = replace(s.contacts["left"],wrench_valid=False)
        self.assertEqual(evaluate_goal(request(Opcode.APPLY_WRENCH),s).evidence.truth,Truth.UNKNOWN)
    def test_wrench_reference_point_moment_and_rotated_axes(self):
        cmd = request(Opcode.APPLY_WRENCH)
        goal = replace(cmd.goal,contacts=(REQ[0],),reference_frame="palm",reference_point=(1.,0.,0.))
        s = state(); s.frames["palm"] = replace(s.frames["palm"],orientation_wxyz=(2**-.5,0.,0.,2**-.5))
        measured, *_ = wrench_at(goal,cmd,s)
        # World force +X; ref point is world +Y; torque about it is +Z.
        np.testing.assert_allclose(measured,(0.,-1.,0.,0.,0.,1.),atol=1e-8)
    def test_grasp_load_capacity_is_not_contact_existence(self):
        cmd = request(Opcode.CONTROL_GRASP)
        self.assertTrue(grasp_capacity(cmd.goal,cmd,state()))
        self.assertFalse(grasp_capacity(replace(cmd.goal,load_cases=((0.,0.,-5.,0.,0.,0.),)),cmd,state()))
    def test_grasp_missing_contact_geometry_is_unknown(self):
        cmd = request(Opcode.CONTROL_GRASP); s = state()
        s.contacts["left"] = replace(s.contacts["left"],points=())
        self.assertIsNone(grasp_capacity(cmd.goal,cmd,s))
    def test_grasp_velocity_cannot_be_hidden_by_feasible_load(self):
        s = state(); s.frames["object"] = replace(s.frames["object"],linear_velocity=(0.,0.,1.))
        self.assertEqual(evaluate_goal(request(Opcode.CONTROL_GRASP),s).evidence.truth,Truth.VIOLATED)
    def test_invalid_modes_dimensions_contacts_and_bounds_rejected(self):
        cmd = request()
        bad = [replace(cmd,mode=Mode.SUSTAIN),replace(cmd,goal=replace(cmd.goal,contacts=REQ+REQ)),
               replace(cmd,goal=replace(cmd.goal,forbidden_contacts=("left",))),
               replace(cmd,goal=replace(cmd.goal,motions=(replace(MOTIONS[0],max_configuration_delta=-1.),))),
               replace(request(Opcode.APPLY_WRENCH),mode=Mode.REACH),
               replace(request(Opcode.CONTROL_GRASP),goal=replace(request(Opcode.CONTROL_GRASP).goal,load_cases=((0.,)*6,)))]
        for c in bad:
            with self.assertRaises(AdapterError): validate_instruction(c)
    def test_runtime_make_then_break_explicit_handoff(self):
        adapter=ContactAdapter(); runtime=Runtime(adapter)
        call=runtime.submit(request(dwell_s=.1))
        for _ in range(4): runtime.step()
        done=runtime.query(call.call_id)
        self.assertEqual(done.status,Status.SUCCEEDED)
        self.assertEqual(runtime.submit(request(Opcode.BREAK_CONTACT)).reason,"RESOURCE_CONFLICT")
        release=runtime.submit(request(Opcode.BREAK_CONTACT),replace_handle=done.control_handle)
        for _ in range(4): runtime.step()
        self.assertEqual(runtime.query(release.call_id).status,Status.SUCCEEDED)
    def test_grasp_and_wrench_cannot_acquire_missing_contact(self):
        for op in (Opcode.CONTROL_GRASP,Opcode.APPLY_WRENCH):
            adapter=ContactAdapter(); runtime=Runtime(adapter)
            self.assertEqual(runtime.submit(request(op)).status,Status.REJECTED)
            self.assertEqual(adapter.calls,0)
    def test_entry_grasp_requires_load_certificate_if_requested(self):
        adapter=ContactAdapter(); adapter.present=True; adapter.force=.2
        runtime=Runtime(adapter); c=request(Opcode.CONTROL_GRASP)
        c=replace(c,goal=replace(c.goal,require_valid_at_entry=True))
        self.assertEqual(runtime.submit(c).status,Status.REJECTED)
    def test_contact_loss_is_terminal_and_cannot_silently_regrasp(self):
        adapter=ContactAdapter(); adapter.present=True; adapter.force=1.
        runtime=Runtime(adapter); c=replace(request(Opcode.CONTROL_GRASP,dwell_s=0.),mode=Mode.SUSTAIN)
        call=runtime.submit(c); runtime.step()
        self.assertEqual(runtime.query(call.call_id).status,Status.ACTIVE)
        adapter.present=False; adapter.force=0.; runtime.step()
        self.assertEqual(runtime.query(call.call_id).reason,"CONTACT_LOST")
        adapter.present=True; adapter.force=1.; runtime.step()
        self.assertEqual(runtime.query(call.call_id).status,Status.FAILED)
    def test_overforce_fails_before_further_actuation(self):
        adapter=ContactAdapter(); runtime=Runtime(adapter)
        call=runtime.submit(request()); adapter.force=11.; adapter.present=True
        runtime.step()
        self.assertEqual(runtime.query(call.call_id).reason,"FORCE_LIMIT")
        self.assertEqual(adapter.calls,0)
    def test_forbidden_contact_and_protected_loss(self):
        adapter=ContactAdapter(); adapter.present=True; adapter.force=1.
        c=request(); plan=adapter.prepare(c,state()); s=state()
        s.contacts["extra"]=contact("extra")
        forbidden=replace(c,goal=replace(c.goal,forbidden_contacts=("extra",)))
        self.assertEqual(adapter.check(forbidden,plan,s)["forbidden_contact:extra"].truth,Truth.VIOLATED)
        s.contacts["extra"]=replace(s.contacts["extra"],present=False,normal_force=0.)
        protected=replace(c,goal=replace(c.goal,protected_contacts=(ContactRequirement("extra"),)))
        self.assertEqual(adapter.check(protected,plan,s)["keep_contact:extra"].truth,Truth.VIOLATED)
    def test_acquisition_bound_cannot_be_reset_by_goal_update(self):
        adapter=ContactAdapter(); runtime=Runtime(adapter); call=runtime.submit(request())
        self.assertFalse(runtime.update(call.call_id,request())["accepted"])
        s=state(); s.joints["hand"]=replace(s.joints["hand"],position=(.1,-.1))
        plan=adapter.prepare(request(),state())
        self.assertEqual(adapter.check(request(),plan,s)["acquisition:hand"].truth,Truth.VIOLATED)
    def test_grasp_drift_fails_even_with_existing_contacts(self):
        adapter=ContactAdapter(); c=request(Opcode.CONTROL_GRASP); plan=adapter.prepare(c,state())
        s=state(); s.frames["object"]=replace(s.frames["object"],position=(0.,0.,.1))
        self.assertEqual(adapter.check(c,plan,s)["grasp_relative_motion"].truth,Truth.VIOLATED)
    def test_requested_friction_cannot_exceed_binding_calibration(self):
        class Source:
            bindings = {n:SimpleNamespace(actor="hand",target="object",friction_lower_bound=.2) for n in ("left","right")}
            def pair_key(self,n): return n
        adapter=ContactAdapter(); adapter.contact_source=Source()
        with self.assertRaisesRegex(AdapterError,"calibrated"):
            adapter.prepare_interaction(request(Opcode.CONTROL_GRASP),state())
    def test_contact_feedback_is_strict_json_with_numpy_scalars(self):
        import json
        s=state(); s.contacts["left"]=replace(s.contacts["left"],force_on_target_w=tuple(np.array((1.,0.,0.),dtype=np.float32)))
        json.dumps(s.to_dict(),allow_nan=False)
    def test_grasp_active_load_loss_has_specific_failure_reason(self):
        adapter=ContactAdapter(); adapter.present=True; adapter.force=1.
        runtime=Runtime(adapter); c=replace(request(Opcode.CONTROL_GRASP,dwell_s=0.),mode=Mode.SUSTAIN)
        call=runtime.submit(c); runtime.step(); adapter.force=.2; runtime.step()
        self.assertEqual(runtime.query(call.call_id).reason,"GRASP_INVALID")
    def test_admittance_preserves_preload_and_integrates_target(self):
        import torch
        from manipisa.adapters.isaaclab import IsaacLabAdapter
        c=request(Opcode.APPLY_WRENCH)
        c=replace(c,actors=("arm",),goal=replace(c.goal,contacts=(REQ[0],),wrench=(2.,0.,0.,0.,0.,0.)))
        s=state()
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros((1,1))),
            root_physx_view=SimpleNamespace(get_jacobians=lambda:torch.tensor([[[[1.],[0.],[0.],[0.],[0.],[0.]]]])))
        r=SimpleNamespace(spec=SimpleNamespace(articulation=robot),body=1,joints=[0])
        adapter=IsaacLabAdapter.__new__(IsaacLabAdapter)
        adapter.dt=.01; adapter._actors={"arm":r}; adapter._targets={id(robot):torch.tensor([[.001]])}
        adapter.math=SimpleNamespace(skew_symmetric_matrix=lambda x:torch.zeros((1,3,3)))
        adapter._tcp=lambda r:(None,None,None,None,torch.zeros((1,3)))
        adapter.observe=lambda:s
        goals=[]; adapter._drive_target=lambda command,resolved,goal:goals.append(goal)
        plan=PreparedTask(frozenset(("arm",)),frozenset(),binding=InteractionPlan(("arm",),{}, {"arm":s.frames["palm"]}))
        adapter.command_interaction(c,plan); adapter.command_interaction(c,plan)
        self.assertGreater(goals[0].position[0],.001)
        self.assertGreater(goals[1].position[0],goals[0].position[0])


class RawView:
    def get_contact_data(self,dt):
        return np.array([2.]),np.array([[0.,1.,0.]]),np.array([[-1.,0.,0.]]),np.array([-.001]),np.array([1]),np.array([0])
    def get_contact_force_matrix(self,dt): return np.array([[-2.,0.,0.]])
    def get_friction_data(self,dt):
        return np.array([[0.,-3.,0.]]),np.array([[1.,0.,0.]]),np.array([1]),np.array([0])


class PhysXDecodeTests(unittest.TestCase):
    def source(self,view=None):
        src=PhysXContactSource.__new__(PhysXContactSource)
        src.bindings={"pair":ContactBinding("hand","obj","/A","/B",lambda now:(-.001,now))}
        src.dt=.01; src.capacity=10; src._timestamp=None; src._cache={}; src._views={"pair":view or RawView()}
        return src
    def test_sign_friction_and_per_patch_moment_are_preserved(self):
        c=self.source().observe(.1)["pair"]
        self.assertTrue(c.wrench_valid)
        np.testing.assert_allclose(c.force_on_target_w,(2.,3.,0.))
        np.testing.assert_allclose(c.torque_on_target_world_origin_w,(0.,0.,1.))
    def test_buffer_or_aggregate_inconsistency_is_invalid(self):
        class Wrong(RawView):
            def get_contact_force_matrix(self,dt): return np.array([[99.,0.,0.]])
        self.assertFalse(self.source(Wrong()).observe(.1)["pair"].valid)
    def test_friction_exception_cannot_fabricate_full_wrench(self):
        class Missing(RawView):
            def get_friction_data(self,dt): raise RuntimeError("not available")
        c=self.source(Missing()).observe(.1)["pair"]
        self.assertTrue(c.valid)
        self.assertFalse(c.wrench_valid)
    def test_stale_separation_is_unknown_not_zero(self):
        src=self.source(); src.bindings["pair"]=replace(src.bindings["pair"],separation=lambda now:(1.,now-1.))
        self.assertIsNone(src.observe(.1)["pair"].separation_m)


if __name__ == "__main__": unittest.main()
