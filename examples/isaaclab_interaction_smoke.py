"""Contact physics acceptance fixture: free object + two jaws, and a force probe.

Geometry and joints here are test-fixture bindings, never ISA operands. The
separate UR5+Wuji smoke checks the production asset and IK integration.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app


def main():
    import numpy as np
    import torch
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from manipisa import (Runtime, Instruction, Opcode, Mode, PoseGoal, ShapeGoal,
        ContactGoal, GraspGoal, WrenchGoal, BoundedMotion, ContactRequirement, Status)
    from manipisa.types import TERMINAL
    from manipisa.adapters.isaaclab import ActorBinding, IsaacLabAdapter
    from manipisa.adapters.physx_contacts import PhysXContactSource, ContactBinding, axis_aligned_box_separation

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device=args.device, gravity=(0.,0.,0.)))
    stage = sim.stage
    def body(path, position, size, mass=.1):
        prim = UsdGeom.Xform.Define(stage, path)
        prim.AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
        UsdPhysics.MassAPI.Apply(prim.GetPrim()).CreateMassAttr(mass)
        PhysxSchema.PhysxContactReportAPI.Apply(prim.GetPrim()).CreateThresholdAttr(0.)
        cube = UsdGeom.Cube.Define(stage, path+"/collision")
        cube.CreateSizeAttr(1.)
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        collision.CreateContactOffsetAttr(.001)
        collision.CreateRestOffsetAttr(0.)
        return prim

    def articulation(path, origin, links):
        root = UsdGeom.Xform.Define(stage, path)
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        body(path+"/base", origin, (.005,.005,.005), 1.)
        # Base collider disabled; it is only the fixed reference body.
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path+"/base/collision")).GetCollisionEnabledAttr().Set(False)
        fixed = UsdPhysics.FixedJoint.Define(stage, path+"/fixed")
        fixed.CreateBody1Rel().SetTargets([path+"/base"])
        fixed.CreateLocalPos0Attr(Gf.Vec3f(*origin))
        for name, offset, size in links:
            body(path+"/"+name, tuple(np.array(origin)+offset), size)
            j = UsdPhysics.PrismaticJoint.Define(stage, path+"/"+name+"_joint")
            j.CreateBody0Rel().SetTargets([path+"/base"])
            j.CreateBody1Rel().SetTargets([path+"/"+name])
            j.CreateLocalPos0Attr(Gf.Vec3f(*offset))
            j.CreateAxisAttr("X")
            j.CreateLowerLimitAttr(-.05)
            j.CreateUpperLimitAttr(.05)
            drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(2000.)
            drive.CreateDampingAttr(50.)
            drive.CreateMaxForceAttr(50.)
        return Articulation(ArticulationCfg(prim_path=path, spawn=None,
            actuators={"drive": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=2000., damping=50., effort_limit_sim=50., velocity_limit_sim=.2)}))

    hand = articulation("/World/Grip", (0.,0.,.6), [("left",(-.035,0.,0.),(.01,.06,.06)), ("right",(.035,0.,0.),(.01,.06,.06))])
    probe = articulation("/World/Probe", (1.,0.,.6), [("tip",(0.,0.,0.),(.04,.04,.04))])
    material = sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.)
    material.func("/World/ContactMaterial", material)
    from isaaclab.sim import bind_physics_material
    for path in ("/World/Grip/left/collision", "/World/Grip/right/collision", "/World/Probe/tip/collision"):
        bind_physics_material(path, "/World/ContactMaterial")
    obj = RigidObject(RigidObjectCfg(prim_path="/World/Object", spawn=sim_utils.CuboidCfg(size=(.04,.04,.04),
        mass_props=sim_utils.MassPropertiesCfg(mass=.1), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.), physics_material=material),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,.6))))
    target = RigidObject(RigidObjectCfg(prim_path="/World/Target", spawn=sim_utils.CuboidCfg(size=(.04,.08,.08),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.), physics_material=material),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.06,0.,.6))))
    sim.reset()
    for robot in (hand,probe):
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.reset()
        robot.update(.005)
    obj.update(.005)
    target.update(.005)
    def robot_bounds(robot, name, half):
        def bounds():
            pos = robot.data.body_link_pos_w[0,robot.body_names.index(name)].cpu().numpy()
            return pos-np.array(half), pos+np.array(half)
        return bounds
    def object_bounds():
        from manipisa.interaction import rotation
        state = obj.data.root_state_w[0].cpu().numpy()
        extent = np.abs(rotation(state[3:7])) @ np.array((.02,.02,.02))
        return state[:3]-extent,state[:3]+extent
    source = PhysXContactSource(sim, {
        "left_contact": ContactBinding("hand", "object", "/World/Grip/left", "/World/Object",
            axis_aligned_box_separation(robot_bounds(hand,"left",(.005,.03,.03)), object_bounds), friction_lower_bound=.5),
        "right_contact": ContactBinding("hand", "object", "/World/Grip/right", "/World/Object",
            axis_aligned_box_separation(robot_bounds(hand,"right",(.005,.03,.03)), object_bounds), friction_lower_bound=.5),
        "press": ContactBinding("arm", "surface", "/World/Probe/tip", "/World/Target",
            axis_aligned_box_separation(robot_bounds(probe,"tip",(.02,.02,.02)), lambda: (np.array((1.04,-.04,.56)),np.array((1.08,.04,.64)))))})
    disturbance = [0.]
    def load():
        obj.permanent_wrench_composer.set_forces_and_torques(torch.tensor([[[0.,0.,-disturbance[0]]]], device=sim.device),
                                          torch.zeros((1,1,3), device=sim.device), is_global=True)
        obj.write_data_to_sim()
    adapter = IsaacLabAdapter(sim, {
        "hand": ActorBinding(hand,("left_joint","right_joint"),"hand","base","grip_frame"),
        "arm": ActorBinding(probe,("tip_joint",),"arm","tip","probe_frame"),
    }, entities={"object":obj,"surface":target}, contact_source=source, pre_step_hooks=(load,))
    for _ in range(40):
        adapter.step()
    runtime, trace, checks, results = Runtime(adapter), [], {}, {}
    def run(label, request, previous=None):
        submitted = runtime.submit(request, replace_handle=previous.control_handle if previous else None)
        results[label] = submitted.to_dict()
        for _ in range(int(request.timeout_s/adapter.dt)+20):
            if runtime.query(submitted.call_id).status in TERMINAL:
                break
            runtime.step()
            trace.append(runtime.feedback())
        result = runtime.query(submitted.call_id)
        results[label] = result.to_dict()
        checks[label] = result.status == Status.SUCCEEDED
        print("INTERACTION " + label + " " + json.dumps(result.to_dict()), flush=True)
        return result
    common = dict(execution_domain="bounded_contact", timeout_s=6., max_joint_speed=.15, dwell_s=.05, hold_for_s=30.)
    req = (ContactRequirement("left_contact",.10,20.,.003), ContactRequirement("right_contact",.10,20.,.003))
    motion = (BoundedMotion("hand",ShapeGoal((.013,-.013)),max_configuration_delta=.03),)
    contact = run("make_contact", Instruction(Opcode.MAKE_CONTACT,("hand",),ContactGoal(req,motion),**common))
    grasp = run("grasp_reach", Instruction(Opcode.CONTROL_GRASP,("hand",),GraspGoal(req,motion,"object","grip_frame",
        ((0.,0.,-.2,0.,0.,0.),),.5,max_relative_translation=.006),**common), contact)
    disturbance[0] = .2
    sustain = run("grasp_sustain_under_load", Instruction(Opcode.CONTROL_GRASP,("hand",),GraspGoal(req,motion,"object","grip_frame",
        ((0.,0.,-.2,0.,0.,0.),),.5,max_relative_translation=.006,require_valid_at_entry=True),mode=Mode.SUSTAIN,sustain_s=.3,**common), grasp)
    checks["grasp_relation_returned"] = sustain.relation is not None
    pose = adapter.observe().frames["probe_frame"]
    press_req = (ContactRequirement("press",.1,20.,.003),)
    pressmotion = (BoundedMotion("arm",PoseGoal("probe_frame",(pose.position[0]+.024,*pose.position[1:]),pose.orientation_wxyz),max_translation=.035),)
    press = run("probe_contact",Instruction(Opcode.MAKE_CONTACT,("arm",),ContactGoal(press_req,pressmotion),**common))
    pose = adapter.observe().frames["probe_frame"]
    wrench = run("apply_wrench",Instruction(Opcode.APPLY_WRENCH,("arm",),WrenchGoal(press_req,"probe_frame",
        (2.,0.,0.,0.,0.,0.),"world",pose.position,controlled_axes=(True,False,False,False,False,False),
        tolerances=(.25,.25,.25,.1,.1,.1),admittance=(.003,.003,.003,.01,.01,.01)),mode=Mode.SUSTAIN,sustain_s=.2,**common),press)
    from dataclasses import replace
    from manipisa.interaction import wrench_at
    observed = adapter.observe()
    force_request = Instruction(Opcode.APPLY_WRENCH,("arm",),WrenchGoal(press_req,"probe_frame",(2.,0.,0.,0.,0.,0.),"world",pose.position),mode=Mode.SUSTAIN)
    measured = wrench_at(force_request.goal,force_request,observed)
    shifted = wrench_at(replace(force_request.goal,reference_point=(pose.position[0],pose.position[1]+.1,pose.position[2])),force_request,observed)
    checks["physical_wrench_reference_point"] = bool(measured is not None and shifted is not None and abs(
        shifted[0][5]-measured[0][5]-.1*measured[0][0]) < 1e-5)
    # Explicitly retire the old grasp before a topology-changing instruction.
    disturbance[0] = 0.
    release = run("break_contact",Instruction(Opcode.BREAK_CONTACT,("hand",),ContactGoal(req,
        (BoundedMotion("hand",ShapeGoal((0.,0.)),max_configuration_delta=.03),)),**common),sustain)
    checks["physical_gap_proven"] = all(c.separation_m is not None and c.separation_m >= .003 and not c.present
                                      for k,c in adapter.observe().contacts.items() if k != "press")
    # A second real grasp is deliberately overloaded AFTER SUSTAIN activation.
    again = run("reestablish_contact", Instruction(Opcode.MAKE_CONTACT,("hand",),ContactGoal(req,motion),**common),release)
    overload_command = Instruction(Opcode.CONTROL_GRASP,("hand",),GraspGoal(req,motion,"object","grip_frame",
        ((0.,0.,-.2,0.,0.,0.),),.5,max_relative_translation=.006),mode=Mode.SUSTAIN,**common)
    overload = runtime.submit(overload_command,replace_handle=again.control_handle)
    for _ in range(200):
        if runtime.query(overload.call_id).status != Status.RUNNING:
            break
        runtime.step(); trace.append(runtime.feedback())
    checks["overload_test_started_active"] = runtime.query(overload.call_id).status == Status.ACTIVE
    disturbance[0] = 20.
    for _ in range(60):
        if runtime.query(overload.call_id).status in TERMINAL:
            break
        runtime.step(); trace.append(runtime.feedback())
    failed = runtime.query(overload.call_id)
    results["overload_expected_failure"] = failed.to_dict()
    checks["overload_fails_grasp"] = failed.status == Status.FAILED and failed.reason in ("GRASP_INVALID","CONTACT_LOST")
    disturbance[0] = 0.
    for _ in range(4): runtime.step(); trace.append(runtime.feedback())
    checks["failure_remains_terminal"] = runtime.query(overload.call_id).status == Status.FAILED
    report = {"passed":all(checks.values()),"checks":checks,"results":results,"feedback":runtime.feedback(),
              "scope":"Isaac Lab physical acceptance fixture, actual contact/friction and free object; not a Wuji grasp or Bench2Dex score", "physics_dt":adapter.dt,
              "measured_wrench_before_release": measured[0].tolist() if measured else None}
    source_files = sorted((ROOT/"manipisa").rglob("*.py"))+[Path(__file__).resolve()]
    report["source_sha256"] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    (out/"trace.jsonl").write_text("\n".join(json.dumps(x,allow_nan=False) for x in trace),encoding="utf-8")
    print("MANIPISA_INTERACTION " + json.dumps({"passed":report["passed"],"checks":checks}),flush=True)
    if not report["passed"]:
        raise AssertionError("Interaction checks failed; inspect report")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        if os.name == "nt":
            print(f"MANIPISA_PROCESS_EXIT {code}",flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
        app.close(wait_for_replicator=False)
    sys.exit(code)
