"""One state interface for CPU MuJoCo and batched NVIDIA MuJoCo Warp."""
import numpy as np
import mujoco
import torch


class NativeBackend:
    def __init__(self,model,count,device,**kwargs):
        if device != "cpu": raise ValueError("Native backend uses CPU; use warp for CUDA physics")
        self.model=model; self.data=[mujoco.MjData(model) for _ in range(count)]
        self.device=device

    def reset(self,ids,qpos):
        for i,q in zip(ids.tolist(),qpos.detach().cpu().numpy()):
            d=self.data[i];mujoco.mj_resetData(self.model,d);d.qpos[:]=q
            d.ctrl[:]=q[7:];mujoco.mj_forward(self.model,d)

    def step(self,targets,substeps):
        for d,ctrl in zip(self.data,targets.detach().cpu().numpy()):
            d.ctrl[:]=ctrl;mujoco.mj_step(self.model,d,nstep=substeps)
        self.check()

    def tensor(self,name):
        return torch.tensor(np.stack([np.asarray(getattr(d,name)) for d in self.data]),dtype=torch.float32)

    def check(self):
        for d in self.data:
            if any(w.number for w in d.warning):
                raise RuntimeError("MuJoCo simulation warning; inspect contacts/constraints and timestep")


class WarpBackend:
    def __init__(self,model,count,device,nconmax=128,njmax=512):
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("Warp training requires an NVIDIA CUDA device; use native/cpu for local checks")
        import warp as wp
        import mujoco_warp as mjw
        self.wp,self.mjw,self.model,self.device=wp,mjw,model,device
        wp.init()
        # Both frameworks share the same CUDA stream to preserve tensor/kernel ordering.
        self.stream=wp.stream_from_torch(torch.cuda.current_stream(device))
        with wp.ScopedStream(self.stream):
            self.m=mjw.put_model(model)
            self.d=mjw.make_data(model,nworld=count,nconmax=nconmax,njmax=njmax)
        self.graphs={}

    def tensor(self,name):
        return self.wp.to_torch(getattr(self.d,name))

    def reset(self,ids,qpos):
        self.tensor("qpos")[ids]=qpos
        for name in ("qvel","qacc","qacc_warmstart","qfrc_applied","xfrc_applied","time"):
            self.tensor(name)[ids]=0
        self.tensor("ctrl")[ids]=qpos[:,7:]
        with self.wp.ScopedStream(self.stream): self.mjw.forward(self.m,self.d)

    def step(self,targets,substeps):
        self.tensor("ctrl").copy_(targets)
        with self.wp.ScopedStream(self.stream):
            if substeps not in self.graphs:
                # Compile outside CUDA capture, then restore the physical state:
                # warming must not add an unrequested integration step.
                saved={name:self.tensor(name).clone() for name in ('qpos','qvel','qacc_warmstart','time')}
                self.mjw.step(self.m,self.d)
                for name,value in saved.items():self.tensor(name).copy_(value)
                self.mjw.forward(self.m,self.d)
                # Capture itself does not execute; launch once for this step.
                with self.wp.ScopedCapture() as capture:
                    for _ in range(substeps): self.mjw.step(self.m,self.d)
                self.graphs[substeps]=capture.graph
            self.wp.capture_launch(self.graphs[substeps])

    def check(self):
        # Warp can exceed allocations rather than throwing a Python exception.
        if int(self.tensor("nacon").max().item())>self.d.naconmax or int(self.tensor("nefc").max().item())>self.d.njmax:
            raise RuntimeError("Warp contact/constraint capacity exceeded; raise nconmax/njmax")
