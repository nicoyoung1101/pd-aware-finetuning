"""Data-free tests of the actual trainer utilities, with a real-sized pose RNN."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Evaluation'))
import torch
from icnr2026_finetune_core import (
    configure_capacity, prepare_imu_input, SubjectBalancedEpochSampler,
    absolute_log_ratio, snapshot_trainable_parameters, l2sp_loss,
    differentiable_joint_velocity,
)

from net import RNN

class ContractTests(unittest.TestCase):
    def test_head_channel_after_root_copy(self):
        generator=torch.Generator().manual_seed(19)
        acc=torch.randn(19,6,3,generator=generator)
        angles=torch.linspace(0,1,19)
        ori=torch.eye(3).repeat(19,6,1,1)
        ori[:,:,0,0]=angles.cos()[:,None];ori[:,:,0,2]=angles.sin()[:,None]
        ori[:,:,2,0]=-angles.sin()[:,None];ori[:,:,2,2]=angles.cos()[:,None]
        original=acc.clone()
        data=prepare_imu_input(acc,ori,'C5NoHead')
        self.assertEqual(tuple(data.shape),(19,72))
        self.assertTrue(torch.allclose(data[:,:18].reshape(-1,6,3)[:,4],torch.zeros(19,3),atol=1e-7))
        self.assertTrue(torch.allclose(data[:,18:].reshape(-1,6,3,3)[:,4],torch.eye(3).expand(19,3,3),atol=1e-6))
        self.assertTrue(torch.equal(acc,original))

    def test_freezing_and_l2sp_restoring_gradient(self):
        net=torch.nn.Module()
        net.pose_s3=RNN(69+72,90,128)
        net.other=torch.nn.Linear(8,8)
        params=configure_capacity(net,'S1')
        self.assertEqual(sum(p.numel() for p in params),23130)
        self.assertTrue(all(p.requires_grad==name.startswith('pose_s3.linear2.') for name,p in net.named_parameters()))
        reference=snapshot_trainable_parameters(net)
        self.assertEqual(float(l2sp_loss(net,reference).detach()),0)
        with torch.no_grad():
            for p in params:p.add_(0.001)
        l2sp_loss(net,reference).backward()
        for p in params:self.assertTrue(torch.allclose(p.grad,torch.full_like(p,0.002),atol=1e-7))
        self.assertTrue(all(p.grad is None for p in net.other.parameters()))

    def test_subject_balancing_keeps_whole_walk_indices(self):
        class FakeDataset:
            indices=list(range(6))
            manifest=[{'subject_id':s} for s in ['A','A','A','A','B','C']]
            def __len__(self):return len(self.indices)
        sampler=SubjectBalancedEpochSampler(FakeDataset(),seed=17)
        self.assertEqual(sampler.subject_counts(3),{'A':2,'B':2,'C':2})
        self.assertEqual(sampler.epoch_positions(3),sampler.epoch_positions(3))
        self.assertTrue(all(i in range(6) for i in sampler.epoch_positions(3)))

    def test_log_ratio_and_translation_invariance(self):
        self.assertAlmostEqual(float(absolute_log_ratio(torch.tensor(2.),torch.tensor(2.))),0)
        self.assertAlmostEqual(float(absolute_log_ratio(torch.tensor(2.),torch.tensor(1.))),float(absolute_log_ratio(torch.tensor(1.),torch.tensor(2.))),places=6)
        joints=torch.randn(10,24,3)
        drift=torch.arange(10,dtype=joints.dtype).view(10,1,1)
        self.assertTrue(torch.allclose(differentiable_joint_velocity(joints),differentiable_joint_velocity(joints+drift),rtol=1e-6))

if __name__=='__main__':unittest.main()
