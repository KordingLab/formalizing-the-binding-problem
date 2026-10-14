"""CPU regression test: python -m unittest discover -s tests -v."""

import ast
import logging
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch


def load_train_method():
    # Exercise the production epoch loop without importing backbone dependencies
    # (timm/transformers) or constructing datasets and activation caches.
    path = Path(__file__).resolve().parents[1] / "src" / "trainer.py"
    tree = ast.parse(path.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "Trainer")
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                  and node.name == "train")
    namespace = {"os": os, "math": math, "logger": logging.getLogger(__name__),
                 "tqdm": lambda iterable: iterable}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["train"]


class RecordingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.1)
        self.gradients = []

    def step(self, closure=None):
        self.gradients.append([p.grad.detach().clone()
                               for group in self.param_groups for p in group["params"]])
        return super().step(closure)


class GradientAccumulationTest(unittest.TestCase):
    def test_group_gradients_and_updates(self):
        train = load_train_method()
        # Odd counts, a multi-minibatch remainder, complete groups, and no accumulation.
        for num_batches, accum_steps in [(3, 2), (77, 2), (5, 3), (1, 4), (4, 2), (3, 1)]:
            for enabled in [(True, False), (False, True), (True, True)]:
                with self.subTest(num_batches=num_batches, accum_steps=accum_steps,
                                  enabled=enabled), TemporaryDirectory() as directory:
                    probes = [torch.nn.Linear(1, 1, bias=False) for _ in range(2)]
                    for probe in probes:
                        torch.nn.init.constant_(probe.weight, 0.5)
                    params = [probe.weight for probe, active in zip(probes, enabled) if active]
                    optimizer = RecordingSGD(params)
                    batches = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                        torch.ones(2 * num_batches - 1, 1),
                        torch.zeros(2 * num_batches - 1, 1),
                        torch.zeros(2 * num_batches - 1, 1),
                        torch.zeros(2 * num_batches - 1, 1)), batch_size=2)
                    trainer = SimpleNamespace(
                        cfg=SimpleNamespace(trainer=SimpleNamespace(
                            grad_accum_steps=accum_steps, max_epoch=1)),
                        ckpt_path=os.path.join(directory, "checkpoint.pth"), start_epoch=0,
                        train_probe_Y=enabled[0], train_probe_F=enabled[1],
                        model=torch.nn.Identity(), probe_Y=probes[0], probe_F=probes[1],
                        train_dataloader=batches, use_cached_activations=True,
                        optimizer=optimizer, scheduler=SimpleNamespace(step=lambda: None),
                        y_dim=1, f_dim=1, best_val={"Y": 0, "F": 0},
                        eval=lambda split: {f"{split}_lossY": 1, f"{split}_lossF": 1},
                        save_model=lambda *args, **kwargs: None,
                        _maybe_save_probe_weights_if_missing=lambda: None,
                        _maybe_plot_training_history=lambda history: None,
                        _add_binding_metrics=lambda *args: None,
                    )
                    batch_index = iter(range(num_batches))

                    def train_loop(*args, **kwargs):
                        coefficient = next(batch_index) + 1
                        return {"lossY": probes[0].weight.sum() * coefficient,
                                "lossF": probes[1].weight.sum() * (2 * coefficient),
                                "accY": 0.0, "accF": 0.0}

                    trainer.train_loop = train_loop
                    train(trainer)
                    expected = [sum(range(start + 1, min(start + accum_steps, num_batches) + 1))
                                / min(accum_steps, num_batches - start)
                                for start in range(0, num_batches, accum_steps)]
                    self.assertEqual(len(optimizer.gradients), len(expected))
                    multipliers = [i + 1 for i, active in enumerate(enabled) if active]
                    for gradients, mean in zip(optimizer.gradients, expected):
                        for gradient, multiplier in zip(gradients, multipliers):
                            torch.testing.assert_close(gradient, torch.full_like(gradient, mean * multiplier))
                    for parameter, multiplier in zip(params, multipliers):
                        torch.testing.assert_close(parameter, torch.full_like(
                            parameter, 0.5 - 0.1 * sum(expected) * multiplier))
                        self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
