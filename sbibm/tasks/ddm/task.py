from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pyro
import torch
from pyro import distributions as pdist
from sbibm.tasks.ddm.utils import DDMJulia

import sbibm  # noqa -- needed for setting sysimage path
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.utils.decorators import lazy_property


class DDM(Task):
    def __init__(
        self,
        dt: float = 0.001,
        num_trials: int = 1,
        dim_parameters: int = 3,
    ):
        """Drift-diffusion model.

        Args:
            dt: integration step size in s.
            num_trials: number of trials to run for each parameter.
            dim_parameters: if 2, use only drift v and boundary separation a
                if 4 use v, a, w and tau as in the LAN paper.
        """
        self.dt = dt
        self.num_trials = num_trials
        assert dim_parameters in [2, 3], "dim_parameters must be 2 or 3."

        super().__init__(
            dim_parameters=dim_parameters,
            dim_data=num_trials,
            name=Path(__file__).parent.name,
            name_display="DDM",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=torch.tensor([1, 2, 4, 5, 6]).repeat_interleave(4),
        )
        self.num_trials_per_observation = torch.tensor([1, 10, 100, 1000]).repeat(5)

        # Prior
        self.prior_params = {
            "low": torch.tensor([-2.0, 0.5, 0.3][:dim_parameters]),
            "high": torch.tensor([2.0, 2.0, 0.7][:dim_parameters]),
        }
        self.prior_labels = ["v", "a", "w"][:dim_parameters]

        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)

    @lazy_property
    def ddm(self):
        return DDMJulia(
            dt=self.dt, num_trials=self.num_trials, dim_parameters=self.dim_parameters
        )

    def get_labels_parameters(self) -> List[str]:
        """Get list containing parameter labels"""
        return self.prior_labels

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(
        self,
        max_calls: Optional[int] = None,
        seed: int = -1,
        num_trials: int = -1,
    ) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """

        # Two-parameter case.
        if self.dim_parameters == 2:

            def simulator(parameters):
                v, a = parameters.numpy().T
                rts, choices = self.ddm.simulate(
                    v,
                    a,
                    seed=seed,
                    num_trials=num_trials if num_trials > 0 else self.num_trials,
                )
                # encode zero choices as negative RTs.
                rts[choices == 0] *= -1
                return torch.tensor(rts, dtype=torch.float32)

        elif self.dim_parameters == 3:

            def simulator(parameters):
                v, a, w = parameters.numpy().T
                # using boundary separation a and offset w
                # pass negative lower bound as required by DiffModels.
                bl = -w * a
                bu = (1 - w) * a

                rts, choices = self.ddm.simulate_simpleDDM(
                    v,
                    bl,
                    bu,
                    seed=seed,
                    num_trials=num_trials if num_trials > 0 else self.num_trials,
                )
                # encode zero choices as negative RTs.
                rts[choices == 0] *= -1
                return torch.tensor(rts, dtype=torch.float32)

        else:
            raise NotImplementedError()

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def get_log_likelihood(self, parameters, data):
        """Return likelihood given parameters and data.

        Takes product of likelihoods across iid trials.

        Batch dimension is only across parameters, the data is fixed.
        """
        num_trials = data.shape[1]
        parameters = parameters.numpy()
        # Decode choices from RT sign.
        choices = torch.where(
            data[0, :] > 0, torch.ones(num_trials), torch.zeros(num_trials)
        ).numpy()
        # Convert negative RTs.
        data = data.numpy()
        rts = abs(data[0, :])

        if self.dim_parameters == 2:
            log_likelihoods = self.ddm.log_likelihood(
                parameters[:, 0],  # v
                parameters[:, 1],  # boundary separation a
                # Pass rts and choices separately.
                rts,
                choices,
            )
        elif self.dim_parameters == 3:
            # using boundary separation a and offset w
            # pass negative lower bound as required by DiffModels.
            v, a, w = parameters.T
            bl = -w * a
            bu = (1 - w) * a
            log_likelihoods = self.ddm.log_likelihood_simpleDDM(
                v,
                bl,
                bu,
                # Pass rts and choices separately.
                rts,
                choices,
            )
        else:
            raise NotImplementedError()

        return torch.tensor(log_likelihoods)

    def get_potential_fn(
        self,
        num_observation: int,
        observation: torch.Tensor,
        automatic_transforms_enabled: bool,
    ) -> Callable:
        """Return potential function for fixed data.

        Potential: $-[\log r(x_o, \theta) + \log p(\theta)]$

        The data can consists of multiple iid trials.
        Then the overall likelihood is defined as the product over iid likelihood.
        """
        log_prob_fun = self._get_log_prob_fn(
            num_observation,
            observation,
            posterior=True,
            implementation="experimental",
            **dict(automatic_transforms_enabled=automatic_transforms_enabled),
        )

        def potential_fn(parameters: Dict) -> torch.Tensor:
            return -log_prob_fun(parameters["parameters"])

        return potential_fn

    def _get_log_prob_fn(
        self,
        num_observation: Optional[int],
        observation: Optional[torch.Tensor],
        implementation: str,
        posterior: bool,
        **kwargs: Any,
    ) -> Callable:

        transforms = self._get_transforms(
            num_observation=num_observation,
            observation=observation,
            automatic_transforms_enabled=kwargs["automatic_transforms_enabled"],
        )["parameters"]
        if observation is None:
            observation = self.get_observation(num_observation)

        def log_prob_fn(parameters: torch.Tensor) -> torch.Tensor:

            # We need to calculate likelihoods in constrained space.
            parameters_constrained = transforms.inv(parameters)

            # Get likelihoods from DiffModels.jl in constrained space.
            log_likelihood_constrained = self.get_log_likelihood(
                parameters_constrained, observation
            )
            # But we need log probs in unconstrained space. Get log abs det jac
            log_abs_det = transforms.log_abs_det_jacobian(
                parameters_constrained, parameters
            )
            # Without transforms, logabsdet returns second dimension.
            if log_abs_det.ndim > 1:
                log_abs_det = log_abs_det.sum(-1)

            # Likelihood in unconstrained space is:
            # prob_constrained * 1/abs_det_jacobian
            # log_prob_constrained - log_abs_det
            log_likelihood = log_likelihood_constrained - log_abs_det
            if posterior:
                return log_likelihood + self.get_prior_dist().log_prob(
                    parameters_constrained
                )
            else:
                return log_likelihood

        return log_prob_fn

    def _sample_reference_posterior(
        self,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample reference posterior for given observation

        Args:
            num_observation: Observation number
            num_samples: Number of samples to generate
            observation: Observed data, if None, will be loaded using `num_observation`
            kwargs: Passed to run_mcmc

        Returns:
            Samples from reference posterior
        """
        from sbibm.algorithms.pyro.mcmc import run as run_mcmc
        from sbibm.algorithms.pytorch.baseline_grid import run as run_grid
        from sbibm.algorithms.pytorch.baseline_rejection import run as run_rejection
        from sbibm.algorithms.pytorch.utils.proposal import get_proposal

        if num_observation is not None:
            initial_params = self.get_true_parameters(num_observation=num_observation)
        else:
            initial_params = None

        # if num_observation in [1, 5, 9, 13, 17]:
        #     samples = run_grid(
        #         task=self,
        #         num_samples=num_samples,
        #         num_observation=num_observation,
        #         observation=observation,
        #         resolution=20000,
        #         batch_size=10000,
        #         **dict(automatic_transforms_enabled=False),
        #     )

        # idxs = torch.tensor([[1, 2, 4, 5, 6]])
        # if num_observation in idxs:
        #     samples = run_rejection(
        #         task=self,
        #         num_observation=num_observation,
        #         observation=observation,
        #         num_samples=num_samples,
        #         batch_size=10000,
        #         num_batches_without_new_max=1000,
        #         multiplier_M=1.2,
        #         proposal_dist=self.prior_dist,
        #         **dict(automatic_transforms_enabled=False),
        #     )
        # if num_observation in idxs:
        num_chains = 5
        num_warmup = 10_000
        automatic_transforms_enabled = True

        samples = run_mcmc(
            task=self,
            potential_fn=self.get_potential_fn(
                num_observation,
                observation,
                automatic_transforms_enabled=automatic_transforms_enabled,
            ),
            kernel="Slice",
            jit_compile=False,
            num_warmup=num_warmup,
            num_chains=num_chains,
            num_observation=num_observation,
            observation=observation,
            num_samples=num_samples,
            initial_params=initial_params.repeat(num_chains, 1),
            automatic_transforms_enabled=automatic_transforms_enabled,
        )
        # else:
        #     num_chains = 5
        #     num_warmup = 10_000
        #     automatic_transforms_enabled = True

        #     proposal_samples = run_mcmc(
        #         task=self,
        #         potential_fn=self.get_potential_fn(
        #             num_observation,
        #             observation,
        #             automatic_transforms_enabled=automatic_transforms_enabled,
        #         ),
        #         kernel="Slice",
        #         jit_compile=False,
        #         num_warmup=num_warmup,
        #         num_chains=num_chains,
        #         num_observation=num_observation,
        #         observation=observation,
        #         num_samples=num_samples,
        #         initial_params=initial_params.repeat(num_chains, 1),
        #         automatic_transforms_enabled=automatic_transforms_enabled,
        #     )

        #     proposal_dist = get_proposal(
        #         task=self,
        #         samples=proposal_samples,
        #         prior_weight=0.1,
        #         bounded=True,
        #         density_estimator="flow",
        #         flow_model="nsf",
        #     )

        #     samples = run_rejection(
        #         task=self,
        #         num_observation=num_observation,
        #         observation=observation,
        #         num_samples=num_samples,
        #         batch_size=10000,
        #         num_batches_without_new_max=1000,
        #         multiplier_M=1.2,
        #         proposal_dist=proposal_dist,
        #         **dict(automatic_transforms_enabled=False),
        #     )

        return samples

    def _setup(self, n_jobs: int = -1, create_reference: bool = True, **kwargs: Any):
        """Setup the task: generate observations and reference posterior samples

        In most cases, you don't need to execute this method, since its results are stored to disk.

        Re-executing will overwrite existing files.

        Args:
            n_jobs: Number of to use for Joblib
            create_reference: If False, skips reference creation
        """
        from joblib import Parallel, delayed

        def run(num_observation, observation_seed, **kwargs):
            np.random.seed(observation_seed)
            torch.manual_seed(observation_seed)
            self._save_observation_seed(num_observation, observation_seed)

            prior = self.get_prior()
            true_parameters = prior(num_samples=1)
            self._save_true_parameters(num_observation, true_parameters)

            num_trials = int(self.num_trials_per_observation[num_observation - 1])
            self.dim_data = 2 * num_trials
            self.num_trials = num_trials
            simulator = self.get_simulator(
                seed=int(observation_seed),
                num_trials=num_trials,
            )
            observation = simulator(true_parameters)
            self._save_observation(num_observation, observation)

            if create_reference:
                reference_posterior_samples = self._sample_reference_posterior(
                    num_observation=num_observation,
                    num_samples=self.num_reference_posterior_samples,
                    **kwargs,
                )
                num_unique = torch.unique(reference_posterior_samples, dim=0).shape[0]
                assert num_unique == self.num_reference_posterior_samples
                self._save_reference_posterior_samples(
                    num_observation,
                    reference_posterior_samples,
                )

        Parallel(n_jobs=n_jobs, verbose=50, backend="loky")(
            delayed(run)(num_observation, observation_seed, **kwargs)
            for num_observation, observation_seed in enumerate(
                self.observation_seeds, start=1
            )
        )


if __name__ == "__main__":
    task = DDM(num_trials=1, dim_parameters=3)
    task._setup(n_jobs=-1)
