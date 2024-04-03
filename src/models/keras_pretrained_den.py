import os
from typing import Tuple

import numpy as np
from genesis.generator import st_hardmax_softmax, st_sampled_softmax
from keras.models import load_model
from numpy.random import default_rng

model_names = {
    0.0: 'genesis_splicing_cnn_target_isoform_00_pwm_and_multisample_hek_only_random_regions_50_epochs_harderentropy',
    0.25: 'genesis_splicing_cnn_target_isoform_025_pwm_and_multisample_hek_only_random_regions_50_epochs_harderentropy',
    0.5: 'genesis_splicing_cnn_target_isoform_05_pwm_and_multisample_hek_only_random_regions_50_epochs_harderentropy',
    0.75: 'genesis_splicing_cnn_target_isoform_075_pwm_and_multisample_hek_only_random_regions_50_epochs_harderentropy',
    1.0: 'genesis_splicing_cnn_target_isoform_10_pwm_and_multisample_hek_only_random_regions_70_epochs_harderentropy',
}

sequence_template = 'AGGTGCTTGGNNNNNNNNNNNNNNNNNNNNNNNNNGGTCGACCCAGGTTCGTGNNNNNNNNNNNNNNNNNNNNNNNNNGAGGTATTCTTATCACCTTCGTGGCTACAGA'

target_isos = [0.00, 0.25, 0.5, 0.75, 1.0]


class KerasPretrainedDENPredictor(object):

    def __init__(self, seed: int, batch_size: int, path: str, target_iso: float) -> None:
        self.path = path
        self.batch_size = batch_size
        self.target_iso = target_iso
        self.rng = default_rng(seed)

        assert target_iso in target_isos

        model_name = model_names[target_iso] + '_predictor.h5'
        model_path = os.path.join(self.path, model_name)

        # load model
        self.predictor = load_model(model_path,
                                    custom_objects={
                                        'st_sampled_softmax': st_sampled_softmax,
                                        'st_hardmax_softmax': st_hardmax_softmax
                                    })

    def generate_sequences_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        sequence_class = np.array([0] * self.batch_size).reshape(-1, 1)

        noise_1 = self.rng.uniform(-1, 1, (self.batch_size, 100))
        noise_2 = self.rng.uniform(-1, 1, (self.batch_size, 100))

        pred_outputs = self.predictor.predict([sequence_class, noise_1, noise_2], batch_size=self.batch_size)

        _, _, _, optimized_pwm, _, sampled_pwm_1, _, _, _, hek_pred, _, _, _ = pred_outputs

        return sampled_pwm_1, hek_pred
