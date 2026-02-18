import os
import sys
import torch
import torch.nn as nn
import numpy as np

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../../'))
sys.path.append(weavercoredir)
from weaver.nn.model.ParticleTransformer import ParticleTransformerTagger


class ParticleTransformerTaggerWrapper(torch.nn.Module):
    """Wrapper for ParticleTransformerTagger to handle input format conversion.
    
    Maps from data config input format to model input format:
    - pf_points -> pf_v (4-vectors are in pf_vectors)
    - pf_features -> pf_x
    - pf_mask -> pf_mask
    - sv_points -> sv_v
    - sv_features -> sv_x
    - sv_mask -> sv_mask
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = ParticleTransformerTagger(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'model.part.cls_token', }

    def forward(self, pf_points, pf_features, pf_vectors, pf_mask, sv_points, sv_features, sv_mask, sv_vectors):
        # pf_features -> pf_x (particle features)
        # pf_vectors -> pf_v (4-vectors: px, py, pz, e)
        # sv_features -> sv_x (SV features)
        # sv_vectors -> sv_v (SV 4-vectors: sv_px, sv_py, sv_pz, sv_e)
        return self.model(
            pf_x=pf_features,
            pf_v=pf_vectors,
            pf_mask=pf_mask,
            sv_x=sv_features,
            sv_v=sv_vectors,  # Now using proper 4-momentum
            sv_mask=sv_mask
        )


def get_model(data_config, **kwargs):
    
    # settings defined in data config file
    pf_features_dims = len(data_config.input_dicts['pf_features'])
    sv_features_dims = len(data_config.input_dicts['sv_features'])
    num_classes = len(data_config.label_value)
    print(f'Found following particle input feature dims: {pf_features_dims}')
    print(f'Found following secondary vertex input feature dims: {sv_features_dims}')
    print(f'Found following number of classes: {num_classes}')

    # get model
    model = ParticleTransformerTaggerWrapper(
        pf_input_dim=pf_features_dims,
        sv_input_dim=sv_features_dims,
        num_classes=num_classes,
        **kwargs
    )

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
    }

    print('Built following model:')
    print(model)
    print('Built following model info:')
    print(model_info)

    return model, model_info
