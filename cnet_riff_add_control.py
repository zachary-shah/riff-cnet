import os, argparse

import torch
from share import *
from cldm.model import create_model
from huggingface_hub import hf_hub_download

# add control to model (Source: from ControlNet)
def tool_add_control(input_path, output_path, cntrl_mdl_config_path):

    assert os.path.exists(input_path), 'Input model does not exist.'
    assert not os.path.exists(output_path), 'Output filename already exists.'
    assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

    def get_node_name(name, parent_name):
        if len(name) <= len(parent_name):
            return False, ''
        p = name[:len(parent_name)]
        if p != parent_name:
            return False, ''
        return True, name[len(parent_name):]

    model = create_model(config_path=cntrl_mdl_config_path)

    pretrained_weights = torch.load(input_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    scratch_dict = model.state_dict()

    target_dict = {}
    for k in scratch_dict.keys():
        is_control, name = get_node_name(k, 'control_')
        if is_control:
            copy_k = 'model.diffusion_' + name
        else:
            copy_k = k
        if copy_k in pretrained_weights:
            target_dict[k] = pretrained_weights[copy_k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are newly added: {k}')

    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), output_path)
    print('Done.')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cntrl_riff_path",
        type=str,
        nargs="?",
        default="./models/control_riffusion_ini.ckpt",
        help="path to save riff + control net to"
    )
    parser.add_argument(
        "--cntrl_mdl_config_path",
        type=str,
        nargs="?",
        default='./models/cldm_v15.yaml',
        help="path to yaml config file for loading controlnet structure."
    ) 

    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)

    riffusion_path = hf_hub_download(repo_id="riffusion/riffusion-model-v1", filename="riffusion-model-v1.ckpt")
    print(F"Riffusion .ckpt saved to {riffusion_path}")
    # add control to riffusion and save controlled model to cntrl_riff_path
    tool_add_control(riffusion_path, args.cntrl_riff_path, args.cntrl_mdl_config_path)
    print(f"Control via {args.cntrl_mdl_config_path} added to riffusion! Model saved to {args.cntrl_riff_path}")

if __name__ ==  '__main__':
    main()