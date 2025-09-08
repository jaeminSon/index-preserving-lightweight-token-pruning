import sys
import os
import shutil
import json
import importlib.util
import argparse
import torch
from transformers import (
    AutoConfig,
)
from transformers import Qwen2_5_VLForConditionalGeneration


def load_class(file_path, class_name):
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot find spec for {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Class {class_name} not found in {file_path}")
    return cls


def copy_submodule_weights(src_model, target_model, submodule_name):
    src_submodule = getattr(src_model, submodule_name)
    target_submodule = getattr(target_model, submodule_name)
    target_submodule.load_state_dict(src_submodule.state_dict())


def copy_files(src_dir, dest_dir):
    if not os.path.exists(src_dir):
        print(f"Source directory {src_dir} does not exist.")
        return

    files = os.listdir(src_dir)

    for file in files:
        if file.endswith(".safetensors") or file == "model.safetensors.index.json":
            src_file = os.path.join(src_dir, file)
            dest_file = os.path.join(dest_dir, file)
            shutil.copy2(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")


def get_classname_from_hf_directory(hf_directory):
    path_config = os.path.join(hf_directory, "config.json")
    with open(path_config, "r") as file:
        data = json.load(file)

    architecture_values = data.get("architectures")
    assert len(architecture_values) == 1, "Support only single-elemented architecture."

    return architecture_values[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="path to model from which weights are copied.",
    )
    parser.add_argument(
        "--path_model_python",
        type=str,
        help="path to modeling python file for target model.",
    )
    parser.add_argument(
        "--classifier_weights", type=str, help="path to classifier weights."
    )
    args = parser.parse_args()

    source = args.source_dir
    path_target_model = args.path_model_python
    classifier_weights = args.classifier_weights

    classifier = torch.load(classifier_weights)

    target_dir = os.path.dirname(path_target_model)
    class_name = get_classname_from_hf_directory(target_dir)
    target_model_cls = load_class(path_target_model, class_name)
    config = AutoConfig.from_pretrained(target_dir, trust_remote_code=True)
    target_model = target_model_cls(config).to(torch.bfloat16)

    src_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(source)

    # Qwen2.5-VL pruned model has 4 components - patch_classifier, visual, model, lm_head
    target_model.patch_classifier.load_state_dict(classifier["state_dict"])
    target_model.visual.load_state_dict(src_model.visual.state_dict())
    target_model.model.load_state_dict(src_model.model.state_dict())
    target_model.lm_head.load_state_dict(src_model.lm_head.state_dict())
    print("state_dict loaded copied.")

    tmp_dir = "tmp_weight_dir"
    target_model.save_pretrained(tmp_dir, safe_serialization=True)
    copy_files(tmp_dir, target_dir)
    shutil.rmtree(tmp_dir)
