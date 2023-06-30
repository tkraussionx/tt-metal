import torch
import pickle
import numpy as np
import pandas as pd

tensor_files = [
    ("Pt_input.pkl", "tt_input.pkl"),
    ("Encoder_input.pkl", "tt_encoder_input.pkl"),
    ("swin_stage_output_0.pkl", "tt_swin_stage_output_0.pkl"),
    ("swin_stage_output_1.pkl", "tt_swin_stage_output_1.pkl"),
    ("swin_stage_output_2.pkl", "tt_swin_stage_output_2.pkl"),
    ("swin_stage_output_3.pkl", "tt_swin_stage_output_3.pkl"),
    ("layer_0_swin_layer_output_0.pkl", "layer_0_tt_swin_layer_output_0.pkl"),
    ("layer_0_swin_layer_output_1.pkl", "layer_0_tt_swin_layer_output_1.pkl"),
    ("layer_1_swin_layer_output_0.pkl", "layer_1_tt_swin_layer_output_0.pkl"),
    ("layer_1_swin_layer_output_1.pkl", "layer_1_tt_swin_layer_output_1.pkl"),
    ("layer_2_swin_layer_output_0.pkl", "layer_2_tt_swin_layer_output_0.pkl"),
    ("layer_2_swin_layer_output_1.pkl", "layer_2_tt_swin_layer_output_1.pkl"),
    ("layer_2_swin_layer_output_2.pkl", "layer_2_tt_swin_layer_output_2.pkl"),
    ("layer_2_swin_layer_output_3.pkl", "layer_2_tt_swin_layer_output_3.pkl"),
    ("layer_2_swin_layer_output_4.pkl", "layer_2_tt_swin_layer_output_4.pkl"),
    ("layer_2_swin_layer_output_5.pkl", "layer_2_tt_swin_layer_output_5.pkl"),
    ("layer_3_swin_layer_output_0.pkl", "layer_3_tt_swin_layer_output_0.pkl"),
    ("layer_3_swin_layer_output_1.pkl", "layer_3_tt_swin_layer_output_1.pkl"),
    ("layer_0_swin_attention_output_0.pkl", "layer_0_tt_swin_attention_output_0.pkl"),
    ("layer_0_swin_attention_output_1.pkl", "layer_0_tt_swin_attention_output_1.pkl"),
    ("layer_1_swin_attention_output_0.pkl", "layer_1_tt_swin_attention_output_0.pkl"),
    ("layer_1_swin_attention_output_1.pkl", "layer_1_tt_swin_attention_output_1.pkl"),
    ("layer_2_swin_attention_output_0.pkl", "layer_2_tt_swin_attention_output_0.pkl"),
    ("layer_2_swin_attention_output_1.pkl", "layer_2_tt_swin_attention_output_1.pkl"),
    ("layer_2_swin_attention_output_2.pkl", "layer_2_tt_swin_attention_output_2.pkl"),
    ("layer_2_swin_attention_output_3.pkl", "layer_2_tt_swin_attention_output_3.pkl"),
    ("layer_2_swin_attention_output_4.pkl", "layer_2_tt_swin_attention_output_4.pkl"),
    ("layer_2_swin_attention_output_5.pkl", "layer_2_tt_swin_attention_output_5.pkl"),
    ("layer_3_swin_attention_output_0.pkl", "layer_3_tt_swin_attention_output_0.pkl"),
    ("layer_3_swin_attention_output_1.pkl", "layer_3_tt_swin_attention_output_1.pkl"),
    ("layer_0_self_attention_output_0.pkl", "layer_0_tt_self_attention_output_0.pkl"),
    ("layer_0_self_attention_output_1.pkl", "layer_0_tt_self_attention_output_1.pkl"),
    ("layer_1_self_attention_output_0.pkl", "layer_1_tt_self_attention_output_0.pkl"),
    ("layer_1_self_attention_output_1.pkl", "layer_1_tt_self_attention_output_1.pkl"),
    ("layer_2_self_attention_output_0.pkl", "layer_2_tt_self_attention_output_0.pkl"),
    ("layer_2_self_attention_output_1.pkl", "layer_2_tt_self_attention_output_1.pkl"),
    ("layer_2_self_attention_output_2.pkl", "layer_2_tt_self_attention_output_2.pkl"),
    ("layer_2_self_attention_output_3.pkl", "layer_2_tt_self_attention_output_3.pkl"),
    ("layer_2_self_attention_output_4.pkl", "layer_2_tt_self_attention_output_4.pkl"),
    ("layer_2_self_attention_output_5.pkl", "layer_2_tt_self_attention_output_5.pkl"),
    ("layer_3_self_attention_output_0.pkl", "layer_3_tt_self_attention_output_0.pkl"),
    ("layer_3_self_attention_output_1.pkl", "layer_3_tt_self_attention_output_1.pkl"),
    ("swin_stage_input_0.pkl", "tt_swin_stage_input_0.pkl"),
    ("swin_stage_input_1.pkl", "tt_swin_stage_input_1.pkl"),
    ("swin_stage_input_2.pkl", "tt_swin_stage_input_2.pkl"),
    ("swin_stage_input_3.pkl", "tt_swin_stage_input_3.pkl"),
    ("layer_0_swin_layer_input_0.pkl", "layer_0_tt_swin_layer_input_0.pkl"),
    ("layer_0_swin_layer_input_1.pkl", "layer_0_tt_swin_layer_input_1.pkl"),
    ("layer_1_swin_layer_input_0.pkl", "layer_1_tt_swin_layer_input_0.pkl"),
    ("layer_1_swin_layer_input_1.pkl", "layer_1_tt_swin_layer_input_1.pkl"),
    ("layer_2_swin_layer_input_0.pkl", "layer_2_tt_swin_layer_input_0.pkl"),
    ("layer_2_swin_layer_input_1.pkl", "layer_2_tt_swin_layer_input_1.pkl"),
    ("layer_2_swin_layer_input_2.pkl", "layer_2_tt_swin_layer_input_2.pkl"),
    ("layer_2_swin_layer_input_3.pkl", "layer_2_tt_swin_layer_input_3.pkl"),
    ("layer_2_swin_layer_input_4.pkl", "layer_2_tt_swin_layer_input_4.pkl"),
    ("layer_2_swin_layer_input_5.pkl", "layer_2_tt_swin_layer_input_5.pkl"),
    ("layer_3_swin_layer_input_0.pkl", "layer_3_tt_swin_layer_input_0.pkl"),
    ("layer_3_swin_layer_input_1.pkl", "layer_3_tt_swin_layer_input_1.pkl"),
    ("layer_0_swin_attention_input_0.pkl", "layer_0_tt_swin_attention_input_0.pkl"),
    ("layer_0_swin_attention_input_1.pkl", "layer_0_tt_swin_attention_input_1.pkl"),
    ("layer_1_swin_attention_input_0.pkl", "layer_1_tt_swin_attention_input_0.pkl"),
    ("layer_1_swin_attention_input_1.pkl", "layer_1_tt_swin_attention_input_1.pkl"),
    ("layer_2_swin_attention_input_0.pkl", "layer_2_tt_swin_attention_input_0.pkl"),
    ("layer_2_swin_attention_input_1.pkl", "layer_2_tt_swin_attention_input_1.pkl"),
    ("layer_2_swin_attention_input_2.pkl", "layer_2_tt_swin_attention_input_2.pkl"),
    ("layer_2_swin_attention_input_3.pkl", "layer_2_tt_swin_attention_input_3.pkl"),
    ("layer_2_swin_attention_input_4.pkl", "layer_2_tt_swin_attention_input_4.pkl"),
    ("layer_2_swin_attention_input_5.pkl", "layer_2_tt_swin_attention_input_5.pkl"),
    ("layer_3_swin_attention_input_0.pkl", "layer_3_tt_swin_attention_input_0.pkl"),
    ("layer_3_swin_attention_input_1.pkl", "layer_3_tt_swin_attention_input_1.pkl"),
    ("layer_0_self_attention_input_0.pkl", "layer_0_tt_self_attention_input_0.pkl"),
    ("layer_0_self_attention_input_1.pkl", "layer_0_tt_self_attention_input_1.pkl"),
    ("layer_1_self_attention_input_0.pkl", "layer_1_tt_self_attention_input_0.pkl"),
    ("layer_1_self_attention_input_1.pkl", "layer_1_tt_self_attention_input_1.pkl"),
    ("layer_2_self_attention_input_0.pkl", "layer_2_tt_self_attention_input_0.pkl"),
    ("layer_2_self_attention_input_1.pkl", "layer_2_tt_self_attention_input_1.pkl"),
    ("layer_2_self_attention_input_2.pkl", "layer_2_tt_self_attention_input_2.pkl"),
    ("layer_2_self_attention_input_3.pkl", "layer_2_tt_self_attention_input_3.pkl"),
    ("layer_2_self_attention_input_4.pkl", "layer_2_tt_self_attention_input_4.pkl"),
    ("layer_2_self_attention_input_5.pkl", "layer_2_tt_self_attention_input_5.pkl"),
    ("layer_3_self_attention_input_0.pkl", "layer_3_tt_self_attention_input_0.pkl"),
    ("layer_3_self_attention_input_1.pkl", "layer_3_tt_self_attention_input_1.pkl"),
    ("Encoder_output.pkl", "tt_encoder_output.pkl"),
    ("encoder_output_in_swin_model.pkl", "tt_encoder_output_in_swin_model.pkl"),
    ("layer_Norm_weight.pkl", "tt_layer_nrom_weight.pkl"),
    ("layer_Norm_bias.pkl", "tt_layer_nrom_bias.pkl"),
    ("swin_model.pkl", "tt_swin_model.pkl"),
]


def get_file_names(path):
    file_name_list = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            file_name_list.append(line)
    return file_name_list


def compute_pcc(file_name_list, tensor_file_names):
    output = []
    for file_name in file_name_list:
        for tensor_files in tensor_file_names:
            if file_name == tensor_files[0]:
                with open(tensor_files[0], "rb") as file:
                    golden_tensor = pickle.load(file)

                with open(tensor_files[1], "rb") as file:
                    Calculated_tensor = pickle.load(file)

                cal_pcc = np.min(
                    np.ma.corrcoef(
                        np.ma.masked_invalid(
                            torch.squeeze(golden_tensor).detach().numpy()
                        ).flatten(),
                        np.ma.masked_invalid(
                            torch.squeeze(Calculated_tensor).detach().numpy()
                        ).flatten(),
                    )
                )
                output.append((tensor_files[0][:-4], tensor_files[1][:-4], cal_pcc))

    df = pd.DataFrame(output, columns=["Golden file", "Calculated file", "pcc"])
    return df


# compute pcc of both input and output tensors
path = "flow.txt"
file_name_list = get_file_names(path)
df = compute_pcc(file_name_list, tensor_files)
df.to_csv("swin_pcc.csv", index=False)

# compute pcc of input tensor
path = "input_flow.txt"
file_name_list = get_file_names(path)
df = compute_pcc(file_name_list, tensor_files)
df.to_csv("swin_input_pcc.csv", index=False)

# compute pcc of output tensor
path = "output_flow.txt"
file_name_list = get_file_names(path)
df = compute_pcc(file_name_list, tensor_files)
df.to_csv("swin_output_pcc.csv", index=False)
