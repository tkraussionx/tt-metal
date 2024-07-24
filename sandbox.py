import pandas as pd

df = pd.read_csv("/home/ubuntu/tt-metal/results_addcmul_bw_insysmem/backward_addcmul_sweep.csv")


df_bva = df.loc[df["test_output"] == "bad_variant_access"]

print(df_bva["args"].tolist()[0])
