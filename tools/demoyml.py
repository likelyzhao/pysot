from pysot.core.config import cfg

cfg.merge_from_file("experiments/siamrpn_r50_l234_dwxcorr_8gpu/config.yaml")
print(cfg)