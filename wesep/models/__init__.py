import wesep.models.tse_bsrnn_spk as bsrnn_spk
import wesep.models.tse_bsrnn_visual as bsrnn_visual
import wesep.models.tse_bsrnn_spatial as bsrnn_spatial
import wesep.models.tse_nbc2_spatial as nbc2_spatial
from wesep.models.tse_online_avcrossnet_mamba_visual import OnlineAVCrossNetMamba


def get_model(model_name: str):
    if model_name.startswith("TSE_BSRNN_SPK"):
        return getattr(bsrnn_spk, model_name)
    elif model_name.startswith("TSE_BSRNN_VISUAL"):
        return getattr(bsrnn_visual, model_name)
    elif model_name.startswith("TSE_BSRNN_SPATIAL"):
        return getattr(bsrnn_spatial, model_name)
    elif model_name.startswith("TSE_NBC2_SPATIAL"):
        return getattr(nbc2_spatial, model_name)
    elif model_name.startswith("TSE_ONLINE_AVCROSSNET_MAMBA_VISUAL"):
        return OnlineAVCrossNetMamba
    else:  # model_name error !!!
        print(model_name + " not found !!!")
        exit(1)
