from ControlNet.cldm.hack import disable_verbosity, enable_sliced_attention

save_memory = False

disable_verbosity()

if save_memory:
    enable_sliced_attention()
