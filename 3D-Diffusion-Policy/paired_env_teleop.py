from diffusion_policy_3d.env.adroit.rrl_local.rrl_multicam_pair import BasicPairedAdroitEnv 
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = BasicPairedAdroitEnv("door-v0", ['top'], latent_dim=7056, hybrid_state=True, 
                               channels_first=True, height=84, width=84, test_image=False, 
                               num_repeats=2, num_frames=1, encoder_type=None, device='cuda')
    env.reset()
    img = env.get_paired_display()
    plt.imshow(img)
    plt.show()

