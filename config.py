from loguru import logger

# ========== General Parameters ==========
TRAIN_VAE = False
SAMPLE_GRAPHS = False
NUM_SAMPLES = 1000

logger.info(
    f"Loaded general parameters: TRAIN_VAE={TRAIN_VAE}, NUM_SAMPLES={NUM_SAMPLES}, SAMPLE_GRAPHS={SAMPLE_GRAPHS}"
)

# ========== GraphVAE Hyperparameters ==========
IN_CHANNELS = 7
MAX_NODES = 28
HIDDEN_CHANNELS = 64
LATENT_DIM = 32
EPOCHS = 1000
NUM_LAYERS = 5

logger.info(
    f"Loaded GraphVAE hyperparameters: IN_CHANNELS={IN_CHANNELS}, MAX_NODES={MAX_NODES}, "
    f"HIDDEN_CHANNELS={HIDDEN_CHANNELS}, LATENT_DIM={LATENT_DIM}, EPOCHS={EPOCHS}, NUM_LAYERS={NUM_LAYERS}"
)
