from __future__ import annotations

from dataclasses import field
from typing import Any, List, Union

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer settings."""

    _target_: str
    """Target optimizer class (e.g., torch.optim.AdamW)."""

    weight_decay: float = 0.001
    """Weight decay parameter for the optimizer."""


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for neural network layer settings."""

    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    """List of hidden layer dimensions."""

    activation: str = "ELU"
    """Activation function name."""

    dropout_prob: float = 0.0
    """Dropout probability."""

    use_layer_norm: bool = False
    """Whether to use layer normalization."""

    encoder_activation: str = "ELU"
    """Activation function name for encoder layers."""

    encoder_output_dim: int | None = None
    """Output dimension for encoder. Only used for encoder modules."""

    encoder_hidden_dims: List[int] | None = None
    """Hidden dimensions for encoder. Only used for encoder modules."""

    encoder_input_name: str = ""
    """Input name for encoder. Only used for encoder modules."""

    input_channels: int = 1
    """Number of input channels. Only used for CNN modules."""

    input_height: int = 1
    """Height of input feature maps. Only used for CNN modules."""

    input_width: int = 1
    """Width of input feature maps. Only used for CNN modules."""

    hidden_channels: tuple[int, ...] | None = None
    """Hidden channel dimensions. Only used for CNN modules."""

    kernel_size: int | tuple[int, ...] = 3
    """Kernel size for convolutions. Only used for CNN modules."""

    stride: int | tuple[int, ...] = 1
    """Stride for convolutions. Only used for CNN modules."""

    padding: str | int | tuple[str | int, ...] = "same"
    """Padding mode for convolutions. Only used for CNN modules."""

    module_input_name: tuple[str, ...] = ()
    """Input names for module. Only used for encoder modules."""


@dataclass(frozen=True)
class ModuleConfig:
    """Configuration for neural network modules."""

    type: str
    """Module type (e.g., MLP)."""

    input_dim: List[str] = field(default_factory=list)
    """Input dimension specification."""

    output_dim: List[str | int] = field(default_factory=list)
    """Output dimension specification."""

    layer_config: LayerConfig = field(default_factory=LayerConfig)
    """Layer configuration settings."""

    min_noise_std: float | None = None
    """Minimum noise standard deviation."""

    min_mean_noise_std: float | None = None
    """Minimum mean noise standard deviation."""


@dataclass(frozen=True)
class MotionEncoderConfig:
    """Configuration for motion encoder (Conv1D-based)."""
    
    input_dim: int = 0
    """Input dimension per timestep (will be auto-calculated from observation)."""
    
    hidden_dim: int = 60
    """Hidden dimension after initial linear layer."""
    
    output_dim: int = 128
    """Output dimension of the encoder."""
    
    num_timesteps: int = 20
    """Number of future timesteps."""
    
    activation: str = "SiLU"
    """Activation function name."""


@dataclass(frozen=True)
class HeightMapEncoderConfig:
    """Configuration for height map encoder (CNN + CrossAttention)."""

    latent_dim: int = 64
    """Latent dimension for the encoder output and attention."""

    num_heads: int = 16
    """Number of heads for multi-head attention."""

    map_height: int = 11
    """Height of the height map grid."""

    map_width: int = 17
    """Width of the height map grid."""

    num_channels: int = 3
    """Number of channels per grid point (default: x, y, height)."""

    conv_hidden_channels: int = 16
    """Hidden channels in Conv2D feature extractor."""

    combine_mode: str = "concat"
    """How to combine heightmap latent with the policy MLP. Either ``"concat"``
    (concatenate the latent with actor/critic obs before the MLP, the original
    VideoMimic ``flatten_then_embed_with_attention`` behaviour) or
    ``"add_to_hidden"`` (add the latent to the activation right after the MLP's
    first Linear layer — VideoMimic's ``flatten_then_embed_with_attention_to_hidden``
    behaviour). When ``"add_to_hidden"`` is used, ``latent_dim`` must equal the
    actor/critic's first hidden dim, and the pretrained MLP weights remain shape-compatible
    with the finetuned model (the heightmap branch is gated to zero at init via
    the learnable attention parameter)."""


@dataclass(frozen=True)
class PPOModuleDictConfig:
    """Configuration for PPO module dictionary."""

    actor: ModuleConfig
    """Actor module configuration."""

    critic: ModuleConfig
    """Critic module configuration."""

    motion_encoder: MotionEncoderConfig | None = None
    """Motion encoder configuration (optional, for future motion targets)."""

    height_map_encoder: HeightMapEncoderConfig | None = None
    """Height map encoder configuration (optional, for terrain perception)."""


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for PPO algorithm."""

    module_dict: PPOModuleDictConfig
    """PPO module configurations (actor, critic)."""

    num_learning_epochs: int = 8
    """Number of learning epochs per update."""

    num_mini_batches: int = 4
    """Number of mini-batches per epoch."""

    clip_param: float = 0.2
    """PPO clipping parameter."""

    gamma: float = 0.99
    """Discount factor for future rewards."""

    lam: float = 0.95
    """GAE lambda parameter."""

    value_loss_coef: float = 1.0
    """Value loss coefficient."""

    entropy_coef: float = 0.01
    """Entropy coefficient for exploration."""

    actor_learning_rate: float = 1e-5
    """Learning rate for actor network."""

    actor_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Actor optimizer configuration."""

    critic_learning_rate: float = 1e-5
    """Learning rate for critic network."""

    critic_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Critic optimizer configuration."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    schedule: str = "adaptive"
    """Learning rate schedule type."""

    desired_kl: float = 0.01
    """Desired KL divergence for adaptive learning rate."""

    use_symmetry: bool = False
    """Whether to use symmetry in training."""

    symmetry_actor_coef: float = 1.0
    """Symmetry coefficient for actor."""

    symmetry_critic_coef: float = 0.0
    """Symmetry coefficient for critic."""

    num_steps_per_env: int = 24
    """Number of steps per environment."""

    save_interval: int = 100
    """Interval for saving model checkpoints."""

    load_optimizer: bool = True
    """Whether to load optimizer state."""

    init_noise_std: float = 0.8
    """Initial noise standard deviation."""

    num_learning_iterations: int = 1000000
    """Total number of learning iterations."""

    init_at_random_ep_len: bool = True
    """Whether to initialize at random episode length."""

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""

    eval_interval: int = 0
    """Run evaluation every N training iterations (0 = no eval during training)."""

    max_actor_learning_rate: float | None = None
    min_actor_learning_rate: float | None = None
    max_critic_learning_rate: float | None = None
    min_critic_learning_rate: float | None = None

    # ────────────────────────────────────────────────────────────────────
    # Optional teacher-BC scheduler (off by default)
    #
    # When ``teacher_checkpoint`` is set, a frozen teacher actor is loaded
    # at setup time, queried each rollout step on its own obs slice, and
    # contributes a behaviour-cloning loss that is linearly faded into the
    # PPO loss according to the schedule below. This lets a single PPO run
    # do "stage-3 + stage-4" in one shot, matching the BC→PPO transition
    # the user asks for.
    #
    # Schedule (in iterations of `current_learning_iteration`):
    #   it < bc_warmup_iters                                  → bc=1.0,  ppo=0.0
    #   bc_warmup_iters <= it < bc_warmup_iters + bc_to_ppo_iters
    #         alpha = (it - bc_warmup_iters) / bc_to_ppo_iters
    #         bc=1-alpha, ppo=alpha
    #   it >= bc_warmup_iters + bc_to_ppo_iters                → bc=0.0,  ppo=1.0
    #
    # The critic value loss is always on (it warms up from rewards
    # regardless of phase). bc_loss_coef_max scales the BC actor loss
    # during the warmup phase; ppo coefficients scale (surrogate, entropy,
    # symmetry-actor) terms.
    # ────────────────────────────────────────────────────────────────────
    teacher_checkpoint: str = ""
    """Path to teacher .pt for the BC term. Empty disables BC entirely
    (plain PPO behaviour, no extra cost)."""

    teacher_actor_obs_key: str = "teacher_actor_obs"
    """Obs group the env publishes for the teacher's actor input."""

    teacher_actor_obs_history_key: str | None = None
    """Optional history key for teacher obs (defaults to
    ``teacher_actor_obs_key``)."""

    bc_warmup_iters: int = 0
    """Iterations of pure BC (PPO actor loss off, critic value loss still on)
    at the start of training. 0 disables the warmup phase."""

    bc_to_ppo_iters: int = 0
    """Iterations of the linear BC→PPO transition that follows the warmup.
    0 means an instant switch the moment the warmup ends."""

    bc_loss_coef_max: float = 1.0
    """Max BC actor-loss coefficient (active during warmup)."""

    bc_sigma_loss_coef: float = 1.0
    """Coefficient on the (sigma_student − sigma_teacher) term of the BC loss.
    Mirrors ``DAggerConfig.bc_sigma_loss_coef``."""

    clip_teacher_actions: bool = True
    """Clip teacher action targets before the BC loss."""

    clip_actions_threshold: float = 8.0
    """Symmetric clip range for the teacher actions when
    ``clip_teacher_actions`` is True."""


@dataclass(frozen=True)
class FastSACConfig:
    num_learning_iterations: int = 25000
    """total timesteps of the experiments"""

    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""

    actor_learning_rate: float = 3e-4
    """the learning rate for the actor"""

    alpha_learning_rate: float = 3e-4
    """the learning rate for the alpha"""

    buffer_size: int = 1024
    """the replay memory buffer size per environment"""

    num_steps: int = 1
    """the number of steps to use for the multi-step return"""

    gamma: float = 0.97
    """the discount factor gamma"""

    tau: float = 0.125
    """target smoothing coefficient (default: 0.005)"""

    batch_size: int = 8192
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""

    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    num_updates: int = 8
    """the number of updates to perform per step"""

    target_entropy_ratio: float = 0.0
    """the ratio of the target entropy to the number of actions"""

    num_atoms: int = 101
    """the number of atoms"""

    v_min: float = -20.0
    """the minimum value of the support"""

    v_max: float = 20.0
    """the maximum value of the support"""

    critic_hidden_dim: int = 768
    """the hidden dimension of the critic network"""

    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""

    use_symmetry: bool = False
    """whether to use symmetry"""

    alpha_init: float = 0.001
    """the initial value of the alpha"""

    use_autotune: bool = True
    """whether to use autotune for the alpha"""

    use_tanh: bool = True
    """whether to use tanh for the action"""

    log_std_max: float = 0.0
    """the maximum value of the log std"""

    log_std_min: float = -5.0
    """the minimum value of the log std"""

    compile: bool = True
    """whether to use torch.compile."""

    obs_normalization: bool = True
    """whether to enable observation normalization"""

    use_layer_norm: bool = True
    """whether to use layer normalization"""

    num_q_networks: int = 2
    """number of Q-networks to ensemble"""

    max_grad_norm: float = 0.0
    """the maximum gradient norm"""

    amp: bool = True
    """whether to use amp"""

    amp_dtype: str = "bf16"
    """the dtype of the amp"""

    weight_decay: float = 0.001
    """the weight decay of the optimizer"""

    save_interval: int = 1000
    """the interval to save the model"""

    logging_interval: int = 100
    """the interval to log the metrics"""

    encoder_obs_key: str = "perception_obs"
    """the key of the encoder observation. only valid if use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """the shape of the encoder observation. only valid if use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN for the encoder"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])


@dataclass(frozen=True)
class DAggerModuleDictConfig:
    """Configuration for DAgger student module dictionary.

    Pure DAgger has no critic — only an actor + optional perception encoders that
    mirror the teacher (heightmap, etc.). The motion encoder is intentionally
    omitted by default: the student's whole point is to *not* see future motion
    targets.
    """

    actor: ModuleConfig
    """Actor module configuration."""

    motion_encoder: MotionEncoderConfig | None = None
    """Optional motion encoder for the student. Default None — student does not
    see future motion targets in the VideoMimic-style setup."""

    height_map_encoder: HeightMapEncoderConfig | None = None
    """Optional height map encoder configuration."""


@dataclass(frozen=True)
class DAggerConfig:
    """Configuration for the DAgger distillation algorithm (pure DAgger, no critic)."""

    module_dict: DAggerModuleDictConfig
    """Student module configurations (actor + optional perception encoders)."""

    policy_to_clone: str = ""
    """Path to the teacher checkpoint (.pt file). The teacher's experiment config
    is recovered from the checkpoint metadata, so the only field needed here is
    the file path."""

    teacher_actor_obs_key: str = "teacher_actor_obs"
    """Observation group key the env publishes for the teacher's actor. The
    student algo remaps this onto the teacher's expected ``actor_obs`` slot at
    rollout time."""

    teacher_actor_obs_history_key: str | None = None
    """Optional history key for the teacher's actor obs (defaults to
    ``teacher_actor_obs_key``)."""

    take_teacher_actions: bool = False
    """If True, env steps with teacher actions instead of student actions
    (β-DAgger). Default False — student rollout, on-policy."""

    clip_teacher_actions: bool = False
    """Clip teacher action targets to ``[-clip_actions_threshold, +clip_actions_threshold]``
    before computing the BC loss. Useful when the teacher occasionally produces
    out-of-range actions and we don't want the student chasing them."""

    clip_actions_threshold: float = 100.0
    """Threshold used for ``clip_teacher_actions``."""

    bc_sigma_loss_coef: float = 1.0
    """Coefficient on the (sigma_student − sigma_teacher) term of the BC loss.
    Set to 0.0 to skip matching action noise std."""

    actor_learning_rate: float = 1e-3
    """Learning rate for the student actor."""

    actor_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Student actor optimizer configuration."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    num_steps_per_env: int = 24
    """Number of rollout steps per env per iteration."""

    num_learning_epochs: int = 5
    """Number of passes over the rollout buffer per update."""

    num_mini_batches: int = 4
    """Number of mini-batches per epoch."""

    init_noise_std: float = 0.8
    """Initial noise std for the student actor."""

    save_interval: int = 1000
    """Iteration interval for saving checkpoints."""

    eval_interval: int = 0
    """Run evaluation every N iterations (0 = disabled)."""

    eval_callbacks: Any = None
    """Evaluation callbacks configuration (mirrors PPOConfig.eval_callbacks)."""

    num_learning_iterations: int = 1000000
    """Total number of learning iterations."""

    init_at_random_ep_len: bool = False
    """Whether to initialize episode length buffers randomly at start."""

    load_optimizer: bool = True
    """Whether to load student optimizer state when resuming."""


@dataclass(frozen=True)
class PPOAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: PPOConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class FastSACAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: FastSACConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class DAggerAlgoConfig:
    """Configuration for algorithm wrapper (DAgger)."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: DAggerConfig
    """Algorithm-specific configuration."""


AlgoInitConfig = Union[PPOConfig, FastSACConfig, DAggerConfig]

AlgoConfig = Union[PPOAlgoConfig, FastSACAlgoConfig, DAggerAlgoConfig]
