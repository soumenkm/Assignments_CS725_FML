from dataclasses import dataclass

ep_list = [100,250,500,1000]
lr_list = [1e-1, 1e-2, 1e-4, 1e-6]
mom_list = [0,0.9]
model_list = ['logistic_regression','linear_classifier']
ds_list = ['binary','iris']

@dataclass
class TrainingArguments:
    log_dir: str = './checkpoints/'    # Directory where the models will be stored
    dataset: str = ds_list[0]           # Dataset to run the experiments on (binary/digits)
    model: str = model_list[0] # Model to use (linear_classifier/logistic_regression)
    num_epochs: int = ep_list[1]            # Number of epochs for the optimization
    learning_rate: float = lr_list[0]       # Learning rate for the optimization
    momentum: float = mom_list[0]                # Momentum term for the optimization

@dataclass
class TrainingWithVisualizationArguments:
    log_dir: str = './checkpoints/'    # Directory where the models will be stored
    num_epochs: int = ep_list[1]           # Number of epochs for the optimization
    learning_rate: float = lr_list[0]   # Learning rate for the optimization
    momentum: float = mom_list[0]                # Momentum term for the optimization
    grid_size: int = 128               # How many points should be evaluated on one axis? Total number of evaluation points is grid_size * grid_size
    epsilon: float = 0.75              # Padding around the [x1_min,x2_max], [y1_min, y2_max] box
    contourf_alpha: float = 0.25       # Alpha value for the contourf
    contour_linewidth: float = 0.25    # Contour linewidth
    cmap: str = 'bwr'                  # Colormap to use from here: `https://matplotlib.org/stable/tutorials/colors/colormaps.html`
    gif_fps: int = 60                  # Animation FPS
    gif_bitrate: int = 1800            # Animation bitrate
    gif_dpi: int = 100                 # Animation DPI
