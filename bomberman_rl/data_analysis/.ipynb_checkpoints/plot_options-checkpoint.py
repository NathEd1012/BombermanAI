from cycler import cycler
from matplotlib import rcParams

# Enable grid and adjust grid appearance
rcParams['axes.grid'] = True
rcParams['grid.linestyle'] = '--'
rcParams['grid.alpha'] = 0.5

# Set label and title font sizes
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 22
rcParams['figure.figsize'] = (12, 7)
rcParams['figure.titlesize'] = 26
rcParams['font.size'] = 16

# Set color palette for the plots
color_palette = ['#C61A27', '#FBB13C', '#B6C649', '#3891A6', '#40376E', '#B88C9E']
rcParams['axes.prop_cycle'] = cycler('color', color_palette)

# Set colormap for images
rcParams['image.cmap'] = 'magma'

# Set line and marker styles
rcParams['lines.marker'] = 'o'  # Use circle markers
rcParams['lines.markeredgewidth'] = 2  # Width of the edge of markers
rcParams['lines.markerfacecolor'] = 'white'  # White fill for the marker face
#rcParams['lines.markerfacecoloralt'] = 'none'  # Alternate face color, none to keep it white
rcParams['markers.fillstyle'] = 'full'  # Full to fill the circle completely

# #B88C9E #FBB13C #B6C649
# #FFBC42 #7EB77F #CE84AD