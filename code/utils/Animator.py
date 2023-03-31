from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from IPython import display


class Animator:
    """For plotting data in animation."""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        self.use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        # self.config_axes = lambda: plt.set_axes(
        #     self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

        # 设置坐标轴
        self.axes[0].set_xlabel("epoch")
        self.axes[0].set_ylabel(ylabel)
        self.axes[0].set_xscale(xscale)
        self.axes[0].set_yscale(yscale)
        self.axes[0].set_xlim(xlim)
        self.axes[0].set_ylim(ylim)
        if legend:
            self.axes[0].legend(legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        # self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

    def use_svg_display(self):
        """Use the svg format to display a plot in Jupyter.

        Defined in :numref:`sec_calculus`"""
        backend_inline.set_matplotlib_formats('svg')
