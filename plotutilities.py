from matplotlib import pyplot
import os

def plot(list_of_things_to_plot, legend, title, plots_folder):
    '''plot the list of things to plot in a plot with the given legend, title and colors,
        then save the plot in the given folder as title.png'''

    pyplot.figure()

    for things_to_plot in list_of_things_to_plot:
        pyplot.plot(things_to_plot)

    pyplot.xlim([0, 365])
    pyplot.legend(legend)
    pyplot.title(title)
    pyplot.xlabel('Days')
    pyplot.savefig(os.path.join(plots_folder, title + '.png'))
