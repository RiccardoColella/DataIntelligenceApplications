from matplotlib import pyplot
import os

def plot(list_of_things_to_plot, legend, title, plots_folder, color=0):
    '''plot the list of things to plot in a plot with the given legend, title and colors,
        then save the plot in the given folder as title.png'''

    pyplot.figure()

    palette_list=[['#006A4E', '#FFF154','#FF00FF', '#87CEEB']]

    if color == 0:
        for things_to_plot in list_of_things_to_plot:
            pyplot.plot(things_to_plot)
    else:
        col = palette_list[color-1]
        for i in range(len(list_of_things_to_plot)):
            pyplot.plot(list_of_things_to_plot[i],col[i])

    pyplot.xlim([0, 365])
    pyplot.legend(legend)
    pyplot.title(title)
    pyplot.xlabel('Days')
    pyplot.savefig(os.path.join(plots_folder, title + '.png'))
