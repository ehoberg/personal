import csv
import numpy as np
import matplotlib.pyplot as plt


# You can copy this code to your personal pipeline project or execute it here.
def plot_data(csv_file_path: str):
    """
    This code plots the precision-recall curve based on data from a .csv file,
    where precision is on the x-axis and recall is on the y-axis.
    It it not so important right now what precision and recall means.

    :param csv_file_path: The CSV file containing the data to plot.


    **This method is part of a series of debugging exercises.**
    **Each Python method of this series contains bug that needs to be found.**

    | ``1   For some reason the plot is not showing correctly, can you find out what is going wrong?``
    | ``2   How could this be fixed?``

    This example demonstrates the issue.
    It first generates some data in a csv file format and the plots it using the ``plot_data`` method.
    If you manually check the coordinates and then check the plot, they do not correspond.

    >>> f = open("data_file.csv", "w")
    >>> w = csv.writer(f)
    >>> _ = w.writerow(["precision", "recall"])
    >>> w.writerows([[0.013,0.951],
    ...              [0.376,0.851],
    ...              [0.441,0.839],
    ...              [0.570,0.758],
    ...              [0.635,0.674],
    ...              [0.721,0.604],
    ...              [0.837,0.531],
    ...              [0.860,0.453],
    ...              [0.962,0.348],
    ...              [0.982,0.273],
    ...              [1.0,0.0]])
    >>> f.close()
    >>> plot_data('data_file.csv')
    """
    # load data
    results = []
    with open(csv_file_path) as result_csv:
        csv_reader = csv.reader(result_csv, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            results.append([float(row[0]), float(row[1])]) if len(row) == 2 else None
        results = np.stack(results)

    
    # plot precision-recall curve
    plt.plot(results[:, 0], results[:, 1])
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    
    plt.grid(True)
    
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.show(block=True)


def main():
    """Main entry point of the program."""
    print("Hello, World!")


    f = open("c:\\temp\\data_file.csv", "w", newline='')
    w = csv.writer(f)
    _ = w.writerow(["precision", "recall"])
    w.writerows([[0.013,0.951],
                [0.376,0.851],
                [0.441,0.839],
                [0.570,0.758],
                [0.635,0.674],
                [0.721,0.604],
                [0.837,0.531],
                [0.860,0.453],
                [0.962,0.348],
                [0.982,0.273],
                [1.0,0.0]])
    f.close()
    plot_data("data_file.csv")




if __name__ == "__main__":
    main()