import numpy as np
import matplotlib.pyplot as plt


def dirichlet(client_num, class_num, alpha):
    dirichlet_pdf = np.random.dirichlet([alpha / class_num] * class_num, client_num)
    for idx in range(client_num):
        local_pdf = dirichlet_pdf[idx]

        local_pdf_list = []
        start_index = 0
        for class_idx in range(class_num):
            local_pdf_list.append((start_index, local_pdf[class_idx]))
            start_index += local_pdf[class_idx]

        plt.broken_barh(local_pdf_list,
                        (idx*4, 3),
                        facecolors=("#1C62A5", "#FF7F50", "#279321", "#C9101F", "#804FAF", "#A1433A", "#D95DB5", "#808080", "#AEB31A", "#1FB2C4"),
                        alpha=1)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    dirichlet(20, 10, 0.1)

