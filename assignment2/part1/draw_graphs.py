import numpy as np
import matplotlib.pyplot as plt

# results from LISA
results_resnet18 = {
    'none': 0.6578,
    'gaussian_noise-1': 0.6487, 'gaussian_noise-2': 0.6315, 'gaussian_noise-3': 0.5957,
    'gaussian_noise-4': 0.5329, 'gaussian_noise-5': 0.4019, 'gaussian_blur-1': 0.6565,
    'gaussian_blur-2': 0.6395, 'gaussian_blur-3': 0.6178, 'gaussian_blur-4': 0.5502,
    'gaussian_blur-5': 0.4396, 'contrast-1': 0.622, 'contrast-2': 0.5362, 'contrast-3': 0.4428,
    'contrast-4': 0.3542, 'contrast-5': 0.2819, 'jpeg-1': 0.6537, 'jpeg-2': 0.6507, 'jpeg-3': 0.6508,
    'jpeg-4': 0.643, 'jpeg-5': 0.6114
}
results_resnet34 = {
    'none': 0.6067,
    'gaussian_noise-1': 0.6029, 'gaussian_noise-2': 0.5971, 'gaussian_noise-3': 0.5732, 'gaussian_noise-4': 0.534, 'gaussian_noise-5': 0.4273,
    'gaussian_blur-1': 0.6067, 'gaussian_blur-2': 0.5956, 'gaussian_blur-3': 0.5721, 'gaussian_blur-4': 0.523, 'gaussian_blur-5': 0.4416,
    'contrast-1': 0.5563, 'contrast-2': 0.4654, 'contrast-3': 0.3937, 'contrast-4': 0.3238, 'contrast-5': 0.27,
    'jpeg-1': 0.604, 'jpeg-2': 0.603, 'jpeg-3': 0.6022, 'jpeg-4': 0.5957, 'jpeg-5': 0.5645
}
results_vgg11 = {
    'none': 0.7599,
    'gaussian_noise-1': 0.7296, 'gaussian_noise-2': 0.6509, 'gaussian_noise-3': 0.5441, 'gaussian_noise-4': 0.3746, 'gaussian_noise-5': 0.167,
    'gaussian_blur-1': 0.7604, 'gaussian_blur-2': 0.7425, 'gaussian_blur-3': 0.7041, 'gaussian_blur-4': 0.6131, 'gaussian_blur-5': 0.4812,
    'contrast-1': 0.745, 'contrast-2': 0.7021, 'contrast-3': 0.6472, 'contrast-4': 0.5591, 'contrast-5': 0.4281,
    'jpeg-1': 0.7584, 'jpeg-2': 0.7508, 'jpeg-3': 0.7449, 'jpeg-4': 0.7327, 'jpeg-5': 0.6602
}
results_vgg11_bn = {
    'none': 0.759,
    'gaussian_noise-1': 0.6878, 'gaussian_noise-2': 0.5738, 'gaussian_noise-3': 0.4624, 'gaussian_noise-4': 0.3255, 'gaussian_noise-5': 0.1922,
    'gaussian_blur-1': 0.7609, 'gaussian_blur-2': 0.7454, 'gaussian_blur-3': 0.6962, 'gaussian_blur-4': 0.5945, 'gaussian_blur-5': 0.4532,
    'contrast-1': 0.7453, 'contrast-2': 0.6887, 'contrast-3': 0.6142, 'contrast-4': 0.5023, 'contrast-5': 0.3853,
    'jpeg-1': 0.7479, 'jpeg-2': 0.7387, 'jpeg-3': 0.7275, 'jpeg-4': 0.7114, 'jpeg-5': 0.6257
}
results_densenet121 = {
    'none': 0.668,
    'gaussian_noise-1': 0.6526, 'gaussian_noise-2': 0.5974, 'gaussian_noise-3': 0.5222, 'gaussian_noise-4': 0.4027, 'gaussian_noise-5': 0.2362,
    'gaussian_blur-1': 0.6719, 'gaussian_blur-2': 0.6543, 'gaussian_blur-3': 0.611, 'gaussian_blur-4': 0.5182, 'gaussian_blur-5': 0.4079,
    'contrast-1': 0.6195, 'contrast-2': 0.4985, 'contrast-3': 0.3965, 'contrast-4': 0.2992, 'contrast-5': 0.2187,
    'jpeg-1': 0.6646, 'jpeg-2': 0.6602, 'jpeg-3': 0.6522, 'jpeg-4': 0.6482, 'jpeg-5': 0.6028
}

legend_kwargs = {
    'loc': 'upper center',
    'bbox_to_anchor': (0.5, -0.05),
    'fancybox': True,
    'shadow': True,
    'ncol': 1
}


def compute_ce(relative: bool =False):
    model_data = [results_resnet18, results_resnet34, results_vgg11_bn, results_vgg11, results_densenet121]
    for c in ['gaussian_noise', 'gaussian_blur', 'contrast', 'jpeg']:
        c_points = []
        for data in model_data:
            denominator = 0
            nominator = 0
            for s in range(1, 6):
                nominator += (1 - data[f"{c}-{s}"])
                denominator += (1 - results_resnet18[f"{c}-{s}"])
                if relative:
                    nominator -= (1 - data['none'])
                    denominator -= (1 - results_resnet18['none'])
            c_points.append(nominator/denominator)

        x_point = np.arange(1, len(model_data) + 1)
        plt.xlabel("Model")
        plt.ylabel(f"{'RCE' if relative else 'CE'} ")

        plt.plot(x_point, c_points, label=c)

    plt.xticks(x_point, ['resnet18', 'resnet34', 'vgg11bn', 'vgg11', 'densenett'])

    plt.title(f"{'RCE' if relative else 'CE'} per corruption function per model")
    plt.legend(**legend_kwargs)
    plt.show()

compute_ce()
compute_ce(True)

def graph_resnet18():
    x_point = np.arange(1, 6)
    plt.xlabel("Severity")
    plt.ylabel("Accuracy")
    plt.title("Accuracy loss by image distortion method")
    plt.plot(x_point, [
        results_resnet18['gaussian_blur-1'],
        results_resnet18['gaussian_blur-2'],
        results_resnet18['gaussian_blur-3'],
        results_resnet18['gaussian_blur-4'],
        results_resnet18['gaussian_blur-5']
    ], label="Gaussian blur")
    plt.xticks(x_point)
    plt.plot(x_point, [
        results_resnet18['gaussian_noise-1'],
        results_resnet18['gaussian_noise-2'],
        results_resnet18['gaussian_noise-3'],
        results_resnet18['gaussian_noise-4'],
        results_resnet18['gaussian_noise-5']
    ], label="Gaussian noise")
    plt.plot(x_point, [
        results_resnet18['jpeg-1'],
        results_resnet18['jpeg-2'],
        results_resnet18['jpeg-3'],
        results_resnet18['jpeg-4'],
        results_resnet18['jpeg-5']
    ], label="JPEG compression loss")
    plt.plot(x_point, [
        results_resnet18['contrast-1'],
        results_resnet18['contrast-2'],
        results_resnet18['contrast-3'],
        results_resnet18['contrast-4'],
        results_resnet18['contrast-5']
    ], label="Contrast")
    plt.legend(**legend_kwargs)
    plt.show()
