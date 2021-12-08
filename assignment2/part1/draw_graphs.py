import numpy as np
import matplotlib.pyplot as plt

# results from LISA
results_resnet18_epochs250 = {
    'none': 0.6597, 'gaussian_noise-1': 0.6498, 'gaussian_noise-2': 0.6309, 'gaussian_noise-3': 0.6008, 'gaussian_noise-4': 0.5411, 'gaussian_noise-5': 0.4189,
    'gaussian_blur-1': 0.6624, 'gaussian_blur-2': 0.6502, 'gaussian_blur-3': 0.6187, 'gaussian_blur-4': 0.5547, 'gaussian_blur-5': 0.4563,
    'contrast-1': 0.6257, 'contrast-2': 0.5471, 'contrast-3': 0.463, 'contrast-4': 0.3661, 'contrast-5': 0.2917,
    'jpeg-1': 0.6548, 'jpeg-2': 0.6494, 'jpeg-3': 0.6489, 'jpeg-4': 0.6434, 'jpeg-5': 0.5978
}
results_resnet34_epochs250 = {
    'none': 0.62,
    'gaussian_noise-1': 0.6109, 'gaussian_noise-2': 0.6003, 'gaussian_noise-3': 0.5789, 'gaussian_noise-4': 0.5369, 'gaussian_noise-5': 0.436,
    'gaussian_blur-1': 0.6194, 'gaussian_blur-2': 0.6108, 'gaussian_blur-3': 0.5945, 'gaussian_blur-4': 0.542, 'gaussian_blur-5': 0.4399,
    'contrast-1': 0.5723, 'contrast-2': 0.4865, 'contrast-3': 0.4003, 'contrast-4': 0.3182, 'contrast-5': 0.2448,
    'jpeg-1': 0.6196, 'jpeg-2': 0.6161, 'jpeg-3': 0.6115, 'jpeg-4': 0.6085, 'jpeg-5': 0.5686
}
results_vgg11_epochs250 = {
    'none': 0.777,
    'gaussian_noise-1': 0.749, 'gaussian_noise-2': 0.6665, 'gaussian_noise-3': 0.5537, 'gaussian_noise-4': 0.3593, 'gaussian_noise-5': 0.1523,
    'gaussian_blur-1': 0.7796, 'gaussian_blur-2': 0.7653, 'gaussian_blur-3': 0.7285, 'gaussian_blur-4': 0.6484, 'gaussian_blur-5': 0.5204,
    'contrast-1': 0.7753, 'contrast-2': 0.7545, 'contrast-3': 0.7136, 'contrast-4': 0.6313, 'contrast-5': 0.4812,
    'jpeg-1': 0.7708, 'jpeg-2': 0.761, 'jpeg-3': 0.7551, 'jpeg-4': 0.738, 'jpeg-5': 0.6532
}
results_vgg11_bn_epochs250 = {
    'none': 0.7774,
    'gaussian_noise-1': 0.717, 'gaussian_noise-2': 0.6037, 'gaussian_noise-3': 0.4878, 'gaussian_noise-4': 0.3344, 'gaussian_noise-5': 0.1813,
    'gaussian_blur-1': 0.7793, 'gaussian_blur-2': 0.7621, 'gaussian_blur-3': 0.7179, 'gaussian_blur-4': 0.6152, 'gaussian_blur-5': 0.4755,
    'contrast-1': 0.7616, 'contrast-2': 0.7098, 'contrast-3': 0.6442, 'contrast-4': 0.5369, 'contrast-5': 0.4109,
    'jpeg-1': 0.7688, 'jpeg-2': 0.7525, 'jpeg-3': 0.7406, 'jpeg-4': 0.7268, 'jpeg-5': 0.6414
}
results_densenet121 = {
    'none': 0.668,
    'gaussian_noise-1': 0.6526, 'gaussian_noise-2': 0.5974, 'gaussian_noise-3': 0.5222, 'gaussian_noise-4': 0.4027, 'gaussian_noise-5': 0.2362,
    'gaussian_blur-1': 0.6719, 'gaussian_blur-2': 0.6543, 'gaussian_blur-3': 0.611, 'gaussian_blur-4': 0.5182, 'gaussian_blur-5': 0.4079,
    'contrast-1': 0.6195, 'contrast-2': 0.4985, 'contrast-3': 0.3965, 'contrast-4': 0.2992, 'contrast-5': 0.2187,
    'jpeg-1': 0.6646, 'jpeg-2': 0.6602, 'jpeg-3': 0.6522, 'jpeg-4': 0.6482, 'jpeg-5': 0.6028
}

legend_kwargs = {
    'loc': 'upper right',
    'bbox_to_anchor': (1.0, -0.05),
    'fancybox': True,
    'shadow': True,
    'ncol': 1
}


def compute_ce(relative: bool =False):
    model_data = [results_resnet18_epochs250, results_resnet34_epochs250, results_vgg11_bn_epochs250, results_vgg11_epochs250, results_densenet121]
    for c in ['gaussian_noise', 'gaussian_blur', 'contrast', 'jpeg']:
        c_points = []
        for data in model_data:
            denominator = 0
            nominator = 0
            for s in range(1, 6):
                nominator += (1 - data[f"{c}-{s}"])
                denominator += (1 - results_resnet18_epochs250[f"{c}-{s}"])
                if relative:
                    e_f_clean = (1 - data['none'])
                    e_resnet_clean = 1 - results_resnet18_epochs250['none']
                    nominator -= e_f_clean
                    denominator -= e_resnet_clean
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
    plt.title("Accuracy by corruption method")
    plt.plot(x_point, [
        results_resnet18_epochs250['gaussian_blur-1'],
        results_resnet18_epochs250['gaussian_blur-2'],
        results_resnet18_epochs250['gaussian_blur-3'],
        results_resnet18_epochs250['gaussian_blur-4'],
        results_resnet18_epochs250['gaussian_blur-5']
    ], label="Gaussian blur")
    plt.xticks(x_point)
    plt.plot(x_point, [
        results_resnet18_epochs250['gaussian_noise-1'],
        results_resnet18_epochs250['gaussian_noise-2'],
        results_resnet18_epochs250['gaussian_noise-3'],
        results_resnet18_epochs250['gaussian_noise-4'],
        results_resnet18_epochs250['gaussian_noise-5']
    ], label="Gaussian noise")
    plt.plot(x_point, [
        results_resnet18_epochs250['jpeg-1'],
        results_resnet18_epochs250['jpeg-2'],
        results_resnet18_epochs250['jpeg-3'],
        results_resnet18_epochs250['jpeg-4'],
        results_resnet18_epochs250['jpeg-5']
    ], label="JPEG compression loss")
    plt.plot(x_point, [
        results_resnet18_epochs250['contrast-1'],
        results_resnet18_epochs250['contrast-2'],
        results_resnet18_epochs250['contrast-3'],
        results_resnet18_epochs250['contrast-4'],
        results_resnet18_epochs250['contrast-5']
    ], label="Contrast")
    plt.legend(**legend_kwargs)
    plt.show()


graph_resnet18()