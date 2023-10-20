from sklearn.metrics import roc_curve, auc

n_classes = len(np.unique(y_test))
alphas = [0.9, 0.8] * 7
linestyles = ['-.', '-'] * 7
linewidthss = [2, 1] * 7

def plot_roc_curves(dummy_y_test, predictions_test_numpy, n_classes, linestyles, linewidthss, label_for_auc):
    roc_data = {}
    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(dummy_y_test[:, i], predictions_test_numpy[:, i], drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        roc_data[i] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
        plt.plot(fpr, tpr, linestyle=linestyles[i], linewidth=linewidthss[i], label=f"({label_for_auc} AUC = {roc_auc:0.3f} for class {i}")

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random guess')
    plt.title("ROC curve", fontsize=16)
    plt.legend(loc='lower right', fontsize=16)
    ax.tick_params(axis='both', which='major', length=10, direction='in', labelsize=18, zorder=4)
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)

    plt.show()
    return roc_data
