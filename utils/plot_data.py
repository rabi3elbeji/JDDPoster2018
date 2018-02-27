###################################################################
####################  class for plotting data  ####################
###################################################################

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import itertools
import numpy as np
from sklearn.metrics import roc_curve, auc
import math


class PlotData(object):
    """docstring for PlotData"""

    def __init__(self):
        super(PlotData, self).__init__()
        # matplotlib.style.use('ggplot')

    def plot_2d(self, x, y, x_label, y_label, title, legend_arr, path_to_save):
        plt.clf()
        plt.figure()
        plt.plot(x)
        plt.plot(y)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.legend(legend_arr, loc='best')
        plt.title(title)
        plt.savefig(path_to_save)

    def plot_2d_2in1(self, acc, loss, val_acc, val_loss, x_label, y_label, title, legend_arr, path_to_save):
        plt.clf()
        plt.figure()

        font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}
        matplotlib.rc('font', **font)
        plt.plot(acc, color='blue',  linewidth=3.0)
        plt.plot(val_acc, color='deeppink', linewidth=3.0)
        plt.plot(loss, color='forestgreen', linewidth=3.0)
        plt.plot(val_loss, color='red', linewidth=3.0)
 
        #plt.plot(loss, val_loss)
        plt.ylabel(y_label, fontsize=15)
        plt.xlabel(x_label, fontsize=15)



        plt.legend(legend_arr, loc='best')
        plt.title(title)
        plt.savefig(path_to_save)


    def plot_2d_markers(self, data_x, data_y, x_label, y_label, markers, linestyles, colors, legend_arr, path_to_save):
        plt.clf()
        plt.figure()

        # Extract max values

        ymax0 = max(data_y[0])
        xpos0 = data_y[0].index(ymax0)
        xmax0 = data_x[xpos0]

        ymax1 = max(data_y[1])
        xpos1 = data_y[1].index(ymax1)
        xmax1 = data_x[xpos1]

        fig, ax1 = plt.subplots()

        color = 'tab:green'
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label, color=color)
        ax1.plot(data_x, data_y[0], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel(y_label, color=color)  # we already handled the x-label with ax1
        ax2.plot(data_x, data_y[1], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # Set Max Values
        #ax1.annotate(' ', xy=(xmax0, ymax0),                arrowprops=dict(facecolor='red', shrink=0.005) )


        '''
        for i in xrange(0, len(data_y)):

            

            plt.plot(data_x, data_y[i], color=colors[i],
                     linestyle=linestyles[0])

            plt.annotate(' ', xy=(xmax, ymax),
                         arrowprops=dict(facecolor='red', shrink=0.005)
                         )

            text_max = 'Lambd = '+str(xmax) + '\n' + 'MSE = '+str(ymax)
            plt.annotate(text_max, (xmax, ymax),
                (xmax+2.5, ymax-.01),
                #xycoords="figure fraction", textcoords="figure fraction",
                ha="right", va="center",
                size=8,
                arrowprops=dict(arrowstyle='fancy',
                                shrinkA=5,
                                shrinkB=5,
                                fc="k", ec="k",
                                connectionstyle="arc3,rad=-0.05",
                                ),
                bbox=dict(boxstyle="square", fc="w"))
        '''
    
      
        #plt.legend(legend_arr, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,  ncol=2, mode="expand", borderaxespad=0.)

        ax1.legend([legend_arr[0]], loc='best')
        ax2.legend([legend_arr[1]], loc='best')

 
        plt.savefig(path_to_save)


    def plot_2d_filters_images(self, data_x, data_y, x_label, y_label, markers, linestyles, colors, legend_arr, path_to_save):
        plt.clf()
        plt.figure()

        # Extract max values

        fig, ax1 = plt.subplots()
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label, color=colors[0])
        lns1 = ax1.scatter(data_x[1], data_y[1], color=colors[0], marker=markers[0])
        ax1.tick_params(axis='y', labelcolor=colors[0])


        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        

        ax2.set_xlabel(x_label)
        ax2.set_ylabel(y_label, color=colors[1])
        lns2 = ax2.scatter(data_x[0], data_y[0], color=colors[1], marker=markers[1])
        ax2.tick_params(axis='y', labelcolor=colors[1])

     
        # added these three lines

        ax1.legend([legend_arr[1]], bbox_to_anchor=(0., 1.01, 1., .102), loc=3,  ncol=2)
        ax2.legend([legend_arr[0]], bbox_to_anchor=(0., 1.01, 1., .102), loc=4,  ncol=2)
        
    
        ax1.axvline(1.41, linestyle='dashed', linewidth=1)
        ax1.axvline(0.2, linestyle='dashed', linewidth=1)
        ax2.axvline(7.47, linestyle='dashed', linewidth=1)
        ax2.axvline(6.26, linestyle='dashed', linewidth=1)
        ax2.axvline(6.77, linestyle='dashed', linewidth=1)

        #ax2.legend(legend_arr, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,  ncol=2, mode="expand", borderaxespad=0.)

        plt.savefig(path_to_save)


    def plot_model(self, model_details, path_to_save):
        # Create sub-plots
        plt.clf()
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # Summarize history for accuracy
        axs[0].plot(range(1, len(model_details.history['acc']) + 1),
                    model_details.history['acc'])
        axs[0].plot(range(1, len(model_details.history['val_acc']) + 1),
                    model_details.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1, len(model_details.history[
                          'acc']) + 1), len(model_details.history['acc']) / 10)
        axs[0].legend(['train', 'validation'], loc='best')

        # Summarize history for loss
        axs[1].plot(range(1, len(model_details.history['loss']) + 1),
                    model_details.history['loss'])
        axs[1].plot(range(1, len(model_details.history['val_loss']) + 1),
                    model_details.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1, len(model_details.history[
                          'loss']) + 1), len(model_details.history['loss']) / 10)
        axs[1].legend(['train', 'validation'], loc='best')

        # Save plot
        plt.savefig(path_to_save)

    def plot_model_bis(self, model_details, path_to_save):

        # Create sub-plots
        plt.clf()
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # Summarize history for accuracy
        axs[0].plot(range(1, len(model_details['acc']) + 1),
                    model_details['acc'])
        axs[0].plot(range(1, len(model_details['val_acc']) + 1),
                    model_details['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1, len(model_details[
                          'acc']) + 1), len(model_details['acc']) / 10)
        axs[0].legend(['train', 'validation'], loc='best')

        # Summarize history for loss
        axs[1].plot(range(1, len(model_details['loss']) + 1),
                    model_details['loss'])
        axs[1].plot(range(1, len(model_details['val_loss']) + 1),
                    model_details['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1, len(model_details[
                          'loss']) + 1), len(model_details['loss']) / 10)
        axs[1].legend(['train', 'validation'], loc='best')

        # Save plot
        plt.savefig(path_to_save)

        # This is not part of the code (is just for plotting custom data)
    def plot_detection_error(self, data_x, data_y, colors, linestyles, markers, legend, path_to_save):
        plt.clf()

        plt.style.use('classic')
        plt.plot(data_x, data_y[0], color=colors[0],
                 linestyle=linestyles[0], marker=markers[0])
        plt.plot(data_x, data_y[1], color=colors[1],
                 linestyle=linestyles[0], marker=markers[1])
        plt.plot(data_x, data_y[2], color=colors[2],
                 linestyle=linestyles[0], marker=markers[2])
        plt.ylabel(r'$P_{E}$')
        plt.xlabel('Payload (bpp)')
        plt.legend(legend, loc='best', prop={
                   'size': 10.2, 'weight': 'semibold', 'family': 'monospace'})
        plt.grid(True)
        plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                    0.7, 0.8, 0.9, 1.0, 1.1])
        # Save plot
        plt.savefig(path_to_save)

    # This is not part of the code (is just for plotting custom data)
    def plot_custom_data(self, model_details, colors, linestyles, markers, legend, path_to_save):

        # Create sub-plots
        plt.clf()
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        plt.style.use('classic')

        # Summarize history for accuracy
        # payload 0.7 bpp
        axs[0].plot(range(1, len(model_details['acc07']) + 1),
                    model_details['acc07'], color=colors[0], linestyle=linestyles[0], marker=markers[0])
        axs[0].plot(range(1, len(model_details['val_acc07']) + 1),
                    model_details['val_acc07'], color=colors[0], linestyle=linestyles[1], marker=markers[1])

        # payload 0.5 bpp
        axs[0].plot(range(1, len(model_details['acc05']) + 1),
                    model_details['acc05'], color=colors[1], linestyle=linestyles[0], marker=markers[0])
        axs[0].plot(range(1, len(model_details['val_acc05']) + 1),
                    model_details['val_acc05'], color=colors[1], linestyle=linestyles[1], marker=markers[1])

        # payload 0.3 bpp
        axs[0].plot(range(1, len(model_details['acc03']) + 1),
                    model_details['acc03'], color=colors[2], linestyle=linestyles[0], marker=markers[0])
        axs[0].plot(range(1, len(model_details['val_acc03']) + 1),
                    model_details['val_acc03'], color=colors[2], linestyle=linestyles[1], marker=markers[1])

        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')

        axs[0].legend(legend, loc='best', prop={
                      'size': 8.2, 'weight': 'semibold', 'family': 'monospace'})
        axs[0].grid(True)

        # Summarize history for loss
        # payload 0.7 bpp
        axs[1].plot(range(1, len(model_details['loss07']) + 1),
                    model_details['loss07'], color=colors[0], linestyle=linestyles[0], marker=markers[0])
        axs[1].plot(range(1, len(model_details['val_loss07']) + 1),
                    model_details['val_loss07'], color=colors[0], linestyle=linestyles[1], marker=markers[1])

        # payload 0.5 bpp
        axs[1].plot(range(1, len(model_details['loss05']) + 1),
                    model_details['loss05'], color=colors[1], linestyle=linestyles[0], marker=markers[0])
        axs[1].plot(range(1, len(model_details['val_loss05']) + 1),
                    model_details['val_loss05'], color=colors[1], linestyle=linestyles[1], marker=markers[1])

        # payload 0.3 bpp
        axs[1].plot(range(1, len(model_details['loss03']) + 1),
                    model_details['loss03'], color=colors[2], linestyle=linestyles[0], marker=markers[0])
        axs[1].plot(range(1, len(model_details['val_loss03']) + 1),
                    model_details['val_loss03'], color=colors[2], linestyle=linestyles[1], marker=markers[1])

        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')

        axs[1].legend(legend, loc='best', prop={
                      'size': 8.2, 'weight': 'semibold', 'family': 'monospace'})
        axs[1].grid(True)

        axs[0].set_xlim(0, 11)
        axs[1].set_xlim(0, 11)

        # Save plot
        plt.savefig(path_to_save)

    def plot_confusion_matrix(self, cm, classes,
                              path_to_save,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        # plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     verticalalignment='center',
                     fontsize=20,
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        ax.grid(False)
        plt.savefig(path_to_save, format='png')

    def plot_roc(self, y_true, y_scores, filename):
        '''
        Plot the ROC for this model.
        '''
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        print('ROC AUC:', roc_auc)
        plt.figure()
        plt.clf()
        plt.plot(fpr, tpr, color='darkorange',
                 label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="best")
        plt.savefig(filename)
        plt.close()
        return roc_auc

    def plot_roc(self, y_trues, y_scores, colors, linestyles, legend, filename):
        '''
        Plot the ROC for this model.
        '''

        plt.figure()
        plt.clf()

        fpr0, tpr0, thresholds = roc_curve(y_trues[0], y_scores[0])
        fpr1, tpr1, thresholds = roc_curve(y_trues[1], y_scores[1])
        fpr2, tpr2, thresholds = roc_curve(y_trues[2], y_scores[2])
        fpr3, tpr3, thresholds = roc_curve(y_trues[3], y_scores[3])

        fpr4, tpr4, thresholds = roc_curve(y_trues[4], y_scores[4])
        fpr5, tpr5, thresholds = roc_curve(y_trues[5], y_scores[5])
        fpr6, tpr6, thresholds = roc_curve(y_trues[6], y_scores[6])
        fpr7, tpr7, thresholds = roc_curve(y_trues[7], y_scores[7])

        fpr8, tpr8, thresholds = roc_curve(y_trues[8], y_scores[8])
        fpr9, tpr9, thresholds = roc_curve(y_trues[9], y_scores[9])
        fpr10, tpr10, thresholds = roc_curve(y_trues[10], y_scores[10])
        fpr11, tpr11, thresholds = roc_curve(y_trues[11], y_scores[11])

        plt.plot(fpr0, tpr0, color=colors[0], linestyle=linestyles[0])
        plt.plot(fpr1, tpr1, color=colors[0], linestyle=linestyles[1])
        plt.plot(fpr2, tpr2, color=colors[0], linestyle=linestyles[2])
        plt.plot(fpr3, tpr3, color=colors[0], linestyle=linestyles[3])

        plt.plot(fpr4, tpr4, color=colors[1], linestyle=linestyles[0])
        plt.plot(fpr5, tpr5, color=colors[1], linestyle=linestyles[1])
        plt.plot(fpr6, tpr6, color=colors[1], linestyle=linestyles[2])
        plt.plot(fpr7, tpr7, color=colors[1], linestyle=linestyles[3])

        plt.plot(fpr8, tpr8, color=colors[2], linestyle=linestyles[0])
        plt.plot(fpr9, tpr9, color=colors[2], linestyle=linestyles[1])
        plt.plot(fpr10, tpr10, color=colors[2], linestyle=linestyles[2])
        plt.plot(fpr11, tpr11, color=colors[2], linestyle=linestyles[3])

        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')

        plt.legend(legend, loc='best', prop={
                   'size': 8.2, 'weight': 'semibold', 'family': 'monospace'})

        plt.grid(True)
        plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                    0.7, 0.8, 0.9, 1.0])
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                    0.7, 0.8, 0.9, 1.0])
        plt.savefig(filename)
        plt.close()

    def plot_conv_weights(self, weights, filename, input_channel=0):
        plt.figure()
        plt.clf()
        # Get the lowest and highest values for the weights.
        # This is used to correct the colour intensity across
        # the images so they can be compared with each other.
        w_min = np.min(weights)
        w_max = np.max(weights)

        # Number of filters used in the conv. layer.
        num_filters = weights.shape[3]

        # Number of grids to plot.
        # Rounded-up, square-root of the number of filters.
        num_grids = math.ceil(math.sqrt(num_filters))

        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(int(num_grids), int(num_grids))

        # Plot all the filter-weights.
        for i, ax in enumerate(axes.flat):
            # Only plot the valid filter-weights.
            if i < num_filters:
                # Get the weights for the i'th filter of the input channel.
                # See new_conv_layer() for details on the format
                # of this 4-dim tensor.
                img = weights[:, :, input_channel, i]

                # Plot image.
                im = ax.imshow(img, vmin=w_min, vmax=w_max,
                               interpolation='nearest')

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Colorbar
        cax, kw = matplotlib.colorbar.make_axes([ax for ax in axes.flat])
        plt.colorbar(im, cax=cax, **kw)
        plt.savefig(filename)
        plt.close()

    def plot_conv_weights(self, weights, filename, input_channel=0):
        plt.figure()
        plt.clf()
        # Get the lowest and highest values for the weights.
        # This is used to correct the colour intensity across
        # the images so they can be compared with each other.
        w_min = np.min(weights)
        w_max = np.max(weights)

        # Number of filters used in the conv. layer.
        num_filters = weights.shape[3]

        # Number of grids to plot.
        # Rounded-up, square-root of the number of filters.
        num_grids = math.ceil(math.sqrt(num_filters))

        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(int(num_grids), int(num_grids))

        # Plot all the filter-weights.
        for i, ax in enumerate(axes.flat):
            # Only plot the valid filter-weights.
            if i < num_filters:
                # Get the weights for the i'th filter of the input channel.
                # See new_conv_layer() for details on the format
                # of this 4-dim tensor.
                img = weights[:, :, input_channel, i]

                # Plot image.
                im = ax.imshow(img, vmin=w_min, vmax=w_max,
                               interpolation='nearest')

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Colorbar
        cax, kw = matplotlib.colorbar.make_axes([ax for ax in axes.flat])
        plt.colorbar(im, cax=cax, **kw)
        plt.savefig(filename)
        plt.close()

    def plot_conv_weights_8_4(self, weights, filename, input_channel=0):
        plt.figure()
        plt.clf()
        # Get the lowest and highest values for the weights.
        # This is used to correct the colour intensity across
        # the images so they can be compared with each other.
        w_min = np.min(weights)
        w_max = np.max(weights)

        # Number of filters used in the conv. layer.
        num_filters = weights.shape[3]

        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(1, 4)

        # Plot all the filter-weights.
        for i, ax in enumerate(axes.flat):
            # Only plot the valid filter-weights.
            if i < num_filters:
                # Get the weights for the i'th filter of the input channel.
                # See new_conv_layer() for details on the format
                # of this 4-dim tensor.
                img = weights[:, :, input_channel, i]

                # Plot image.
                im = ax.imshow(img, vmin=w_min, vmax=w_max,
                               cmap=plt.cm.jet,
                               interpolation='nearest')

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Colorbar
        cax, kw = matplotlib.colorbar.make_axes([ax for ax in axes.flat])
        plt.colorbar(im, cax=cax, **kw)
        plt.savefig(filename)
        plt.close()

    def plot_conv_weights_3_4(self, image, weights, outputs, filename, input_channel=0):

        plt.clf()

        # Get the lowest and highest values for the weights.
        # This is used to correct the colour intensity across
        # the images so they can be compared with each other.
        w_min = np.min(weights)
        w_max = np.max(weights)

        # Number of filters used in the conv. layer.
        num_filters = weights.shape[3]

        outputs_images = []
        outputs_images.append(image)
        outputs_images.append(image)
        outputs_images.append(image)
        outputs_images.append(image)

        for i in xrange(4):
            plt.subplot(3, 4, i + 1), plt.imshow(outputs_images[i], 'gray',
                                                 interpolation='nearest')
            plt.xticks([]), plt.yticks([])

        for i in xrange(4):
            img = weights[:, :, input_channel, i]
            plt.subplot(3, 4, i + 5), plt.imshow(img,
                                                 interpolation='nearest', cmap=plt.cm.bone)
            plt.xticks([]), plt.yticks([])

        for i in xrange(4):
            img = outputs[0, :, :, i]
            plt.subplot(3, 4, i + 9), plt.imshow(img,
                                                 interpolation='nearest', cmap=plt.cm.bone)
            plt.xticks([]), plt.yticks([])

        plt.savefig(filename)
        plt.close()

    def plot_conv_weights(self, weights, filename, input_channel=0):
        plt.figure()
        plt.clf()
        # Get the lowest and highest values for the weights.
        # This is used to correct the colour intensity across
        # the images so they can be compared with each other.
        w_min = np.min(weights)
        w_max = np.max(weights)

        # Number of filters used in the conv. layer.
        num_filters = weights.shape[3]

        # Number of grids to plot.
        # Rounded-up, square-root of the number of filters.
        num_grids = math.ceil(math.sqrt(num_filters))

        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(int(num_grids), int(num_grids))

        # Plot all the filter-weights.
        for i, ax in enumerate(axes.flat):
            # Only plot the valid filter-weights.
            if i < num_filters:
                # Get the weights for the i'th filter of the input channel.
                # See new_conv_layer() for details on the format
                # of this 4-dim tensor.
                img = weights[:, :, input_channel, i]

                # Plot image.
                im = ax.imshow(img, vmin=w_min, vmax=w_max,
                               cmap=plt.cm.jet,
                               interpolation='nearest')

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Colorbar
        cax, kw = matplotlib.colorbar.make_axes([ax for ax in axes.flat])
        plt.colorbar(im, cax=cax, **kw)
        plt.savefig(filename)
        plt.close()

    def plot_2d_array(self, arr, filename):
        plt.figure()
        plt.clf()
        plt.imshow(arr, cmap=plt.cm.bone)
        plt.axis('off')
        plt.colorbar(orientation='vertical')
        plt.savefig(filename)
        plt.close()

    def plot_conv_output(self, values, filename):
        # Number of filters used in the conv. layer.
        num_filters = values.shape[3]

        # Number of grids to plot.
        # Rounded-up, square-root of the number of filters.
        num_grids = math.ceil(math.sqrt(num_filters))

        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(int(num_grids), int(num_grids))

        # Plot the output images of all the filters.
        for i, ax in enumerate(axes.flat):
            # Only plot the images for valid filters.
            if i < num_filters:
                # Get the output image of using the i'th filter.
                img = values[0, :, :, i]

                # Plot image.
                im = ax.imshow(img, interpolation='nearest', cmap=plt.cm.jet)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        cax, kw = matplotlib.colorbar.make_axes([ax for ax in axes.flat])
        plt.colorbar(im, cax=cax, **kw)
        plt.savefig(filename)
        plt.close()

    def plot_conv_output_8_4(self, values, filename):
        # Number of filters used in the conv. layer.
        num_filters = values.shape[3]

        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(8, 4)

        # Plot the output images of all the filters.
        for i, ax in enumerate(axes.flat):
            # Only plot the images for valid filters.
            if i < num_filters:
                # Get the output image of using the i'th filter.
                img = values[0, :, :, i]

                # Plot image.
                im = ax.imshow(img, interpolation='nearest', cmap=plt.cm.jet)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        cax, kw = matplotlib.colorbar.make_axes([ax for ax in axes.flat])
        plt.colorbar(im, cax=cax, **kw)
        plt.savefig(filename)
        plt.close()

    def plot_conv_output_1_4(self, values, filename):
        # Number of filters used in the conv. layer.
        num_filters = values.shape[3]

        # Create figure with a grid of sub-plots.
        fig, axes = plt.subplots(1, 4)

        # Plot the output images of all the filters.
        for i, ax in enumerate(axes.flat):
            # Only plot the images for valid filters.
            if i < num_filters:
                # Get the output image of using the i'th filter.
                img = values[0, :, :, i]

                # Plot image.
                im = ax.imshow(img, interpolation='nearest', cmap=plt.cm.bone)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        cax, kw = matplotlib.colorbar.make_axes([ax for ax in axes.flat])
        plt.colorbar(im, cax=cax, **kw)
        plt.savefig(filename)
        plt.close()
