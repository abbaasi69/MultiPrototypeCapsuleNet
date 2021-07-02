import pickle
import utilities.params as par
import matplotlib.pyplot as plt
import numpy as np

dsname = 'mnist'


# num_class, image_size1, image_size2, num_image_channel, \
# checkpoint_path, alpha, n_epochs, m_plus, m_minus, lambda_, \
# init_sigma, caps1_n_dims, caps2_n_dims, caps1_n_maps, \
# primary_cap_size1, primary_cap_size2, num_cluster_per_class\
#     = par.getParamCaps_Competitve(dsname)
# checkpoint_path = checkpoint_path[0:len(checkpoint_path)-1]+"_CCE_NSM/"


num_class, image_size1, image_size2, num_image_channel, \
           checkpoint_path, alpha, n_epochs, m_plus, m_minus, lambda_, \
           init_sigma, caps1_n_dims, caps2_n_dims, caps1_n_maps, \
           primary_cap_size1, primary_cap_size2,n_hidden1,n_hidden2\
    = par.getParamCaps(dsname)


# with open(''+checkpoint_path+'OtherVars', 'rb') as f:
#     acc_plot, loss_train_plot, loss_val_plot, time_per_epochs, start_epoch,y_raws = pickle.load(f)

with open('' + checkpoint_path + 'OtherVars', 'rb') as f:
    acc_plot, loss_train_plot, loss_val_plot, time_per_epochs, start_epoch = pickle.load(f)

# print(str(np.unique(y_raws)))
# plt.hist(np.reshape(y_raws,[-1]),num_cluster_per_class*num_class)
# plt.show()

plt.plot(acc_plot)
plt.show()

plt.plot(loss_train_plot)
plt.title('train/test loss')
plt.plot(loss_val_plot)
plt.show()

print(np.max(acc_plot))