def getNumClass(dsname):
    if dsname == 'cifar10':
        return 10


def getParamGeneral():
    alpha = 0.0005  # reconstruction loss coeff
    n_epochs = 100
    # batch_size = ??? at the Train cell

    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5
    init_sigma = 0.1
    caps1_n_dims = 8
    caps2_n_dims = 16
    caps1_n_maps = 32

    primary_cap_size1 = 6
    primary_cap_size2 = 6

    n_hidden1 = 512
    n_hidden2 = 1024

    return alpha, n_epochs, m_plus, m_minus, lambda_, \
           init_sigma, caps1_n_dims, caps2_n_dims, caps1_n_maps, \
           primary_cap_size1, primary_cap_size2,n_hidden1,n_hidden2


def getParamCaps(dsname):

    alpha, n_epochs, m_plus, m_minus, lambda_, \
    init_sigma, caps1_n_dims, caps2_n_dims, caps1_n_maps, \
    primary_cap_size1, primary_cap_size2,n_hidden1,n_hidden2 = getParamGeneral()

    if dsname == 'feret_feature':
        num_class = 2
        image_size1 = 28
        image_size2 = 28
        num_image_channel = 128
        checkpoint_path = './checkpoints/FERET_FEATURE_CapsNet/'
        primary_cap_size1 = 6
        primary_cap_size2 = 6
    if dsname == 'mnist':
        num_class = 10
        image_size1 = 28
        image_size2 = 28
        num_image_channel = 1
        checkpoint_path = './checkpoints/MNIST_CapsNet/'
    if dsname == 'feret':
        num_class = 2
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 3
        checkpoint_path = './checkpoints/FERET_CapsNet/'
        primary_cap_size1 = 12
        primary_cap_size2 = 12
    if dsname == 'toy_man':
        num_class = 2
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 3
        checkpoint_path = './checkpoints/Toy_man_CapsNet/'
        primary_cap_size1 = 12
        primary_cap_size2 = 12
    if dsname == 'c-cube':
        num_class = 52
        image_size1 = 32
        image_size2 = 32
        num_image_channel = 1
        checkpoint_path = './checkpoints/C-CUBE_CapsNet/'
        primary_cap_size1 = 7
        primary_cap_size2 = 7
        lambda_ = 0.05
    if dsname == 'cifar10':
        num_class = 10
        image_size1 = 32
        image_size2 = 32
        num_image_channel = 3
        checkpoint_path = './checkpoints/CIFAR10_CapsNet/'
    if dsname == 'UTSig':
        num_class = 115
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 1
        checkpoint_path = './checkpoints/UTSig_CapsNet/'
        primary_cap_size1 = 12
        primary_cap_size2 = 12
        lambda_ = 0.02
    if dsname == 'MCYT75':
        num_class = 75
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 1
        checkpoint_path = './checkpoints/MCYT75_CapsNet/'
        primary_cap_size1 = 12
        primary_cap_size2 = 12
        lambda_ = 0.05
    if dsname == 'cedar':
        num_class = 55
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 1
        checkpoint_path = './checkpoints/CEDAR_CapsNet/'
        primary_cap_size1 = 12
        primary_cap_size2 = 12
        lambda_ = 0.02
    if dsname == 'SVHN':
        num_class = 10
        image_size1 = 32
        image_size2 = 32
        num_image_channel = 1
        checkpoint_path = './checkpoints/SVHN_CapsNet/'
        primary_cap_size1 = 6
        primary_cap_size2 = 6
        lambda_ = 0.5

    return num_class, image_size1, image_size2, num_image_channel, \
           checkpoint_path, alpha, n_epochs, m_plus, m_minus, lambda_, \
           init_sigma, caps1_n_dims, caps2_n_dims, caps1_n_maps, \
           primary_cap_size1, primary_cap_size2,n_hidden1,n_hidden2


def getParamCaps_Competitve(dsname):

    num_cluster_per_class = 5

    alpha, n_epochs, m_plus, m_minus, lambda_, \
    init_sigma, caps1_n_dims, caps2_n_dims, caps1_n_maps, \
    primary_cap_size1, primary_cap_size2,n_hidden1,n_hidden2 = getParamGeneral()

    if dsname == 'feret_feature':
        num_cluster_per_class=4
        num_class = 2
        image_size1 = 28
        image_size2 = 28
        num_image_channel = 128
        checkpoint_path = './checkpoints/FERET_FEATURE_CapsNet_Competitive/'
        primary_cap_size1 = 6
        primary_cap_size2 = 6
    if dsname == 'toy_man':
        num_cluster_per_class=3
        num_class = 2
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 3
        checkpoint_path = './checkpoints/Toyman_CapsNet_Competitive/'
        primary_cap_size1 = 6
        primary_cap_size2 = 6
        
    if dsname == 'mnist':
        num_cluster_per_class=6
        num_class = 10
        image_size1 = 28
        image_size2 = 28
        num_image_channel = 1
        checkpoint_path = './checkpoints/MNIST_CapsNet_Competitive/'
    if dsname == 'feret':
        num_cluster_per_class=4
        num_class = 2
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 3
        checkpoint_path = './checkpoints/FERET_CapsNet_Competitive/'
        primary_cap_size1 = 12
        primary_cap_size2 = 12
    if dsname == 'c-cube':
        num_class = 52
        image_size1 = 32
        image_size2 = 32
        num_image_channel = 1
        checkpoint_path = './checkpoints/C-CUBE_CapsNet_Competitive/'
        primary_cap_size1 = 7
        primary_cap_size2 = 7
        lambda_ = 0.05
        num_cluster_per_class = 4
        caps1_n_maps = 20

    if dsname == 'cifar10':
        num_class = 10
        image_size1 = 32
        image_size2 = 32
        num_image_channel = 3
        checkpoint_path = './checkpoints/CIFAR10_CapsNet_Competitive/'

    if dsname == 'UTSig':
        num_class = 115
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 1
        checkpoint_path = './checkpoints/UTSig_CapsNet_Competitive/'
        primary_cap_size1 = 12
        primary_cap_size2 = 12
        lambda_ = 0.02
        num_cluster_per_class = 3

    if dsname == 'MCYT75':
        num_class = 75
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 1
        checkpoint_path = './checkpoints/MCYT75_CapsNet_Competitive/'
        primary_cap_size1 = 12
        primary_cap_size2 = 12
        lambda_ = 0.05
        num_cluster_per_class = 2

    if dsname == 'cedar':
        num_class = 55
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 1
        checkpoint_path = './checkpoints/CEDAR_CapsNet_Competitive/'
        primary_cap_size1 = 12
        primary_cap_size2 = 12
        lambda_ = 0.05
        num_cluster_per_class = 2
        caps1_n_maps = 32

    if dsname == 'SVHN':
        num_class = 10
        image_size1 = 32
        image_size2 = 32
        num_image_channel = 1
        checkpoint_path = './checkpoints/SVHN_CapsNet_Competitive/'
        primary_cap_size1 = 6
        primary_cap_size2 = 6
        lambda_ = 0.5
        num_cluster_per_class = 6
        caps1_n_maps = 32

    return num_class, image_size1, image_size2, num_image_channel, \
           checkpoint_path, alpha, n_epochs, m_plus, m_minus, lambda_, \
           init_sigma, caps1_n_dims, caps2_n_dims, caps1_n_maps, \
           primary_cap_size1, primary_cap_size2,num_cluster_per_class


def getBatchSize(dsname):
    batchSize = 100

    if dsname == 'UTSig':
        batchSize = 2
    if dsname == 'MCYT75':
        batchSize = 2
    if dsname == 'cedar':
        batchSize = 5
    if dsname == 'mnist':
        batchSize = 20
    if dsname == 'c-cube':
        batchSize = 25
    if dsname == 'feret':
        batchSize = 25
    if dsname == 'toy_man':
        batchSize = 32


    return batchSize



def getParamCaps_Competitve_reduce(dsname):

    num_cluster_per_class = 5

    alpha, n_epochs, m_plus, m_minus, lambda_, \
    init_sigma, caps1_n_dims, caps2_n_dims, caps1_n_maps, \
    primary_cap_size1, primary_cap_size2,n_hidden1,n_hidden2 = getParamGeneral()

    if dsname == 'mnist':
        num_class = 10
        image_size1 = 28
        image_size2 = 28
        num_image_channel = 1
        checkpoint_path = './checkpoints/MNIST_CapsNet_Competitive/'
    if dsname == 'c-cube':
        num_class = 52
        image_size1 = 32
        image_size2 = 32
        num_image_channel = 1
        checkpoint_path = './checkpoints/C-CUBE_CapsNet_Competitive_reduce/'
        primary_cap_size1 = 7
        primary_cap_size2 = 7
        lambda_ = 0.05
        num_cluster_per_class = 4
        caps1_n_maps = 32 // num_cluster_per_class

    if dsname == 'cifar10':
        num_class = 10
        image_size1 = 32
        image_size2 = 32
        num_image_channel = 3
        checkpoint_path = './checkpoints/CIFAR10_CapsNet_Competitive/'

    if dsname == 'UTSig':
        num_class = 115
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 1
        checkpoint_path = './checkpoints/UTSig_CapsNet_Competitive/'
        primary_cap_size1 = 12
        primary_cap_size2 = 12
        lambda_ = 0.02
        num_cluster_per_class = 3

    if dsname == 'MCYT75':
        num_class = 75
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 1
        checkpoint_path = './checkpoints/MCYT75_CapsNet_Competitive/'
        primary_cap_size1 = 12
        primary_cap_size2 = 12
        lambda_ = 0.02
        num_cluster_per_class = 3
    if dsname == 'cedar':
        num_class = 55
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 1
        checkpoint_path = './checkpoints/CEDAR_CapsNet_Competitive/'
        primary_cap_size1 = 12
        primary_cap_size2 = 12
        lambda_ = 0.05
        num_cluster_per_class = 2
        caps1_n_maps = 32
    if dsname == 'toy_man':
        num_class = 2
        image_size1 = 64
        image_size2 = 64
        num_image_channel = 3
        checkpoint_path = './checkpoints/Toyman_CapsNet_Competitive/'
        primary_cap_size1 = 12
        primary_cap_size2 = 12
        lambda_ = 0.05
        num_cluster_per_class = 3
        caps1_n_maps = 32
    if dsname == 'SVHN':
        num_class = 10
        image_size1 = 32
        image_size2 = 32
        num_image_channel = 3
        checkpoint_path = './checkpoints/SVHN_CapsNet_Competitive_reduce/'
        primary_cap_size1 = 6
        primary_cap_size2 = 6
        lambda_ = 0.5
        num_cluster_per_class = 4
        caps1_n_maps = 32 // num_cluster_per_class

    return num_class, image_size1, image_size2, num_image_channel, \
           checkpoint_path, alpha, n_epochs, m_plus, m_minus, lambda_, \
           init_sigma, caps1_n_dims, caps2_n_dims, caps1_n_maps, \
           primary_cap_size1, primary_cap_size2,num_cluster_per_class
