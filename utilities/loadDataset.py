import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def rg2gray(rgb):
    return np.dot(rgb[...,:3],[0.2989,0.5870,0.1140])

def loadDataset(dsname):
    if dsname == 'feret_feature':
        img_size = 28
        class_no = 2
        num_male = 822
        num_female = 654
        n_channel = 128
        
        total_img = num_male + num_female
        
        Train = np.zeros(shape=(total_img, img_size, img_size,n_channel))
        Train_label = np.zeros(shape=(total_img))
        list = os.listdir('/content/drive/MyDrive/Checkpoints/FERET/Male/')

        cnt = 0
        for p in list:
            path = '/content/drive/MyDrive/Checkpoints/FERET/Male/'
            path = os.path.join(path, p)
            with open(path, 'rb') as f:
                Train[cnt,:,:,:] = pickle.load(f)   
            Train_label[cnt] = 0
            cnt = cnt + 1
            
        list = os.listdir('/content/drive/MyDrive/Checkpoints/FERET/Female/')
        for p in list:
            path = '/content/drive/MyDrive/Checkpoints/FERET/Female/'
            path = os.path.join(path, p)
            with open(path, 'rb') as f:
                Train[cnt,:,:,:] = pickle.load(f)   
            Train_label[cnt] = 1
            cnt = cnt + 1
            
        from sklearn.model_selection import train_test_split
        Train, Test, Train_label, Test_label = train_test_split(Train, Train_label, test_size=0.33, random_state=42)
    if dsname == 'cifar10':
        from utilities.load_cifar10 import load_CIFAR10

        cifar10_dir = './dataset/cifar10/cifar-10-batches-py'
        Train, Train_label, Test, Test_label = load_CIFAR10(cifar10_dir)

        Train_label = Train_label.reshape(-1)
        Test_label = Test_label.reshape(-1)

        Train = (255.0 - Train) / 255.0
        Test = (255.0 - Test) / 255.0
        # print(Train.max())
        # print(Train.min())
        # print(Train.mean())

    if dsname == 'mnist':
        import scipy.io
        tr_feature = scipy.io.loadmat('./dataset/mnist/tr_feature.mat')
        tr_feature = tr_feature['tr_feature']
        Train = np.zeros(shape=(60000, 28, 28))
        for i in range(60000):
            x = (np.array(tr_feature[:, i]))
            x = np.reshape(x, (1, 28, 28))
            Train[i, :] = x
        Train_label = scipy.io.loadmat('./dataset/mnist/tr_label.mat')
        Train_label = Train_label['tr_label'].reshape(-1)
        te_feature = scipy.io.loadmat('./dataset/mnist/te_feature.mat')
        te_feature = te_feature['te_feature']
        Test = np.zeros(shape=(10000, 28, 28))
        for i in range(10000):
            x = (te_feature[:, i]).reshape((1, 28, 28))
            Test[i, :] = x
        Test_label = scipy.io.loadmat('./dataset/mnist/te_label.mat')
        Test_label = Test_label['te_label'].reshape(-1)

    if dsname == 'c-cube':
        import scipy.io
        f = scipy.io.loadmat('./dataset/c-cube/c-cube.mat')
        tr_feature = f['Xtr']
        Train = np.zeros(shape=(38160, 32, 32))
        for i in range(38160):
            x = (np.array(tr_feature[:, i]))
            x = np.reshape(x, (1, 32, 32))
            Train[i, :] = x
        Train_label = f['Ytr'].reshape(-1)
        te_feature = f['Xte']
        Test = np.zeros(shape=(19133, 32, 32))
        for i in range(19133):
            x = (te_feature[:, i]).reshape((1, 32, 32))
            Test[i, :] = x
        Test_label = f['Yte'].reshape(-1)
        print(Train.max())
        print(Train.min())
        print(Train.mean())

    if dsname == 'UTSig':
        img_size = 64
        class_no = 115
        img_per_class = 27
        img_per_class_train = 20
        img_per_class_test = 7
        total_img = 3105
        if False:
            Train = np.zeros(shape=(class_no * img_per_class_train, img_size, img_size))
            Test = np.zeros(shape=(class_no * img_per_class_test, img_size, img_size))
            Train_label = np.zeros(shape=(class_no * img_per_class_train))
            Test_label = np.zeros(shape=(class_no * img_per_class_test))
            Tr_cnt = 0
            Ts_cnt = 0
            for i in range(total_img):
                path = './dataset/UTSig/UTSig64/C'
                if i // img_per_class + 1 < 10:
                    path = path + '00'
                elif i // img_per_class + 1 < 100:
                    path = path + '0'

                path = path + str(i // img_per_class + 1)
                path += 'G'

                if (i % img_per_class) + 1 < 10:
                    path += '0'

                path += str((i % img_per_class) + 1) + '.PNG'

                I = plt.imread(path)
                I = np.reshape(I, (1, img_size, img_size))
                if i % img_per_class < img_per_class_test:
                    Test[Ts_cnt, :, :] = I
                    Test_label[Ts_cnt] = i // img_per_class
                    Ts_cnt = Ts_cnt + 1
                else:
                    Train[Tr_cnt, :, :] = I
                    Train_label[Tr_cnt] = i // img_per_class
                    Tr_cnt = Tr_cnt + 1

                if i % 100 == 1:
                    print('image ' + str(i) + " from 3105")

            Train = 1 - Train
            Test = 1 - Test

            Train[Train > 0.15] = 1
            Train[Train <= 0.15] = 0
            Test[Test > 0.15] = 1
            Test[Test <= 0.15] = 0
            with open('./dataset/UTSig/DS_20_7', 'wb') as f:
                pickle.dump([Train, Train_label, Test, Test_label], f)

        else:
            with open('./dataset/UTSig/DS_20_7', 'rb') as f:
                Train, Train_label, Test, Test_label = pickle.load(f)
            print(Train.max())
            print(Train.min())
            print(Train.mean())
            print('Dataset loaded from file')
    if dsname == 'MCYT75':
        img_size = 64
        class_no = 75
        img_per_class = 15
        img_per_class_train = 11
        img_per_class_test = 4
        total_img = 1125
        if False:
            Train = np.zeros(shape=(class_no * img_per_class_train, img_size, img_size))
            Test = np.zeros(shape=(class_no * img_per_class_test, img_size, img_size))
            Train_label = np.zeros(shape=(class_no * img_per_class_train))
            Test_label = np.zeros(shape=(class_no * img_per_class_test))
            Tr_cnt = 0
            Ts_cnt = 0
            for i in range(total_img):
                path = './dataset/MCYT/MCYT75/'
                path = path + str(i // img_per_class + 1)
                path += 'v'
                path += str(i % img_per_class) + '.bmp'

                I = plt.imread(path)
                I = np.reshape(I, (1, img_size, img_size))
                if (i % img_per_class) < img_per_class_train:
                    Train[Tr_cnt, :, :] = I
                    Train_label[Tr_cnt] = i // img_per_class
                    Tr_cnt = Tr_cnt + 1
                else:
                    Test[Ts_cnt, :, :] = I
                    Test_label[Ts_cnt] = i // img_per_class
                    Ts_cnt = Ts_cnt + 1

                if i % 100 == 1:
                    print('image ' + str(i) + " from "+str(total_img))

            Train = (255.0 - Train) / 255.0
            Test = (255.0 - Test) / 255.0

            # print(Train.max())
            # print(Train.min())
            # print(Train.mean())

            Train[Train > 0.15] = 1
            Train[Train <= 0.15] = 0
            Test[Test > 0.15] = 1
            Test[Test <= 0.15] = 0
            with open('./dataset/MCYT/DS_11_4', 'wb') as f:  # 11 train and 4 test
                pickle.dump([Train, Train_label, Test, Test_label], f)

        else:
            with open('./dataset/MCYT/DS_11_4', 'rb') as f:
                Train, Train_label, Test, Test_label = pickle.load(f)
            print('Dataset loaded from file')
    if dsname == 'feret':
        from PIL import Image
        img_size = 64
        class_no = 2
        num_male = 822
        num_female = 654
        
        total_img = num_male + num_female
        
        Train = np.zeros(shape=(total_img, img_size, img_size,3))
        Train_label = np.zeros(shape=(total_img))
        list = os.listdir('./dataset/feret/Male/')

        cnt = 0
        for p in list:
            path = './dataset/feret/Male/'
            path = os.path.join(path, p)

            I = Image.open(path)
            I.thumbnail((img_size, img_size), Image.ANTIALIAS)
            I = np.reshape(I, (1, img_size, img_size, 3))
            Train[cnt, :, :,:] = I
            Train_label[cnt] = 1
            cnt = cnt + 1
            
        list = os.listdir('./dataset/feret/Female/')    
        for p in list:
            path = './dataset/feret/Female/'
            path = os.path.join(path, p)

            I = Image.open(path)
            I.thumbnail((img_size, img_size), Image.ANTIALIAS)
            I = np.reshape(I, (1, img_size, img_size, 3))
            Train[cnt, :, :,:] = I
            Train_label[cnt] = 2
            cnt = cnt + 1


#         Train = (255.0 - Train) / 255.0
        from sklearn.model_selection import train_test_split
        Train, Test, Train_label, Test_label = train_test_split(Train, Train_label, test_size=0.33, random_state=42)
        
        Train = (Train) / 255.0
        Test = (Test) / 255.0
        Train_label -= 1
        Test_label -= 1
#         print(Train.max())
#         print(Train.min())
#         print(Train.mean())
    if dsname == 'toy_man':
        from PIL import Image
        img_size = 64
        class_no = 2
        num_per_class = 600
        total_img = num_per_class*2
        Train = np.zeros(shape=(total_img, img_size, img_size,3))
        Train_label = np.zeros(shape=(total_img))
        list = os.listdir('./dataset/toy_man/rectangle_man/')
        cnt = 0
        for p in list:
            path = './dataset/toy_man/rectangle_man/'
            path = os.path.join(path, p)

            I = Image.open(path)
            I.thumbnail((img_size, img_size), Image.ANTIALIAS)
            I = np.reshape(I, (1, img_size, img_size, 3))
            Train[cnt, :, :,:] = I
            Train_label[cnt] = 1
            cnt = cnt + 1
            
        list = os.listdir('./dataset/toy_man/circle_man/')    
        for p in list:
            path = './dataset/toy_man/circle_man/'
            path = os.path.join(path, p)

            I = Image.open(path)
            I.thumbnail((img_size, img_size), Image.ANTIALIAS)
            I = np.reshape(I, (1, img_size, img_size, 3))
            Train[cnt, :, :,:] = I
            Train_label[cnt] = 2
            cnt = cnt + 1
            
        Train = (255.0 - Train) / 255.0
        from sklearn.model_selection import train_test_split
        Train, Test, Train_label, Test_label = train_test_split(Train, Train_label, test_size=0.33, random_state=42)
        Train_label -= 1
        Test_label -= 1
    if dsname == 'toy_man_original_size':
        from PIL import Image
        img_size = 300
        class_no = 2
        num_per_class = 600
        total_img = num_per_class*2
        Train = np.zeros(shape=(total_img, img_size, img_size,3))
        Train_label = np.zeros(shape=(total_img))
        list = os.listdir('./dataset/toy_man/rectangle_man/')
        cnt = 0
        for p in list:
            path = './dataset/toy_man/rectangle_man/'
            path = os.path.join(path, p)

            I = Image.open(path)
#             I.thumbnail((img_size, img_size), Image.ANTIALIAS)
            I = np.reshape(I, (1, img_size, img_size, 3))
            Train[cnt, :, :,:] = I
            Train_label[cnt] = 1
            cnt = cnt + 1
            
        list = os.listdir('./dataset/toy_man/circle_man/')    
        for p in list:
            path = './dataset/toy_man/circle_man/'
            path = os.path.join(path, p)

            I = Image.open(path)
#             I.thumbnail((img_size, img_size), Image.ANTIALIAS)
            I = np.reshape(I, (1, img_size, img_size, 3))
            Train[cnt, :, :,:] = I
            Train_label[cnt] = 2
            cnt = cnt + 1
            
#         Train = (255.0 - Train) / 255.0
        from sklearn.model_selection import train_test_split
        Train, Test, Train_label, Test_label = train_test_split(Train, Train_label, test_size=0.33, random_state=42)
        Train_label -= 1
        Test_label -= 1

    if dsname == 'cedar':
        image_size = 64
        if False:
            from PIL import Image
            Train = np.zeros(shape=(880, image_size, image_size))
            for i in range(880):
                path = './dataset/cedar/cedar_train/'
                path = path + str(i + 1) + '.png'
                # I = plt.imread(path)
                I = Image.open(path)
                I.thumbnail((image_size, image_size), Image.ANTIALIAS)
                I = np.reshape(I, (1, image_size, image_size))
                Train[i, :, :] = I
                if i % 100 == 1:
                    print('train image' + str(i) + "from 880")

            Test = np.zeros(shape=(440, image_size, image_size))
            for i in range(440):
                path = './dataset/cedar/cedar_test/'
                path = path + str(i + 1) + '.png'
                I = plt.imread(path)
                I = Image.open(path)
                I.thumbnail((image_size, image_size), Image.ANTIALIAS)
                I = np.reshape(I, (1, image_size, image_size))
                Test[i, :, :] = I
                if i % 100 == 1:
                    print('test image' + str(i) + "from 440")

            Train = (255.0 - Train) / 255.0
            Test = (255.0 - Test) / 255.0

            print(Train.max())
            print(Train.min())
            print(Train.mean())

            Train[Train > 0.2] = 1
            Train[Train <= 0.2] = 0
            Test[Test > 0.2] = 1
            Test[Test <= 0.2] = 0

            with open('./dataset/cedar/DS_880_440', 'wb') as f:
                pickle.dump([Train, Test], f)
        else:
            with open('./dataset/cedar/DS_880_440', 'rb') as f:
                Train, Test = pickle.load(f)

        Train_label = pd.read_csv('./dataset/cedar/cedarTrain_label.csv', ',',
                                  header=None).values
        Train_label = Train_label[:, 0] - 1

        Test_label = pd.read_csv('./dataset/cedar/cedarTest_label.csv', ',',
                                 header=None).values
        Test_label = Test_label[:, 0] - 1

    if dsname == 'SVHN':
        import scipy.io
        datatr = scipy.io.loadmat('./dataset/SVHN/train_32x32.mat')
        tr_feature = datatr['X']
        numChannel = 1
        Train = np.zeros(shape=(73257, 32, 32, numChannel))
        for i in range(73257):
            x = (np.array(tr_feature[:, :, :, i]))
            x = rg2gray(x)
            x = np.reshape(x, (1, 32, 32, numChannel))
            Train[i, :] = x
        Train_label = datatr['y'].reshape(-1)
        datats = scipy.io.loadmat('./dataset/SVHN/test_32x32.mat')
        te_feature = datats['X']
        Test = np.zeros(shape=(26032, 32, 32, numChannel))
        for i in range(26032):
            x = (te_feature[:,:,:, i]).reshape((1, 32, 32, 3))
            x = rg2gray(x)
            x = np.reshape(x, (1, 32, 32, numChannel))
            Test[i, :] = x
        Test_label = datats['y'].reshape(-1)

        Train = Train / 255.0
        Test = Test / 255.0

        Train_label -= 1
        Test_label -= 1

    return Train, Train_label, Test, Test_label
