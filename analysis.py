
import numpy as np
feature_dim = 3969
test_data = np.load('./features/scattering_train.npz')
features = test_data['arr_0']
labels = test_data['arr_1']
print("SHAPE: ", features.shape, labels.shape)
mu_G = np.mean(features, 0)  #global mean
# print("global_mean = ", mu_G)
print("global_mean shape = ", mu_G.shape)  #global_mean shape =  (3969,)


num_samples = features.shape[0]  #10000

mu_c_dict = {}
len_dict = {}
for feature, label in zip(features, labels):
    # print(feature, label)
    if label in mu_c_dict:
        mu_c_dict[label] += feature
        len_dict[label] += 1
    else:
        mu_c_dict[label] = feature
        len_dict[label] = 1
# print(mu_c_dict[0])
mu_c_array = []
for i in range(10):
    mu_c_dict[i] /= len_dict[i]  # average of samples
    mu_c_array.append(mu_c_dict[i])
# print(mu_c_dict[0])
mu_c_nparray = np.array(mu_c_array)


# right way for sigma_T
sigma_T = np.zeros((feature_dim,feature_dim))
for i in range(num_samples):
    sigma_T += np.dot((features[i,:] - mu_G).reshape(-1,1), (features[i,:] - mu_G).reshape(-1,1).T)
    if i %10000 ==0:
        print(i)
sigma_T /= num_samples
print("sigma_T.shape = ", sigma_T.shape)
# print('flag2: ', sigma_T[0])



sigma_B = np.zeros((feature_dim, feature_dim))
for i in range(10):
    sigma_B += np.dot((mu_c_dict[i] - mu_G).reshape(-1,1), (mu_c_dict[i] - mu_G).reshape(-1,1).T)
sigma_B /= num_samples
print("sigma_B2.shape = ", sigma_B.shape)
# print("sigma_B2 = ", sigma_B)

#%%

sigma_W = np.zeros((feature_dim, feature_dim))
for i in range(num_samples):
    # sigma_W += np.outer(features[i,:] - mu_c_dict[labels[i]], features[i,:] - mu_c_dict[labels[i]]) / num_samples
    sigma_W = np.dot((features[i,:] - mu_c_dict[labels[i]]).reshape(-1,1), (features[i,:] - mu_c_dict[labels[i]]).reshape(-1,1).T)
    if i %10000 ==0:
        print(i)
sigma_W /= num_samples
print("sigma_W.shape = ", sigma_W.shape)
# print("sigma_W = ", sigma_W)

NC1 = np.trace(sigma_W @ np.linalg.inv(sigma_B)) / 10
print("NC1 = ", NC1)
NC1 = np.trace(np.dot(sigma_W, np.linalg.inv(sigma_B))) / 10
print("NC1 = ", NC1)

#%%

CEC = np.std(np.linalg.norm(mu_c_nparray - mu_G, axis=1)) / np.mean(np.linalg.norm(mu_c_nparray - mu_G, axis=1))
print("CEC = ", CEC)
cos = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        cos[i, j] = np.dot(mu_c_nparray[i, :] - mu_G, mu_c_nparray[j, :] - mu_G) / np.linalg.norm(mu_c_nparray[i, :] - mu_G) / np.linalg.norm(mu_c_nparray[j, :] - mu_G)
#print("cos = ", cos)
EA = np.std(cos, axis=1)
print("EA = ", EA)
CME = np.mean(np.fabs(cos + 1 / (10 - 1)))
print("CME = ", CME)
# import numpy as np
# test_data = np.load('./features/test.npz')
# features = test_data['arr_0']
# labels = test_data['arr_1']
# print("SHAPE: ", features.shape, labels.shape)
# mu_G = np.mean(features, 0)  #global mean
# print("global_mean = ", mu_G)
# print("global_mean shape = ", mu_G.shape)  #global_mean shape =  (3969,)
#
# num_samples = features.shape[0]  #10000
#
# mu_c_dict = {}
# len_dict = {}
# for feature, label in zip(features, labels):
#     # print(feature, label)
#     if label in mu_c_dict:
#         mu_c_dict[label] += feature
#         len_dict[label] += 1
#     else:
#         mu_c_dict[label] = feature
#         len_dict[label] = 1
# # print(mu_c_dict[0])
# mu_c_array = []
# for i in range(10):
#     mu_c_dict[i] /= len_dict[i]  # average of samples
#     mu_c_array.append(mu_c_dict[i])
# # print(mu_c_dict[0])
# mu_c_nparray = np.array(mu_c_array)
#
# # wrong way for sigma_T
# # sigma_T_hidden = features - mu_G  #(10000, 3969) broadcast automatically    total covariance matrix
# # # print("sigma_T_hidden", sigma_T_hidden)
# # sigma_T = np.dot(sigma_T_hidden, sigma_T_hidden.transpose(1,0))
# # sigma_T = sigma_T.mean(0)  # sigma_T.shape =  (3969,)
# # print("sigma_T.shape = ", sigma_T.shape)
# # print('flag1: ', sigma_T[0])
#
# # right way for sigma_T
# sigma_T = np.zeros((3969,3969))
# for i in range(num_samples):
#     sigma_T += np.dot((features[i,:] - mu_G).reshape(-1,1), (features[i,:] - mu_G).reshape(-1,1).T)
#     if i %1000 ==0:
#         print(i)
# sigma_T /= num_samples
# print("sigma_T.shape = ", sigma_T.shape)
# print('flag2: ', sigma_T[0])
#
# # wrong way to calculat simg_B
# # sigma_B_hidden = mu_c_nparray - mu_G  #
# # sigma_B = np.dot(sigma_B_hidden, sigma_B_hidden.transpose(1,0))
# # # sigma_B = sigma_B.mean(0)  # sigma_T.shape =  (3969,)
# # print("sigma_B1.shape = ", sigma_B.shape)
# # print("sigma_B1 = ", sigma_B)
# #
#
# #the right way to calculate sigma_B
# sigma_B = np.zeros((3969, 3969))
# for i in range(10):
#     sigma_B += np.dot((mu_c_dict[i] - mu_G).reshape(-1,1), (mu_c_dict[i] - mu_G).reshape(-1,1).T)
# sigma_B /= num_samples
# print("sigma_B2.shape = ", sigma_B.shape)
# print("sigma_B2 = ", sigma_B)
#
#
# sigma_W = np.zeros((3969, 3969))
# for i in range(num_samples):
#     # sigma_W += np.outer(features[i,:] - mu_c_dict[labels[i]], features[i,:] - mu_c_dict[labels[i]]) / num_samples
#     sigma_W = np.dot((features[i,:] - mu_c_dict[labels[i]]).reshape(-1,1), (features[i,:] - mu_c_dict[labels[i]]).reshape(-1,1).T)
# sigma_W /= num_samples
# print("sigma_W.shape = ", sigma_W.shape)
# print("sigma_W = ", sigma_W)
#
# NC1 = np.trace(sigma_W @ np.linalg.inv(sigma_B)) / 10
# print("NC1 = ", NC1)
#
# CEC = np.std(np.linalg.norm(mu_c_nparray - mu_G, axis=1)) / np.mean(np.linalg.norm(mu_c_nparray - mu_G, axis=1))
# print("CEC = ", CEC)
#
# cos = np.zeros((10, 10))
# for i in range(10):
#     for j in range(10):
#         cos[i, j] = np.dot(mu_c_nparray[i, :] - mu_G, mu_c_nparray[j, :] - mu_G) / np.linalg.norm(mu_c_nparray[i, :] - mu_G) / np.linalg.norm(mu_c_nparray[j, :] - mu_G)
# print("cos = ", cos)
#
# EA = np.std(cos, axis=1)
# print("EA = ", EA)
#
# CME = np.mean(np.fabs(cos + 1 / (10 - 1)))
# print("CME = ", CME)