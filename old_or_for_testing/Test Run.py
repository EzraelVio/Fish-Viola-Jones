import numpy as np

# arr = np.array([[1, 2, 3, 4, 5], 
#                 [6, 7, 8, 9, 10],
#                 [11, 12, 13, 14, 15]])

# print(arr[1:2, 1:2])

arr2 = np.array([[1, 2, 3], 
                [4, 5, 6],
                [7, 8, 9]])

# print(np.rot90(arr2, k=3))
print (np.sum(np.triu(arr2), 0))

# value_t0_test = 1.7
# print(int(value_t0_test))
