import numpy as np

a_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
a_2d = np.array([[1, 2, 3, 4] , [5, 6, 7, 8]])
a_3d = np.array([[[1 , 2] ,  [3 , 4]] ,  [[5 , 6] ,  [7 , 8]]])

#shape of 1d
shape1 = a_1d.shape[0]  

#shape of 2d
shape2 = a_2d.shape[0]
shape3 = a_2d.shape[1]

#shape of 3d
shape4 = a_3d.shape[0]
shape5 = a_3d.shape[1]
shape6 = a_3d.shape[2]

#1d to 2d
oneToTw = a_1d.reshape(12, 1)
print(oneToTw)
oneToTwo = a_1d.reshape(3, 4)
oneToTwoo = a_1d.reshape(4, 3)
#calculate my itself
oneToTwooo = a_1d.reshape(4, -1)

#1d to 3d
oneToThre = a_1d.reshape(2, 2, 3)
oneToThree = a_1d.reshape(2, 3, 2)
oneToThreee = a_1d.reshape(3, 2, 2)
#calculate by itself
oneToThreee = a_1d.reshape(-1, 2, 2)

#2d to 3d
twoToThre = a_2d.reshape(2, 4, 1)
twoToThree = a_2d.reshape(2, 2, 2)
print(twoToThre)

#convert anything to 1d
oneTo = a_1d.reshape(-1)
twoTo = a_2d.reshape(-1)