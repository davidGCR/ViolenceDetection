import torch
import numpy as np
from LOCALIZATION.bounding_box import BoundingBox
from LOCALIZATION.point import Point

# class Person(object):
#     def __init__(self, name, ssn):
#         self.name = name
#         self.ssn = ssn

#     def __eq__(self, other):
#         return isinstance(other, Person) and self.ssn == other.ssn and  self.name == other.name

#     def __hash__(self):
#         # use the hashcode of self.ssn since that is used
#         # for equality checks as well
#         return hash((self.ssn, self.name))

# result = []
# # p = Person('Foo Bar', 123456789)
# # q = Person('Foo Bar', 123456789)
# p = BoundingBox(Point(5, 5), Point(24, 24))
# q = BoundingBox(Point(5.3432, 6), Point(24.98, 24.87))
# r = BoundingBox(Point(5.34320,6.0), Point(24.980000,24.8700))
# result.append(p)
# result.append(q)
# result.append(r)
# result = set(result)
# # result = list(result)
# print(len(result)) # len = 2
# for r in result:
#     print(r)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()


def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

im = plt.imshow(f(x, y), animated=True)


def updatefig(*args):
    global x, y
    x += np.pi / 15.
    y += np.pi / 20.
    im.set_array(f(x, y))
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()