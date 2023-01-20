import matplotlib.pyplot as plt
import numpy as np

x=np.random.randint(1,5,10)
y=2*x

plt.plot(x,y, 'red')
plt.title('Correlation')
plt.xlabel('x Values'); plt.ylabel('y values')
plt.show()
